"""
Argus Cortex — entry point.

Usage:
  python main.py run "Full audit of Argus Agent"
  python main.py run "Review only the ML service"
  python main.py history
"""
import uuid
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from tools.observability import init_tracing
from tools.github_tool import sync_pr_states
from orchestrator.graph import build_graph
from memory.long_term import (
    start_run, finish_run, save_findings,
    mark_approved, mark_rejected, mark_pr_opened,
    get_run_history, init_db,
)
from config.settings import settings

app = typer.Typer()
console = Console()


@app.command()
def run(
    goal: str = typer.Argument(..., help="What you want Argus Cortex to do"),
    thread_id: str = typer.Option(None, help="Resume a previous run by thread ID"),
    auto_approve: bool = typer.Option(False, "--auto", help="Skip human review, approve all"),
):
    """Run the Argus Cortex agent network."""

    init_db()
    tracing_enabled = init_tracing()
    thread_id = thread_id or str(uuid.uuid4())
    run_id = start_run(thread_id, goal)
    config = {"configurable": {"thread_id": thread_id}}

    # Sync PR outcomes from GitHub before agents run so long-term memory is current
    if settings.github_token:
        with console.status("[dim]Syncing PR states from GitHub...[/dim]"):
            try:
                synced = sync_pr_states()
                if synced:
                    console.print(f"[dim]↻ Synced {synced} PR(s) from GitHub[/dim]")
            except Exception:
                pass  # non-blocking — a sync failure must never abort a run

    tracing_line = "[green]LangSmith tracing ON[/green]" if tracing_enabled else "[dim]LangSmith tracing OFF (add LANGCHAIN_API_KEY)[/dim]"
    console.print(Panel(
        f"[bold cyan]Argus Cortex[/bold cyan]\n"
        f"[dim]Thread: {thread_id}[/dim]\n"
        f"[dim]Run:    {run_id}[/dim]\n"
        f"{tracing_line}\n\n"
        f"[bold]Goal:[/bold] {goal}",
        border_style="cyan"
    ))

    graph = build_graph()

    initial_state = {
        "goal": goal,
        "run_id": run_id,
        "thread_id": thread_id,
        "plan": [],
        "agents_to_run": [],
        "agent_focus": {},
        "findings": [],
        "cross_cutting_issues": [],
        "coverage_gaps": [],
        "synthesis_notes": "",
        "retry_agents": {},
        "approved_findings": [],
        "rejected_findings": [],
        "evaluation_notes": "",
        "human_approved_indices": [],
        "human_notes": "",
        "pr_urls": [],
        "summary": "",
        "messages": [],
        "error": None,
    }

    # ── Phase 1: Run until interrupt (planner → agents → evaluator) ──
    console.print("\n[bold yellow]Phase 1:[/bold yellow] Running Planner + Generator agents + Evaluator...\n")

    with console.status("[bold green]Agents working..."):
        state = graph.invoke(initial_state, config=config)

    _print_plan(state)
    _print_synthesis(state)
    _print_findings_table(state)

    approved = state.get("approved_findings", [])
    if not approved:
        console.print("[yellow]No findings approved by Evaluator. Nothing to PR.[/yellow]")
        finish_run(run_id, state.get("agents_to_run", []))
        return

    # Save findings to long-term memory
    save_findings(run_id, state.get("findings", []))
    mark_approved(run_id, approved)
    mark_rejected(run_id, state.get("rejected_findings", []))

    # ── Phase 2: Human-in-the-loop review ──
    if auto_approve:
        human_approved_indices = list(range(len(approved)))
        human_notes = "Auto-approved"
        console.print("[dim]--auto flag set: skipping human review, approving all.[/dim]")
    else:
        human_approved_indices, human_notes = _human_review(approved)

    if not human_approved_indices:
        console.print("[yellow]No findings selected. Exiting without opening PRs.[/yellow]")
        finish_run(run_id, state.get("agents_to_run", []))
        return

    # ── Phase 3: Resume graph → human_review node → pr_creator ──
    console.print("\n[bold yellow]Phase 3:[/bold yellow] Opening PRs...\n")

    resume_state = {
        "human_approved_indices": human_approved_indices,
        "human_notes": human_notes,
    }

    with console.status("[bold green]Creating PRs on GitHub..."):
        final_state = graph.invoke(resume_state, config=config)

    # Mark PRs in long-term memory
    for url in final_state.get("pr_urls", []):
        agent = _agent_from_pr_url(url)
        mark_pr_opened(run_id, agent, url)

    _print_results(final_state, thread_id, goal)
    finish_run(run_id, state.get("agents_to_run", []))


@app.command()
def history():
    """Show recent Argus Cortex runs."""
    init_db()
    runs = get_run_history(limit=10)
    if not runs:
        console.print("[dim]No runs yet.[/dim]")
        return

    table = Table(title="Recent Runs", show_header=True, header_style="bold cyan")
    table.add_column("Date")
    table.add_column("Goal")
    table.add_column("Status")
    table.add_column("Thread ID")

    for r in runs:
        table.add_row(
            r["created_at"][:16],
            r["goal"][:60],
            r["status"],
            r["thread_id"][:12] + "...",
        )
    console.print(table)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _print_plan(state: dict):
    plan = state.get("plan", [])
    if plan:
        console.print("\n[bold yellow]Plan[/bold yellow]")
        for i, step in enumerate(plan, 1):
            console.print(f"  {i}. {step}")


def _print_synthesis(state: dict):
    cross = state.get("cross_cutting_issues", [])
    gaps = state.get("coverage_gaps", [])
    notes = state.get("synthesis_notes", "")
    retry = state.get("retry_agents", {})

    if not (cross or gaps or notes):
        return

    console.print("\n[bold magenta]Synthesizer[/bold magenta]")

    if notes:
        console.print(f"  [dim]{notes}[/dim]")

    if cross:
        console.print("\n  [bold]Cross-cutting patterns[/bold]")
        for c in cross:
            console.print(f"  [magenta]•[/magenta] {c}")

    if gaps:
        console.print("\n  [bold]Coverage gaps[/bold]")
        for g in gaps:
            console.print(f"  [yellow]•[/yellow] {g}")

    if retry:
        console.print("\n  [bold]Recommended re-runs[/bold]")
        for agent, focus in retry.items():
            console.print(f"  [cyan]↻[/cyan] {agent}: {focus}")


def _print_findings_table(state: dict):
    approved = state.get("approved_findings", [])
    rejected = state.get("rejected_findings", [])
    notes = state.get("evaluation_notes", "")

    severity_color = {"critical": "red", "high": "orange3", "medium": "yellow", "low": "green"}

    console.print(f"\n[bold green]Approved by Evaluator ({len(approved)})[/bold green]")
    if approved:
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", width=3)
        table.add_column("Agent", width=20)
        table.add_column("Severity", width=10)
        table.add_column("File", width=25)
        table.add_column("Description")

        for i, f in enumerate(approved):
            sev = f.get("severity", "low")
            color = severity_color.get(sev, "white")
            table.add_row(
                str(i),
                f.get("agent", ""),
                f"[{color}]{sev}[/{color}]",
                f.get("file", "").split("/")[-1],
                f.get("description", "")[:70],
            )
        console.print(table)

    if rejected:
        console.print(f"[dim]Rejected by Evaluator: {len(rejected)} (vague or incorrect)[/dim]")

    if notes:
        console.print(f"[dim]Evaluator notes: {notes}[/dim]")


def _human_review(approved: list[dict]) -> tuple[list[int], str]:
    """Interactive human review — returns (approved_indices, notes)."""

    console.print(Panel(
        "[bold]Human Review[/bold]\n\n"
        "The Evaluator approved the findings above.\n"
        "You decide which ones to open PRs for.",
        border_style="yellow",
    ))

    approve_all = Confirm.ask("\nApprove ALL findings and open PRs?", default=True)

    if approve_all:
        notes = Prompt.ask("Any notes? (optional, press Enter to skip)", default="")
        return list(range(len(approved))), notes

    # Individual selection
    console.print("\nEnter the finding numbers to approve (comma-separated, e.g. 0,2,3):")
    raw = Prompt.ask("Findings to approve")
    indices = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 0 <= idx < len(approved):
                indices.append(idx)

    notes = Prompt.ask("Any notes? (optional)", default="")
    return indices, notes


def _print_results(state: dict, thread_id: str, goal: str):
    pr_urls = state.get("pr_urls", [])
    summary = state.get("summary", "")

    if pr_urls:
        console.print(f"\n[bold green]PRs Opened ({len(pr_urls)})[/bold green]")
        for url in pr_urls:
            console.print(f"  → {url}")
    else:
        console.print("\n[yellow]No PRs opened — add GITHUB_TOKEN to .env to enable.[/yellow]")

    if summary:
        console.print(Panel(summary, title="Summary", border_style="green"))

    console.print(f"\n[dim]Resume this run: python main.py run '{goal}' --thread-id {thread_id}[/dim]")
    if tracing_enabled:
        console.print(f"[dim]LangSmith traces: https://smith.langchain.com/o/projects/{settings.langchain_project}[/dim]\n")
    else:
        console.print()


def _agent_from_pr_url(url: str) -> str:
    """Extract agent name from PR URL (best effort)."""
    for agent in ["springboot_agent", "ml_agent", "react_agent", "infra_agent", "observability_agent"]:
        if agent.replace("_", "-") in url or agent in url:
            return agent
    return "unknown"


if __name__ == "__main__":
    app()
