"""
Argus Cortex — entry point.

Usage:
  python main.py "Full audit of Argus Agent"
  python main.py "Review only the ML service and infrastructure"
"""
import uuid
import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from orchestrator.graph import build_graph

app = typer.Typer()
console = Console()


@app.command()
def run(
    goal: str = typer.Argument(..., help="What you want Argus Cortex to do"),
    thread_id: str = typer.Option(None, help="Resume a previous run by thread ID"),
):
    """Run the Argus Cortex agent network against Argus Agent."""

    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    console.print(f"\n[bold cyan]Argus Cortex[/bold cyan]")
    console.print(f"[dim]Thread: {thread_id}[/dim]")
    console.print(f"[bold]Goal:[/bold] {goal}\n")

    graph = build_graph()

    initial_state = {
        "goal": goal,
        "plan": [],
        "agents_to_run": [],
        "findings": [],
        "approved_findings": [],
        "rejected_findings": [],
        "evaluation_notes": "",
        "pr_urls": [],
        "summary": "",
        "messages": [],
        "error": None,
    }

    with console.status("[bold green]Running agent network..."):
        final_state = graph.invoke(initial_state, config=config)

    # Print plan
    console.print("\n[bold yellow]Plan[/bold yellow]")
    for i, step in enumerate(final_state.get("plan", []), 1):
        console.print(f"  {i}. {step}")

    # Print approved findings
    approved = final_state.get("approved_findings", [])
    console.print(f"\n[bold green]Approved Findings ({len(approved)})[/bold green]")

    if approved:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Agent")
        table.add_column("Severity")
        table.add_column("File")
        table.add_column("Description")

        for f in approved:
            table.add_row(
                f.get("agent", ""),
                f.get("severity", ""),
                f.get("file", "").split("/")[-1],
                f.get("description", "")[:80],
            )
        console.print(table)

    # Print rejected findings count
    rejected = final_state.get("rejected_findings", [])
    if rejected:
        console.print(f"[dim]Rejected by Evaluator: {len(rejected)} (too vague or incorrect)[/dim]")

    # Print evaluator notes
    notes = final_state.get("evaluation_notes", "")
    if notes:
        console.print(f"\n[bold]Evaluator Notes:[/bold] {notes}")

    # Print PRs opened
    pr_urls = final_state.get("pr_urls", [])
    if pr_urls:
        console.print(f"\n[bold green]PRs Opened ({len(pr_urls)})[/bold green]")
        for url in pr_urls:
            console.print(f"  → {url}")
    else:
        console.print("\n[yellow]No PRs opened — add GITHUB_TOKEN to .env to enable.[/yellow]")

    # Print summary
    summary = final_state.get("summary", "")
    if summary:
        console.print(f"\n[bold]Summary:[/bold]\n{summary}")

    console.print(f"\n[dim]To resume this run: python main.py '{goal}' --thread-id {thread_id}[/dim]\n")


if __name__ == "__main__":
    app()
