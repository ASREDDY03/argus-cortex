"""
Argus Cortex — entry point.

Usage:
  python main.py run "Full audit of Argus Agent"
  python main.py run "Review only the ML service"
  python main.py run "goal" --dry-run    # preview findings, no PRs
  python main.py run "goal" --web        # send to web dashboard
  python main.py run "goal" --auto       # auto-approve all (CI/webhook)
  python main.py sync-prs               # sync open PR states from GitHub
  python main.py history                # list recent runs
  python main.py history <run-id>       # full findings for one run
  python main.py watch                  # live dashboard
  python main.py trends                 # week-over-week trends
  python main.py send-digest            # email weekly digest
"""
import uuid
import typer
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from tools.observability import init_tracing
from tools.github_tool import sync_pr_states
from tools.file_discovery import discover_files
from tools.slack_tool import notify_run_complete
from orchestrator.graph import build_graph
from memory.long_term import (
    start_run, finish_run, save_findings,
    mark_approved, mark_rejected, mark_pr_opened,
    get_run_history, init_db, save_pending_review, get_stats,
    get_run_findings, get_weekly_stats,
)
from tools.email_tool import send_run_digest, send_weekly_digest
from orchestrator.goal_suggester import suggest_goals
from config.settings import settings

app = typer.Typer()
console = Console()


@app.command()
def run(
    goal: str = typer.Argument(None, help="What you want Argus Cortex to do (omit to get AI suggestions)"),
    thread_id: str = typer.Option(None, help="Resume a previous run by thread ID"),
    auto_approve: bool = typer.Option(False, "--auto", help="Skip human review, approve all"),
    web_review: bool = typer.Option(False, "--web", help="Send findings to web dashboard instead of CLI review"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run agents and show findings but skip review, PRs, and DB writes"),
    min_confidence: int = typer.Option(0, "--min-confidence", help="Reject findings below this confidence (0 = keep all)"),
):
    """Run the Argus Cortex agent network."""

    init_db()

    # ── Goal Suggester — runs BEFORE the orchestrator ─────────────────────────
    # Discover files first so the suggester has file context
    agent_files = discover_files()

    if not goal:
        goal = _suggest_and_pick_goal(agent_files)
        if not goal:
            console.print("[yellow]No goal selected. Exiting.[/yellow]")
            return
    # ─────────────────────────────────────────────────────────────────────────

    tracing_enabled = init_tracing()
    thread_id = thread_id or str(uuid.uuid4())
    run_id = start_run(thread_id, goal)
    config = {"configurable": {"thread_id": thread_id}}

    _print_discovery(agent_files)

    # Sync PR outcomes from GitHub before agents run so long-term memory is current.
    # Non-blocking — a sync failure must never abort a run.
    if settings.github_token:
        with console.status("[dim]Syncing PR states from GitHub...[/dim]"):
            try:
                synced = sync_pr_states()
                if synced:
                    merged  = sum(1 for r in synced if r["state"] == "merged")
                    closed  = sum(1 for r in synced if r["state"] == "closed")
                    still_open = sum(1 for r in synced if r["state"] == "open")
                    parts = []
                    if merged:     parts.append(f"[green]{merged} merged[/green]")
                    if closed:     parts.append(f"[dim]{closed} closed[/dim]")
                    if still_open: parts.append(f"{still_open} still open")
                    console.print(f"[dim]↻ Synced {len(synced)} PR(s):[/dim] {', '.join(parts)}")
            except Exception:
                pass

    tracing_line = "[green]LangSmith tracing ON[/green]" if tracing_enabled else "[dim]LangSmith tracing OFF (add LANGCHAIN_API_KEY)[/dim]"
    dry_run_line = "\n[bold yellow]⚡ DRY RUN — no PRs will be opened, no DB writes[/bold yellow]" if dry_run else ""
    console.print(Panel(
        f"[bold cyan]Argus Cortex[/bold cyan]\n"
        f"[dim]Thread: {thread_id}[/dim]\n"
        f"[dim]Run:    {run_id}[/dim]\n"
        f"{tracing_line}{dry_run_line}\n\n"
        f"[bold]Goal:[/bold] {goal}",
        border_style="cyan" if not dry_run else "yellow"
    ))

    graph = build_graph()

    initial_state = {
        "goal": goal,
        "run_id": run_id,
        "thread_id": thread_id,
        "plan": [],
        "agents_to_run": [],
        "agent_focus": {},
        "agent_files": agent_files,
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
        "retry_count": 0,
        "suppressed_count": 0,
        "agent_tokens_in": 0,
        "agent_tokens_out": 0,
        "orch_tokens_in": 0,
        "orch_tokens_out": 0,
    }

    # ── Phase 1: Stream until interrupt (planner → agents → synthesizer → evaluator) ──
    console.print("\n[bold yellow]Phase 1:[/bold yellow] Running Planner + Generator agents + Synthesizer + Evaluator...\n")

    for chunk in graph.stream(initial_state, config=config):
        node_name, node_output = next(iter(chunk.items()))
        _stream_node(node_name, node_output)

    # Get final accumulated state from the checkpoint (graph paused at interrupt)
    state = graph.get_state(config).values

    _print_synthesis(state)
    _print_findings_table(state)

    approved = state.get("approved_findings", [])

    # Filter by minimum confidence if set
    if min_confidence > 0:
        before = len(approved)
        approved = [f for f in approved if (f.get("confidence") or 0) >= min_confidence]
        if len(approved) < before:
            console.print(f"[dim]--min-confidence {min_confidence}: filtered out {before - len(approved)} low-confidence finding(s)[/dim]")

    if not approved:
        console.print("[yellow]No findings approved by Evaluator. Nothing to PR.[/yellow]")
        if not dry_run:
            finish_run(run_id, state.get("agents_to_run", []))
        return

    # ── Dry-run exit ──────────────────────────────────────────────────────────
    if dry_run:
        cost_usd = _compute_cost(state)
        suppressed = state.get("suppressed_count", 0)
        suppressed_line = f"[dim]{suppressed} suppressed as known duplicates[/dim]\n" if suppressed else ""
        console.print(Panel(
            f"[bold yellow]Dry Run Complete[/bold yellow]\n\n"
            f"[green]{len(approved)}[/green] finding(s) would be sent for review\n"
            f"[dim]{len(state.get('rejected_findings', []))} rejected by Evaluator[/dim]\n"
            f"{suppressed_line}\n"
            f"Cost so far: [bold]${cost_usd:.4f}[/bold] (LLM calls only — no PR created)\n\n"
            f"[dim]Re-run without --dry-run to open PRs.[/dim]",
            border_style="yellow",
            title="⚡ DRY RUN",
        ))
        return
    # ─────────────────────────────────────────────────────────────────────────

    # Save findings to long-term memory
    save_findings(run_id, state.get("findings", []))
    mark_approved(run_id, approved)
    mark_rejected(run_id, state.get("rejected_findings", []))

    # ── Phase 2: Human-in-the-loop review ──
    if web_review:
        # Save pending review for the dashboard and exit — dashboard handles Phase 3
        cost_usd = _compute_cost(state)
        save_pending_review(
            run_id=run_id,
            thread_id=thread_id,
            goal=goal,
            approved_findings=approved,
            agents_run=state.get("agents_to_run", []),
            cost_usd=cost_usd,
        )
        dashboard_url = f"http://localhost:{settings.dashboard_port}/review/{run_id}"
        console.print(f"\n[bold cyan]→ Review ready:[/bold cyan] {dashboard_url}\n")
        console.print("[dim]Open the URL in your browser to approve findings and open PRs.[/dim]")
        console.print("[dim]Start the dashboard with: python dashboard/server.py[/dim]\n")
        finish_run(run_id, state.get("agents_to_run", []), cost_usd=cost_usd)
        return
    elif auto_approve:
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

    # Compute cost from accumulated token counts
    cost_usd = _compute_cost(state)
    _print_results(final_state, thread_id, goal, tracing_enabled, cost_usd)
    finish_run(
        run_id,
        state.get("agents_to_run", []),
        tokens_in=state.get("agent_tokens_in", 0) + state.get("orch_tokens_in", 0),
        tokens_out=state.get("agent_tokens_out", 0) + state.get("orch_tokens_out", 0),
        cost_usd=cost_usd,
    )

    # Extract CI summary from PR creator's summary string ("CI: ..." line)
    summary_text = final_state.get("summary", "")
    ci_line = next((l for l in summary_text.splitlines() if l.startswith("CI:")), "")
    ci_summary = ci_line.removeprefix("CI:").strip()

    notify_run_complete(
        goal=goal,
        run_id=run_id,
        approved_findings=state.get("approved_findings", []),
        agents_used=state.get("agents_to_run", []),
        pr_urls=final_state.get("pr_urls", []),
        ci_summary=ci_summary,
        retry_count=state.get("retry_count", 0),
        cost_usd=cost_usd,
    )


@app.command()
def stats():
    """Show aggregate statistics across all Argus Cortex runs."""
    init_db()
    s = get_stats()

    r  = s["runs"]
    f  = s["findings"]

    total_runs      = r["total_runs"] or 0
    completed_runs  = r["completed_runs"] or 0
    total_cost      = r["total_cost"] or 0.0
    avg_cost        = r["avg_cost"] or 0.0
    total_tokens_in = r["total_tokens_in"] or 0
    total_tokens_out= r["total_tokens_out"] or 0

    total_findings  = f["total"] or 0
    approved        = f["total_approved"] or 0
    rejected        = f["total_rejected"] or 0
    prs_opened      = f["total_prs_opened"] or 0
    approval_rate   = int(approved / max(approved + rejected, 1) * 100)

    pr_outcomes     = s["pr_outcomes"]
    merged          = pr_outcomes.get("merged", 0)
    merge_rate      = int(merged / max(prs_opened, 1) * 100) if prs_opened else 0

    trend           = s["trend"]
    last_7d         = trend["last_7d"] or 0.0
    prev_7d         = trend["prev_7d"] or 0.0
    runs_last_7d    = trend["runs_last_7d"] or 0

    if total_runs == 0:
        console.print("[dim]No runs yet — start with: python main.py run \"goal\"[/dim]")
        return

    # ── Header panel ─────────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold cyan]Argus Cortex[/bold cyan]  [dim]·[/dim]  "
        f"[bold]{total_runs}[/bold] run(s)  [dim]·[/dim]  "
        f"[bold]{total_findings}[/bold] finding(s)  [dim]·[/dim]  "
        f"[bold]${total_cost:.4f}[/bold] total cost",
        border_style="cyan",
        title="Stats",
    ))

    # ── Overall summary ───────────────────────────────────────────────────────
    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_column("Key",   style="dim", width=24)
    summary.add_column("Value", style="bold")
    summary.add_column("Key2",  style="dim", width=24)
    summary.add_column("Value2",style="bold")

    summary.add_row("Completed runs",   str(completed_runs),
                    "Approval rate",    f"{approval_rate}%")
    summary.add_row("Total findings",   str(total_findings),
                    "PR merge rate",    f"{merge_rate}% ({merged}/{prs_opened})")
    summary.add_row("Total cost",       f"${total_cost:.4f}",
                    "Avg cost / run",   f"${avg_cost:.4f}")
    summary.add_row("Tokens in",        f"{total_tokens_in:,}",
                    "Tokens out",       f"{total_tokens_out:,}")
    if runs_last_7d:
        delta = last_7d - prev_7d
        delta_str = f"[green]+${delta:.4f}[/green]" if delta >= 0 else f"[red]-${abs(delta):.4f}[/red]"
        summary.add_row("Cost last 7d",     f"${last_7d:.4f} ({runs_last_7d} run(s))",
                        "vs prior 7d",      delta_str)

    console.print("\n[bold]Overall[/bold]")
    console.print(summary)

    # ── Findings by severity ──────────────────────────────────────────────────
    console.print("\n[bold]Findings by Severity[/bold]")
    sev_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    sev_table.add_column("Severity",  width=10)
    sev_table.add_column("Count",     justify="right", width=7)
    sev_table.add_column("Bar",       width=16, no_wrap=True)
    sev_table.add_column("Approved",  justify="right", width=10)
    sev_table.add_column("Rate",      justify="right", width=7)

    sev_emoji  = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
    sev_color  = {"critical": "red", "high": "orange3", "medium": "yellow", "low": "green"}
    max_sev    = max((row["total"] for row in s["by_severity"]), default=1)

    for row in s["by_severity"]:
        sev   = row["severity"]
        tot   = row["total"] or 0
        appr  = row["approved"] or 0
        rate  = int(appr / max(tot, 1) * 100)
        color = sev_color.get(sev, "white")
        bar   = _bar(tot, max_sev, width=14)
        sev_table.add_row(
            f"{sev_emoji.get(sev, '⚪')} [{color}]{sev}[/{color}]",
            f"[bold]{tot}[/bold]",
            f"[{color}]{bar}[/{color}]",
            str(appr),
            f"{rate}%",
        )
    console.print(sev_table)

    # ── Findings by agent ─────────────────────────────────────────────────────
    console.print("\n[bold]Findings by Agent[/bold]")
    agent_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    agent_table.add_column("Agent",    width=26)
    agent_table.add_column("Found",    justify="right", width=7)
    agent_table.add_column("Approved", justify="right", width=10)
    agent_table.add_column("Rejected", justify="right", width=10)
    agent_table.add_column("Rate",     justify="right", width=7)

    for row in s["by_agent"]:
        tot  = row["total"] or 0
        appr = row["approved"] or 0
        rej  = row["rejected"] or 0
        rate = int(appr / max(tot, 1) * 100)
        style = "red" if "security" in row["agent"] else ""
        agent_table.add_row(
            f"[{style}]{row['agent']}[/{style}]" if style else row["agent"],
            str(tot),
            f"[green]{appr}[/green]",
            f"[dim]{rej}[/dim]",
            f"[bold]{rate}%[/bold]",
        )
    console.print(agent_table)

    # ── Findings by category ──────────────────────────────────────────────────
    console.print("\n[bold]Findings by Category[/bold]")
    cat_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    cat_table.add_column("Category",  width=16)
    cat_table.add_column("Total",     justify="right", width=7)
    cat_table.add_column("Bar",       width=18, no_wrap=True)
    cat_table.add_column("Approved",  justify="right", width=10)

    max_cat = max((row["total"] for row in s["by_category"]), default=1)
    for row in s["by_category"]:
        tot  = row["total"] or 0
        appr = row["approved"] or 0
        cat_table.add_row(
            row["category"],
            str(tot),
            f"[cyan]{_bar(tot, max_cat, 16)}[/cyan]",
            str(appr),
        )
    console.print(cat_table)

    # ── PR outcomes ───────────────────────────────────────────────────────────
    if prs_opened:
        console.print("\n[bold]PR Outcomes[/bold]")
        pr_parts = []
        for state_name, color in [("merged", "green"), ("open", "cyan"), ("closed", "dim")]:
            count = pr_outcomes.get(state_name, 0)
            if count:
                pr_parts.append(f"[{color}]{state_name}: {count}[/{color}]")
        console.print("  " + "  ·  ".join(pr_parts))

    # ── Top flagged files ─────────────────────────────────────────────────────
    if s["top_files"]:
        console.print("\n[bold]Most Flagged Files[/bold]")
        file_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        file_table.add_column("File",     width=40)
        file_table.add_column("Findings", justify="right", width=10)
        file_table.add_column("Approved", justify="right", width=10)

        for row in s["top_files"]:
            from pathlib import Path as _Path
            short = _Path(row["file"]).name
            file_table.add_row(short, str(row["total"] or 0), str(row["approved"] or 0))
        console.print(file_table)

    console.print()


@app.command(name="sync-prs")
def sync_prs():
    """Sync open Cortex PR states from GitHub and update long-term memory."""
    init_db()

    if not settings.github_token:
        console.print("[red]GITHUB_TOKEN not set in .env — cannot reach GitHub.[/red]")
        raise typer.Exit(1)

    with console.status("[bold cyan]Fetching PR states from GitHub...[/bold cyan]"):
        results = sync_pr_states()

    if not results:
        console.print("[dim]No open Cortex PRs found in long-term memory. Nothing to sync.[/dim]")
        return

    _STATE_STYLE = {
        "merged": ("green",  "✔ merged"),
        "closed": ("dim",    "✘ closed"),
        "open":   ("cyan",   "● open"),
        "":       ("red",    "! error"),
    }

    table = Table(title="PR State Sync", show_header=True, header_style="bold cyan", box=None)
    table.add_column("PR #",   width=6,  justify="right")
    table.add_column("State",  width=12)
    table.add_column("Title",  min_width=30)
    table.add_column("URL",    style="dim")

    merged_count = closed_count = open_count = error_count = 0

    for r in results:
        state = r["state"]
        style, label = _STATE_STYLE.get(state, ("red", "! error"))
        error = r.get("error", "")

        if error:
            error_count += 1
            table.add_row("—", f"[red]error[/red]", f"[red]{error[:60]}[/red]", r["pr_url"])
        else:
            pr_num = str(r["pr_number"]) if r["pr_number"] else "—"
            title  = r["title"][:55] + ("…" if len(r["title"]) > 55 else "")
            table.add_row(pr_num, f"[{style}]{label}[/{style}]", title, r["pr_url"])
            if state == "merged":  merged_count  += 1
            elif state == "closed": closed_count += 1
            else:                   open_count   += 1

    console.print(table)

    # Summary line
    parts = []
    if merged_count:  parts.append(f"[green]{merged_count} merged[/green]")
    if closed_count:  parts.append(f"[dim]{closed_count} closed[/dim]")
    if open_count:    parts.append(f"[cyan]{open_count} still open[/cyan]")
    if error_count:   parts.append(f"[red]{error_count} error(s)[/red]")
    console.print(f"\n[bold]Result:[/bold] {len(results)} PR(s) checked — {', '.join(parts)}")

    if merged_count or closed_count:
        console.print(
            "[dim]Merged/closed PRs are now excluded from deduplication — "
            "agents will report regressions on those files.[/dim]"
        )


@app.command()
def watch():
    """Live terminal dashboard — polls DB every 2s showing active runs and findings."""
    import time
    from rich.live import Live

    init_db()

    def _render() -> Table:
        runs = get_run_history(limit=8)
        table = Table(
            title=f"Argus Cortex — Live  [dim]{datetime.utcnow().strftime('%H:%M:%S UTC')}[/dim]",
            show_header=True, header_style="bold cyan", box=None, expand=True,
        )
        table.add_column("Run ID",  width=10)
        table.add_column("Goal",    min_width=30)
        table.add_column("Status",  width=12)
        table.add_column("Cost",    width=10, justify="right")
        table.add_column("Started", width=17)

        for r in runs:
            cost = r.get("cost_usd") or 0.0
            status = r["status"]
            if status == "running":
                status_str = "[bold yellow]● running[/bold yellow]"
            elif status == "completed":
                status_str = "[green]✔ done[/green]"
            else:
                status_str = f"[dim]{status}[/dim]"

            table.add_row(
                r["id"][:8],
                r["goal"][:52],
                status_str,
                f"${cost:.4f}" if cost else "—",
                r["created_at"][:16],
            )
        return table

    console.print("[dim]Watching Argus Cortex — Ctrl+C to exit[/dim]\n")
    try:
        with Live(console=console, refresh_per_second=0.5) as live:
            while True:
                live.update(_render())
                time.sleep(2)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")


@app.command()
def trends():
    """Show week-over-week finding counts, approval rates, and cost for the last 8 weeks."""
    init_db()
    weeks = get_weekly_stats(weeks=8)

    if not weeks:
        console.print("[dim]Not enough data yet — run a few audits first.[/dim]")
        return

    max_findings = max((w["findings"] or 0 for w in weeks), default=1)

    table = Table(
        title="Trends — last 8 weeks",
        show_header=True, header_style="bold cyan", box=None,
    )
    table.add_column("Week",      width=10)
    table.add_column("Runs",      width=6,  justify="right")
    table.add_column("Findings",  width=10, justify="right")
    table.add_column("Bar",       width=18, no_wrap=True)
    table.add_column("Approved",  width=10, justify="right")
    table.add_column("Rate",      width=7,  justify="right")
    table.add_column("Cost",      width=10, justify="right")

    for w in weeks:
        found  = w["findings"] or 0
        appr   = w["approved"] or 0
        rate   = int(appr / max(found, 1) * 100)
        bar    = _bar(found, max_findings, width=16)
        table.add_row(
            w["week"],
            str(w["runs"] or 0),
            str(found),
            f"[cyan]{bar}[/cyan]",
            str(appr),
            f"{rate}%",
            f"${w['cost_usd']:.4f}",
        )

    console.print(table)

    # Week-over-week delta for latest two weeks
    if len(weeks) >= 2:
        latest = weeks[-1]["findings"] or 0
        prior  = weeks[-2]["findings"] or 0
        delta  = latest - prior
        arrow  = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        color  = "red" if delta > 0 else ("green" if delta < 0 else "dim")
        console.print(f"\n[dim]Latest vs prior week:[/dim] [{color}]{arrow} {abs(delta)} finding(s)[/{color}]")


@app.command(name="send-digest")
def send_digest_cmd():
    """Send a weekly aggregate digest email (requires SMTP_* in .env)."""
    init_db()

    if not settings.smtp_host:
        console.print("[red]SMTP not configured — set SMTP_HOST, SMTP_USER, SMTP_PASSWORD, DIGEST_TO in .env[/red]")
        raise typer.Exit(1)

    with console.status("[bold cyan]Generating digest...[/bold cyan]"):
        s = get_stats()
        ok = send_weekly_digest(s)

    if ok:
        console.print(f"[green]✔[/green] Digest sent to [bold]{settings.digest_to}[/bold]")
    else:
        console.print("[red]Failed to send digest — check SMTP settings and logs.[/red]")
        raise typer.Exit(1)


@app.command()
def history(
    run_id: str = typer.Argument(None, help="Run ID to inspect (shows all findings for that run)"),
):
    """Show recent runs, or all findings for a specific run."""
    init_db()

    # ── Full detail for one run ───────────────────────────────────────────────
    if run_id:
        findings = get_run_findings(run_id)
        if not findings:
            console.print(f"[red]No findings found for run {run_id!r}. Check the run ID.[/red]")
            raise typer.Exit(1)

        _SEV_COLOR = {"critical": "red", "high": "orange3", "medium": "yellow", "low": "green"}
        _SEV_EMOJI = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}

        table = Table(
            title=f"Findings — run {run_id[:8]}",
            show_header=True, header_style="bold cyan", box=None,
        )
        table.add_column("Sev",        width=9)
        table.add_column("Category",   width=13)
        table.add_column("Agent",      width=20)
        table.add_column("File:Line",  width=34)
        table.add_column("Conf",       width=6, justify="right")
        table.add_column("Status",     width=10)
        table.add_column("PR",         width=8)

        for f in findings:
            sev   = f.get("severity", "low")
            color = _SEV_COLOR.get(sev, "white")
            emoji = _SEV_EMOJI.get(sev, "⚪")
            file_line = f.get("file", "")
            if f.get("line"):
                file_line += f":{f['line']}"
            file_line = file_line[-32:] if len(file_line) > 32 else file_line

            conf = f.get("confidence")
            conf_str = f"{conf}%" if conf is not None else "—"
            if conf is not None:
                conf_color = "green" if conf >= 80 else "yellow" if conf >= 60 else "red"
                conf_str = f"[{conf_color}]{conf}%[/{conf_color}]"

            if f.get("approved"):
                status = "[green]approved[/green]"
            elif f.get("rejected"):
                status = "[dim]rejected[/dim]"
            else:
                status = "[dim]pending[/dim]"

            pr_state = f.get("pr_state") or ("open" if f.get("pr_url") else "")
            pr_str = (
                f"[green]merged[/green]" if pr_state == "merged" else
                f"[dim]closed[/dim]"    if pr_state == "closed" else
                f"[cyan]open[/cyan]"    if pr_state == "open"   else
                "[dim]—[/dim]"
            )

            table.add_row(
                f"{emoji} [{color}]{sev}[/{color}]",
                f.get("category", ""),
                f.get("agent", ""),
                file_line,
                conf_str,
                status,
                pr_str,
            )

        console.print(table)
        console.print(f"\n[dim]{len(findings)} finding(s) total for run {run_id[:8]}[/dim]")
        return

    # ── Recent runs list ──────────────────────────────────────────────────────
    runs = get_run_history(limit=10)
    if not runs:
        console.print("[dim]No runs yet.[/dim]")
        return

    table = Table(title="Recent Runs", show_header=True, header_style="bold cyan")
    table.add_column("Run ID")
    table.add_column("Date")
    table.add_column("Goal")
    table.add_column("Status")
    table.add_column("Cost", justify="right")

    for r in runs:
        cost = r.get("cost_usd") or 0.0
        cost_str = f"${cost:.4f}" if cost else "—"
        table.add_row(
            r["id"][:8],
            r["created_at"][:16],
            r["goal"][:55],
            r["status"],
            cost_str,
        )
    console.print(table)
    console.print("[dim]Tip: python main.py history <run-id> to see all findings for a run[/dim]")


# ── Helpers ──────────────────────────────────────────────────────────────────

_SEVERITY_COLOR = {"critical": "red", "high": "orange3", "medium": "yellow", "low": "green"}


def _bar(value: int, total: int, width: int = 14) -> str:
    """Unicode block bar chart — e.g. ████████░░░░"""
    if total == 0:
        return "░" * width
    filled = int(round(value / total * width))
    return "█" * filled + "░" * (width - filled)
_GENERATOR_AGENTS = {"springboot_agent", "ml_agent", "react_agent", "infra_agent", "observability_agent", "jenkins_agent", "security_agent", "dependency_agent"}

# Anthropic pricing (USD per million tokens)
_COST_PER_M = {
    settings.orchestrator_model: {"input": 3.00, "output": 15.00},  # claude-sonnet-4-6
    settings.agent_model:        {"input": 0.80, "output":  4.00},  # claude-haiku-4-5
}


def _compute_cost(state: dict) -> float:
    """Compute total run cost from accumulated token counts in state."""
    sonnet = _COST_PER_M[settings.orchestrator_model]
    haiku  = _COST_PER_M[settings.agent_model]
    orch_cost  = (state.get("orch_tokens_in", 0)  * sonnet["input"]  +
                  state.get("orch_tokens_out", 0) * sonnet["output"]) / 1_000_000
    agent_cost = (state.get("agent_tokens_in", 0)  * haiku["input"]  +
                  state.get("agent_tokens_out", 0) * haiku["output"]) / 1_000_000
    return round(orch_cost + agent_cost, 6)


def _stream_node(node_name: str, output: dict):
    """Print a one-line status update as each graph node completes."""
    if node_name == "planner":
        agents = output.get("agents_to_run", [])
        plan = output.get("plan", [])
        console.print(f"[green]✓[/green] [bold]Planner[/bold] — {len(agents)} agent(s) selected")
        for step in plan:
            console.print(f"  [dim]{step}[/dim]")

    elif node_name in _GENERATOR_AGENTS:
        findings = output.get("findings", [])
        is_security = node_name == "security_agent"
        is_dependency = node_name == "dependency_agent"
        if is_security:
            label = f"[bold red]{node_name}[/bold red]"
        elif is_dependency:
            label = f"[bold yellow]{node_name}[/bold yellow]"
        else:
            label = f"[bold]{node_name}[/bold]"
        if findings:
            counts: dict[str, int] = {}
            for f in findings:
                sev = f.get("severity", "low")
                counts[sev] = counts.get(sev, 0) + 1
            parts = [
                f"[{_SEVERITY_COLOR.get(s, 'white')}]{c} {s}[/{_SEVERITY_COLOR.get(s, 'white')}]"
                for s in ("critical", "high", "medium", "low") if s in counts
                for c in [counts[s]]
            ]
            if is_security:
                prefix = "[red]🔒[/red] "
            elif is_dependency:
                prefix = "[yellow]📦[/yellow] "
            else:
                prefix = "[green]✓[/green] "
            suppressed = output.get("suppressed_count", 0)
            dedup_note = f"  [dim]({suppressed} suppressed as known)[/dim]" if suppressed else ""
            console.print(f"{prefix}{label} — {len(findings)} finding(s): {', '.join(parts)}{dedup_note}")
        else:
            suppressed = output.get("suppressed_count", 0)
            dedup_note = f"  [dim]({suppressed} suppressed as known)[/dim]" if suppressed else ""
            console.print(f"[green]✓[/green] {label} — [dim]no findings[/dim]{dedup_note}")

    elif node_name == "synthesizer":
        cross = output.get("cross_cutting_issues", [])
        gaps = output.get("coverage_gaps", [])
        retry = output.get("retry_agents", {})
        parts = []
        if cross:
            parts.append(f"{len(cross)} cross-cutting pattern(s)")
        if gaps:
            parts.append(f"{len(gaps)} coverage gap(s)")
        if retry:
            parts.append(f"[cyan]↻ recommends retry: {', '.join(retry.keys())}[/cyan]")
        detail = ", ".join(parts) if parts else "no patterns or gaps"
        console.print(f"[green]✓[/green] [bold magenta]Synthesizer[/bold magenta] — {detail}")

    elif node_name == "retry_dispatcher":
        agents = output.get("agents_to_run", [])
        count = output.get("retry_count", 1)
        console.print(
            f"[cyan]↻[/cyan] [bold cyan]Retry #{count}[/bold cyan] — "
            f"re-running: {', '.join(agents) if agents else 'none'}"
        )

    elif node_name == "evaluator":
        approved = output.get("approved_findings", [])
        rejected = output.get("rejected_findings", [])
        console.print(
            f"[green]✓[/green] [bold]Evaluator[/bold] — "
            f"[green]{len(approved)} approved[/green]  "
            f"[dim]{len(rejected)} rejected[/dim]"
        )



def _print_discovery(agent_files: dict):
    if not agent_files:
        console.print("[dim]File discovery skipped (ARGUS_REPO_PATH not set — agents use fallback lists)[/dim]\n")
        return

    table = Table(title="Discovered Files", show_header=True, header_style="bold cyan", box=None)
    table.add_column("Agent", width=22)
    table.add_column("Files", style="dim")

    for agent, files in agent_files.items():
        if files:
            from pathlib import Path
            names = ", ".join(Path(f).name for f in files[:5])
            suffix = f" +{len(files) - 5} more" if len(files) > 5 else ""
            table.add_row(agent, f"{names}{suffix}")
        else:
            table.add_row(agent, "[dim]none found — fallback list will be used[/dim]")

    console.print(table)
    console.print()


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


def _print_results(state: dict, thread_id: str, goal: str, tracing_enabled: bool = False, cost_usd: float = 0.0):
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

    suppressed = state.get("suppressed_count", 0)
    if suppressed:
        console.print(f"[dim]{suppressed} finding(s) suppressed as known duplicates (open PR or prior approval)[/dim]")
    approved = state.get("approved_findings", [])
    confidences = [f.get("confidence") for f in approved if f.get("confidence") is not None]
    if confidences:
        avg_conf = int(sum(confidences) / len(confidences))
        conf_color = "green" if avg_conf >= 80 else "yellow" if avg_conf >= 60 else "red"
        console.print(f"[dim]Avg confidence: [{conf_color}]{avg_conf}%[/{conf_color}][/dim]")
    if cost_usd > 0:
        console.print(f"[dim]Cost: ${cost_usd:.4f} USD[/dim]")
    console.print(f"\n[dim]Resume this run: python main.py run '{goal}' --thread-id {thread_id}[/dim]")
    if tracing_enabled:
        console.print(f"[dim]LangSmith traces: https://smith.langchain.com/projects/p/{settings.langchain_project}[/dim]\n")
    else:
        console.print()


def _suggest_and_pick_goal(agent_files: dict) -> str | None:
    """
    Display AI-suggested goals, let the user pick one or type their own.
    Returns the chosen goal string, or None if the user exits.
    """
    recent_runs = get_run_history(limit=6)

    console.print(Panel(
        "[bold cyan]Goal Suggester[/bold cyan]\n\n"
        "[dim]Analysing the Argus Agent repo to suggest audit goals…[/dim]",
        border_style="cyan",
    ))

    with console.status("[bold cyan]Generating suggestions…[/bold cyan]"):
        suggestions = suggest_goals(agent_files, recent_runs)

    if not suggestions:
        console.print("[red]Could not generate suggestions.[/red]")
        goal = Prompt.ask("\nType your goal manually (or press Enter to exit)", default="")
        return goal.strip() or None

    # Display numbered list
    console.print("\n[bold]Suggested goals:[/bold]\n")
    for i, s in enumerate(suggestions):
        console.print(f"  [bold cyan]{i + 1}.[/bold cyan] {s}")

    console.print(f"\n  [bold cyan]{len(suggestions) + 1}.[/bold cyan] [dim]Type my own goal[/dim]")
    console.print(f"  [bold cyan]{len(suggestions) + 2}.[/bold cyan] [dim]Exit[/dim]")

    while True:
        raw = Prompt.ask(
            f"\nPick a number [1-{len(suggestions) + 2}]",
            default="1",
        )
        if not raw.strip().isdigit():
            console.print("[yellow]Enter a number.[/yellow]")
            continue
        choice = int(raw.strip())
        if 1 <= choice <= len(suggestions):
            goal = suggestions[choice - 1]
            console.print(f"\n[green]✓[/green] Selected: [bold]{goal}[/bold]\n")
            return goal
        elif choice == len(suggestions) + 1:
            goal = Prompt.ask("Your goal").strip()
            if goal:
                return goal
            console.print("[yellow]Goal cannot be empty.[/yellow]")
        elif choice == len(suggestions) + 2:
            return None
        else:
            console.print(f"[yellow]Enter a number between 1 and {len(suggestions) + 2}.[/yellow]")


def _agent_from_pr_url(url: str) -> str:
    """Extract agent name from PR URL (best effort)."""
    for agent in ["springboot_agent", "ml_agent", "react_agent", "infra_agent", "observability_agent", "jenkins_agent", "security_agent", "dependency_agent"]:
        if agent.replace("_", "-") in url or agent in url:
            return agent
    return "unknown"


if __name__ == "__main__":
    app()
