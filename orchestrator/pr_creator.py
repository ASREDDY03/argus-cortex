"""
PR Creator node — runs after Human Review.
Sends ALL approved findings from ALL agents to ONE consolidated branch + PR.
"""
from memory.state import CortexState
from tools.github_tool import create_consolidated_pr
from rich.console import Console

console = Console()


def run_pr_creator(state: CortexState) -> dict:
    """LangGraph node: PR Creator."""
    approved = state.get("approved_findings", [])
    goal = state.get("goal", "")
    run_id = state.get("run_id", "unknown")
    human_notes = state.get("human_notes", "")

    if not approved:
        console.print("[yellow]No approved findings — no PR to open.[/yellow]")
        return {"pr_urls": [], "summary": "No findings approved."}

    if not _github_configured():
        console.print("[yellow]GITHUB_TOKEN not set — skipping PR creation.[/yellow]")
        return {
            "pr_urls": [],
            "summary": f"{len(approved)} findings approved but GITHUB_TOKEN not configured.",
        }

    agents = sorted({f.get("agent", "") for f in approved})
    pr_ready_count = sum(1 for f in approved if f.get("pr_ready"))

    console.print(f"[cyan]Creating consolidated branch for {len(agents)} agent(s), {pr_ready_count} patch(es)...[/cyan]")

    generated_tests = state.get("generated_tests", [])

    try:
        result = create_consolidated_pr(run_id, approved, goal, human_notes, generated_tests)
        url = result.get("pr_url")
        ci_passed = result.get("ci_passed")
        ci_summary = result.get("ci_summary", "")

        # Show CI result
        if ci_summary:
            if ci_passed is None:
                console.print(f"[dim]CI: {ci_summary}[/dim]")
            elif ci_passed:
                console.print(f"[green]✓ CI: {ci_summary}[/green]")
            else:
                console.print(f"[red]✗ CI: {ci_summary} — PR not opened[/red]")
                for check in result.get("ci_checks", []):
                    if check["conclusion"] != "success":
                        console.print(f"  [red]✗[/red] {check['name']}: {check['conclusion']} → {check['url']}")
                return {
                    "pr_urls": [],
                    "summary": f"CI failed — {ci_summary}. Branch '{run_id[:8]}' is available for inspection.",
                }

        if url:
            console.print(f"[green]✓ Consolidated PR opened: {url}[/green]")
            summary = (
                f"Argus Cortex completed.\n"
                f"Approved findings: {len(approved)}\n"
                f"Agents: {', '.join(agents)}\n"
                f"CI: {ci_summary or 'not configured'}\n"
                f"PR: {url}"
            )
            return {"pr_urls": [url], "summary": summary}
        else:
            console.print("[yellow]No pr_ready findings — no PR opened.[/yellow]")
            return {"pr_urls": [], "summary": "No pr_ready findings to commit."}
    except Exception as e:
        console.print(f"[red]✗ Failed to open PR: {e}[/red]")
        return {"pr_urls": [], "summary": f"PR creation failed: {e}"}


def _github_configured() -> bool:
    from config.settings import settings
    return bool(settings.github_token)
