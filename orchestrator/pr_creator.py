"""
PR Creator node — runs after Evaluator.
Groups approved findings by agent, opens one draft PR per agent domain.
"""
from collections import defaultdict
from memory.state import CortexState
from tools.github_tool import create_pr_for_agent
from rich.console import Console

console = Console()


def run_pr_creator(state: CortexState) -> dict:
    """LangGraph node: PR Creator."""
    approved = state.get("approved_findings", [])
    goal = state.get("goal", "")

    if not approved:
        console.print("[yellow]No approved findings — no PRs to open.[/yellow]")
        return {"pr_urls": [], "summary": "No findings approved by Evaluator."}

    if not _github_configured():
        console.print("[yellow]GITHUB_TOKEN not set — skipping PR creation.[/yellow]")
        return {
            "pr_urls": [],
            "summary": f"{len(approved)} findings approved but GITHUB_TOKEN not configured.",
        }

    # Group findings by agent
    by_agent: dict[str, list] = defaultdict(list)
    for finding in approved:
        by_agent[finding["agent"]].append(finding)

    pr_urls = []
    for agent_name, findings in by_agent.items():
        console.print(f"[cyan]Opening PR for {agent_name} ({len(findings)} findings)...[/cyan]")
        try:
            url = create_pr_for_agent(agent_name, findings, goal)
            if url:
                pr_urls.append(url)
                console.print(f"[green]✓ PR opened: {url}[/green]")
            else:
                console.print(f"[dim]{agent_name}: no pr_ready findings, skipped.[/dim]")
        except Exception as e:
            console.print(f"[red]✗ Failed to open PR for {agent_name}: {e}[/red]")

    summary = _build_summary(approved, pr_urls)
    return {"pr_urls": pr_urls, "summary": summary}


def _github_configured() -> bool:
    from config.settings import settings
    return bool(settings.github_token)


def _build_summary(approved: list, pr_urls: list) -> str:
    lines = [
        f"Argus Cortex completed.",
        f"Approved findings: {len(approved)}",
        f"PRs opened: {len(pr_urls)}",
    ]
    for url in pr_urls:
        lines.append(f"  → {url}")
    return "\n".join(lines)
