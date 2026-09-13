"""
Retry Dispatcher — routes specific agents back for a focused second pass.

Triggered by the synthesizer when it flags coverage gaps or under-coverage.
Updates agents_to_run and agent_focus to only the agents Synthesizer flagged,
then the graph fans them out for a second, tighter pass.

The findings reducer (a + b) in CortexState automatically merges first-pass
and retry findings — the Evaluator sees everything.

Retry is capped at 1 pass (retry_count < 1 guard in the conditional edge)
so the graph can never loop more than once.
"""
from rich.console import Console
from memory.state import CortexState

console = Console()


def run_retry_dispatcher(state: CortexState) -> dict:
    """LangGraph node: Retry Dispatcher."""
    retry_agents = state.get("retry_agents", {})
    current_focus = state.get("agent_focus", {})
    retry_count = state.get("retry_count", 0)

    if not retry_agents:
        # Synthesizer changed its mind — nothing to retry, just bump count
        return {"retry_count": retry_count + 1}

    # Retry focus takes precedence over original planner focus
    merged_focus = {**current_focus, **retry_agents}
    agent_names = list(retry_agents.keys())

    console.print(
        f"\n[cyan]↻ Retry pass — re-running {len(agent_names)} agent(s): "
        f"{', '.join(agent_names)}[/cyan]"
    )
    for agent, focus in retry_agents.items():
        console.print(f"  [dim]{agent}: {focus}[/dim]")

    return {
        "agents_to_run": agent_names,
        "agent_focus": merged_focus,
        "retry_count": retry_count + 1,
    }
