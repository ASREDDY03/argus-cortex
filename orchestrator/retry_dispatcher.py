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
    current_files = state.get("agent_files", {})
    retry_count = state.get("retry_count", 0)

    if not retry_agents:
        return {"retry_count": retry_count + 1}

    merged_focus: dict[str, str] = dict(current_focus)
    merged_files: dict[str, list[str]] = dict(current_files)
    agent_names: list[str] = []

    for agent, spec in retry_agents.items():
        agent_names.append(agent)
        # spec can be a string (old format) or {"focus": ..., "files": [...]}
        if isinstance(spec, dict):
            focus = spec.get("focus", "")
            files = spec.get("files", [])
        else:
            focus = str(spec)
            files = []

        merged_focus[agent] = focus
        if files:
            # Restrict agent to only the specific files synthesizer flagged
            merged_files[agent] = files
            console.print(f"  [dim]{agent}: {focus} (targeting {len(files)} file(s))[/dim]")
        else:
            console.print(f"  [dim]{agent}: {focus}[/dim]")

    console.print(
        f"\n[cyan]Retry pass — re-running {len(agent_names)} agent(s): "
        f"{', '.join(agent_names)}[/cyan]"
    )

    return {
        "agents_to_run": agent_names,
        "agent_focus":   merged_focus,
        "agent_files":   merged_files,
        "retry_count":   retry_count + 1,
    }
