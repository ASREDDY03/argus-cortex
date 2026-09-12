from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages


class AgentFinding(TypedDict):
    agent: str
    file: str
    line: int | None
    severity: str           # critical | high | medium | low
    category: str           # duplication | security | performance | bug | style
    description: str
    suggested_fix: str
    old_code: str           # exact code to replace (empty string if not applicable)
    new_code: str           # replacement code (empty string if not applicable)
    pr_ready: bool


class CortexState(TypedDict):
    """Shared state across the entire agent graph."""

    # Input
    goal: str

    # Run tracking (long-term memory)
    run_id: str
    thread_id: str

    # Planner output
    plan: List[str]
    agents_to_run: List[str]
    agent_focus: dict[str, str]   # e.g. {"springboot_agent": "focus on auth and duplication"}

    # Generator outputs — merged across all agents automatically
    findings: Annotated[List[AgentFinding], lambda a, b: a + b]

    # Evaluator output
    approved_findings: List[AgentFinding]
    rejected_findings: List[AgentFinding]
    evaluation_notes: str

    # Human-in-the-loop — set during the interrupt pause
    human_approved_indices: List[int]   # which approved_findings to actually PR
    human_notes: str                    # optional human comment

    # Final output
    pr_urls: List[str]
    summary: str

    # Control
    messages: Annotated[list, add_messages]
    error: str | None
