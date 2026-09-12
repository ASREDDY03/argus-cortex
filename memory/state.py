from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages


class AgentFinding(TypedDict):
    agent: str
    file: str
    line: int | None
    severity: str          # critical | high | medium | low
    category: str          # duplication | security | performance | bug | style
    description: str
    suggested_fix: str
    pr_ready: bool


class CortexState(TypedDict):
    """Shared state across the entire agent graph."""
    # Input
    goal: str

    # Planner output
    plan: List[str]
    agents_to_run: List[str]

    # Generator outputs (one per agent)
    findings: Annotated[List[AgentFinding], lambda a, b: a + b]

    # Evaluator output
    approved_findings: List[AgentFinding]
    rejected_findings: List[AgentFinding]
    evaluation_notes: str

    # Final output
    pr_urls: List[str]
    summary: str

    # Control
    messages: Annotated[list, add_messages]
    error: str | None
