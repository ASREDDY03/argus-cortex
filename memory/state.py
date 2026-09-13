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

    # Dynamically discovered files per agent (set before graph runs)
    agent_files: dict[str, list[str]]   # {agent_name: [relative_path, ...]}

    # Generator outputs — merged across all agents automatically
    findings: Annotated[List[AgentFinding], lambda a, b: a + b]

    # Synthesizer output (runs after all agents, before Evaluator)
    cross_cutting_issues: List[str]   # patterns that span multiple agents/domains
    coverage_gaps: List[str]          # files or areas no agent meaningfully covered
    synthesis_notes: str              # overall coverage quality summary
    retry_agents: dict[str, str]      # recommended re-runs: {agent_name: focus}

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

    # Deduplication — findings suppressed as known duplicates across all agents
    suppressed_count: Annotated[int, lambda a, b: a + b]

    # Token usage — separate per model tier so cost can be computed accurately
    # agent_*       : claude-haiku-4-5  (all 8 generator agents)
    # orch_*        : claude-sonnet-4-6 (planner, synthesizer, evaluator)
    agent_tokens_in:  Annotated[int, lambda a, b: a + b]
    agent_tokens_out: Annotated[int, lambda a, b: a + b]
    orch_tokens_in:   Annotated[int, lambda a, b: a + b]
    orch_tokens_out:  Annotated[int, lambda a, b: a + b]

    # Control
    messages: Annotated[list, add_messages]
    error: str | None
    retry_count: int   # number of retry passes completed (caps at 1 to prevent infinite loops)
