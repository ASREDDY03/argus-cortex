"""
LangGraph StateGraph — Argus Cortex agent network.

Full flow:
  planner
    ↓
  [springboot, ml, react, infra, observability, jenkins, security, dependency]  ← parallel (first pass)
    ↓
  synthesizer  ← cross-cutting analysis + coverage gap detection
    ↓ (conditional)
  ┌─ retry_agents non-empty AND retry_count < 1 → retry_dispatcher
  │     ↓  fans out to flagged agents only (second pass)
  │     ↓  findings merged via reducer (a + b)
  │     └─→ synthesizer  (second pass, retry_count=1 → always goes to evaluator)
  └─ otherwise → evaluator
       ↓ (conditional)
     ┌─ no approved findings → END
     └─ approved findings → human_review  ← INTERRUPT HERE
                                 ↓
                             pr_creator   ← opens draft PRs on GitHub
                                 ↓
                               END

Key LangGraph features used:
  - Parallel fan-out (planner → 8 agents simultaneously)
  - State merging (findings from all agents merged via reducer)
  - Synthesizer (cross-agent analysis before evaluation)
  - Retry loop (synthesizer → retry_dispatcher → agents → synthesizer, max 1 pass)
  - Conditional edges (route based on evaluator output)
  - interrupt_before=["human_review"] (pause for human approval)
  - SqliteSaver checkpointing (durable, survives crashes, resumable)
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from memory.state import CortexState
from orchestrator.planner import run_planner
from orchestrator.synthesizer import run_synthesizer
from orchestrator.evaluator import run_evaluator
from orchestrator.human_review import run_human_review
from orchestrator.test_writer import run_test_writer
from orchestrator.pr_creator import run_pr_creator
from orchestrator.retry_dispatcher import run_retry_dispatcher
from agents.springboot_agent import run_springboot_agent
from agents.ml_agent import run_ml_agent
from agents.react_agent import run_react_agent
from agents.infra_agent import run_infra_agent
from agents.observability_agent import run_observability_agent
from agents.jenkins_agent import run_jenkins_agent
from agents.security_agent import run_security_agent
from agents.dependency_agent import run_dependency_agent

GENERATOR_AGENTS = [
    "springboot_agent",
    "ml_agent",
    "react_agent",
    "infra_agent",
    "observability_agent",
    "jenkins_agent",
    "security_agent",
    "dependency_agent",
]


def _route_after_synthesizer(state: CortexState) -> str:
    """
    Conditional edge: after Synthesizer, decide next node.
    - retry_agents non-empty AND first pass only → retry_dispatcher (agents re-run)
    - otherwise → evaluator
    """
    retry_agents = state.get("retry_agents", {})
    retry_count = state.get("retry_count", 0)
    if retry_agents and retry_count < 1:
        return "retry_dispatcher"
    return "evaluator"


def _route_after_evaluator(state: CortexState) -> str:
    """
    Conditional edge: after Evaluator, decide next node.
    - If there are approved findings → go to human_review (interrupt point)
    - If nothing approved → END (no point asking human)
    """
    if state.get("approved_findings"):
        return "human_review"
    return END


def build_graph(checkpoint_path: str = "checkpoints/cortex.db"):
    """Build and compile the Argus Cortex LangGraph network."""

    builder = StateGraph(CortexState)

    # --- Nodes ---
    builder.add_node("planner", run_planner)
    builder.add_node("springboot_agent", run_springboot_agent)
    builder.add_node("ml_agent", run_ml_agent)
    builder.add_node("react_agent", run_react_agent)
    builder.add_node("infra_agent", run_infra_agent)
    builder.add_node("observability_agent", run_observability_agent)
    builder.add_node("jenkins_agent", run_jenkins_agent)
    builder.add_node("security_agent", run_security_agent)
    builder.add_node("dependency_agent", run_dependency_agent)
    builder.add_node("synthesizer", run_synthesizer)
    builder.add_node("retry_dispatcher", run_retry_dispatcher)
    builder.add_node("evaluator", run_evaluator)
    builder.add_node("human_review", run_human_review)
    builder.add_node("test_writer", run_test_writer)
    builder.add_node("pr_creator", run_pr_creator)

    # --- Entry point ---
    builder.set_entry_point("planner")

    # --- Planner → all Generator agents (parallel fan-out) ---
    for agent in GENERATOR_AGENTS:
        builder.add_edge("planner", agent)

    # --- All Generator agents → Synthesizer (fan-in) ---
    # Synthesizer sees the full merged findings before Evaluator scores them
    for agent in GENERATOR_AGENTS:
        builder.add_edge(agent, "synthesizer")

    # --- Synthesizer → conditional: retry or evaluate ---
    # First pass: if synthesizer flagged agents to retry → retry_dispatcher
    # Second pass (retry_count ≥ 1) or no retries needed → evaluator
    builder.add_conditional_edges(
        "synthesizer",
        _route_after_synthesizer,
        {"retry_dispatcher": "retry_dispatcher", "evaluator": "evaluator"},
    )

    # --- retry_dispatcher → all Generator agents (fan-out, only flagged ones actually run) ---
    # Unflagged agents early-return with {"findings": []} when not in agents_to_run
    for agent in GENERATOR_AGENTS:
        builder.add_edge("retry_dispatcher", agent)

    # --- Evaluator → conditional routing ---
    builder.add_conditional_edges(
        "evaluator",
        _route_after_evaluator,
        {
            "human_review": "human_review",
            END: END,
        },
    )

    # --- Human Review -> Test Writer -> PR Creator -> END ---
    builder.add_edge("human_review", "test_writer")
    builder.add_edge("test_writer", "pr_creator")
    builder.add_edge("pr_creator", END)

    # --- Durable checkpointing (SQLite) ---
    checkpointer = SqliteSaver.from_conn_string(checkpoint_path)

    # --- Compile with human-in-the-loop interrupt ---
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],  # Graph pauses here for human approval
    )
