"""
LangGraph StateGraph — the agent network.

Flow:
  planner
    ↓
  [springboot_agent, ml_agent, react_agent, infra_agent, observability_agent]  ← parallel
    ↓
  evaluator
    ↓
  pr_creator  ← opens one draft PR per agent domain on Argus Agent repo
    ↓
  END
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from memory.state import CortexState
from orchestrator.planner import run_planner
from orchestrator.evaluator import run_evaluator
from orchestrator.pr_creator import run_pr_creator
from agents.springboot_agent import run_springboot_agent
from agents.ml_agent import run_ml_agent
from agents.react_agent import run_react_agent
from agents.infra_agent import run_infra_agent
from agents.observability_agent import run_observability_agent


def build_graph(checkpoint_path: str = "checkpoints/cortex.db"):
    """Build and compile the Argus Cortex agent graph."""

    builder = StateGraph(CortexState)

    # Nodes
    builder.add_node("planner", run_planner)
    builder.add_node("springboot_agent", run_springboot_agent)
    builder.add_node("ml_agent", run_ml_agent)
    builder.add_node("react_agent", run_react_agent)
    builder.add_node("infra_agent", run_infra_agent)
    builder.add_node("observability_agent", run_observability_agent)
    builder.add_node("evaluator", run_evaluator)
    builder.add_node("pr_creator", run_pr_creator)

    # Entry
    builder.set_entry_point("planner")

    # Planner → all agents (parallel)
    builder.add_edge("planner", "springboot_agent")
    builder.add_edge("planner", "ml_agent")
    builder.add_edge("planner", "react_agent")
    builder.add_edge("planner", "infra_agent")
    builder.add_edge("planner", "observability_agent")

    # All agents → evaluator
    builder.add_edge("springboot_agent", "evaluator")
    builder.add_edge("ml_agent", "evaluator")
    builder.add_edge("react_agent", "evaluator")
    builder.add_edge("infra_agent", "evaluator")
    builder.add_edge("observability_agent", "evaluator")

    # Evaluator → PR Creator → END
    builder.add_edge("evaluator", "pr_creator")
    builder.add_edge("pr_creator", END)

    # Durable checkpointing — survives crashes
    checkpointer = SqliteSaver.from_conn_string(checkpoint_path)

    return builder.compile(checkpointer=checkpointer)
