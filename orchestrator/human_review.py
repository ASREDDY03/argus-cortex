"""
Human Review node — runs after Evaluator, before PR Creator.

LangGraph interrupts HERE and waits for human input.
The human sees all approved findings and decides:
  - Which ones to actually open PRs for
  - Can reject individual findings before any PR is created

This is the human-in-the-loop checkpoint. Nothing goes to GitHub
until the human says so.
"""
from memory.state import CortexState


def run_human_review(state: CortexState) -> dict:
    """
    LangGraph node: Human Review.

    In normal flow this node is interrupted BEFORE it runs
    (interrupt_before=["human_review"]).

    After the human resumes the graph, this node finalises
    which findings to send to PR Creator based on human_approved_indices.
    If human_approved_indices is empty/not set, all approved findings pass through.
    """
    approved = state.get("approved_findings", [])
    indices = state.get("human_approved_indices", [])
    human_notes = state.get("human_notes", "")

    # If human provided specific indices, filter to those only
    if indices:
        final_findings = [approved[i] for i in indices if i < len(approved)]
    else:
        # Human resumed without changes — all approved findings pass through
        final_findings = approved

    return {
        "approved_findings": final_findings,
        "human_notes": human_notes,
    }
