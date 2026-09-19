"""
Evaluator Agent — claude-sonnet-4-6
Independently scores every finding from the Generator agents.
Uses tool_use for schema-valid output. Full prompt/response traced to LangSmith.
"""
import json
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from memory.state import CortexState, AgentFinding
from config.settings import settings

EVALUATOR_SYSTEM = """You are the Evaluator for Argus Cortex. Your job is to independently assess findings from specialized agents and decide which are worth opening a PR for.

Approval criteria:
- APPROVE if: finding is specific (has file + line), fix is actionable, impact is clear
- REJECT if: finding is vague, duplicated, incorrect, or the fix would break functionality

Call submit_evaluation with your decision for every finding."""

EVALUATION_TOOL = {
    "name": "submit_evaluation",
    "description": "Submit approval/rejection decisions for all findings.",
    "input_schema": {
        "type": "object",
        "properties": {
            "approved": {
                "type": "array",
                "description": "Zero-based indices of findings that pass evaluation.",
                "items": {"type": "integer"},
            },
            "rejected": {
                "type": "array",
                "description": "Zero-based indices of findings that fail evaluation.",
                "items": {"type": "integer"},
            },
            "notes": {
                "type": "string",
                "description": "Overall evaluation summary — patterns seen, general quality assessment.",
            },
        },
        "required": ["approved", "rejected", "notes"],
    },
}

_llm = ChatAnthropic(
    model=settings.orchestrator_model,
    api_key=settings.anthropic_api_key,
    max_tokens=2048,
    max_retries=3,
)
_llm_with_tools = _llm.bind_tools(
    [EVALUATION_TOOL],
    tool_choice={"type": "tool", "name": "submit_evaluation"},
)


def _call_evaluator(messages: list) -> tuple[dict, int, int]:
    """Invoke the evaluator — returns (result, tokens_in, tokens_out)."""
    response = _llm_with_tools.invoke(messages)
    usage = response.usage_metadata or {}
    tokens_in  = usage.get("input_tokens", 0)
    tokens_out = usage.get("output_tokens", 0)
    if response.tool_calls:
        return response.tool_calls[0]["args"], tokens_in, tokens_out
    return {}, tokens_in, tokens_out


@traceable(run_type="chain", name="evaluator")
def run_evaluator(state: CortexState) -> dict:
    """LangGraph node: Evaluator."""
    findings = state.get("findings", [])

    if not findings:
        return {
            "approved_findings": [],
            "rejected_findings": [],
            "evaluation_notes": "No findings to evaluate.",
        }

    # Include Synthesizer output so cross-cutting patterns influence scoring
    synthesis_parts = []
    cross_cutting = state.get("cross_cutting_issues", [])
    coverage_gaps = state.get("coverage_gaps", [])
    synthesis_notes = state.get("synthesis_notes", "")

    if cross_cutting:
        synthesis_parts.append("Cross-cutting patterns identified by Synthesizer:\n" +
                               "\n".join(f"- {c}" for c in cross_cutting))
    if coverage_gaps:
        synthesis_parts.append("Coverage gaps flagged by Synthesizer:\n" +
                               "\n".join(f"- {g}" for g in coverage_gaps))
    if synthesis_notes:
        synthesis_parts.append(f"Synthesizer summary: {synthesis_notes}")

    synthesis_context = ("\n\n" + "\n\n".join(synthesis_parts)) if synthesis_parts else ""
    # Strip old_code/new_code — evaluator scores on severity/description/file, not diffs
    slim_findings = [
        {k: v for k, v in f.items() if k not in ("old_code", "new_code")}
        for f in findings
    ]
    findings_text = json.dumps(slim_findings, indent=2)

    messages = [
        SystemMessage(content=[{
            "type": "text",
            "text": EVALUATOR_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }]),
        HumanMessage(content=(
            f"Evaluate these {len(slim_findings)} findings:{synthesis_context}\n\n{findings_text}"
        )),
    ]

    result, tokens_in, tokens_out = _call_evaluator(messages)

    approved_indices = result.get("approved", [])
    rejected_indices = result.get("rejected", [])

    return {
        "approved_findings": [findings[i] for i in approved_indices if i < len(findings)],
        "rejected_findings": [findings[i] for i in rejected_indices if i < len(findings)],
        "evaluation_notes": result.get("notes", ""),
        "orch_tokens_in": tokens_in,
        "orch_tokens_out": tokens_out,
    }
