"""
Evaluator Agent — claude-sonnet-4-6
Independently scores every finding from the Generator agents.
Uses tool_use to guarantee schema-valid output — no JSON parsing needed.
"""
import json
import anthropic
from langsmith import traceable
from memory.state import CortexState, AgentFinding
from tools.retry import retry_api
from config.settings import settings

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

EVALUATOR_SYSTEM = """You are the Evaluator for Argus Cortex. Your job is to independently assess findings from specialized agents and decide which are worth opening a PR for.

Approval criteria:
- APPROVE if: finding is specific (has file + line), fix is actionable, impact is clear
- REJECT if: finding is vague, duplicated, incorrect, or the fix would break functionality

Call submit_evaluation with your decision for every finding."""

SYSTEM_BLOCK = [{"type": "text", "text": EVALUATOR_SYSTEM, "cache_control": {"type": "ephemeral"}}]

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


@retry_api
def _call_evaluator(messages: list) -> dict:
    """Calls the evaluator with forced tool_use — returns the parsed evaluation dict."""
    response = client.messages.create(
        model=settings.orchestrator_model,
        max_tokens=2048,
        system=SYSTEM_BLOCK,
        messages=messages,
        tools=[EVALUATION_TOOL],
        tool_choice={"type": "tool", "name": "submit_evaluation"},
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    return {}


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

    findings_text = json.dumps(findings, indent=2)
    messages = [{"role": "user", "content": f"Evaluate these {len(findings)} findings:\n\n{findings_text}"}]

    result = _call_evaluator(messages)

    approved_indices = result.get("approved", [])
    rejected_indices = result.get("rejected", [])

    return {
        "approved_findings": [findings[i] for i in approved_indices if i < len(findings)],
        "rejected_findings": [findings[i] for i in rejected_indices if i < len(findings)],
        "evaluation_notes": result.get("notes", ""),
    }
