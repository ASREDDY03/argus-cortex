"""
Evaluator Agent — claude-sonnet-4-6
Independently scores every finding from the Generator agents.
Rejects low quality, vague, or incorrect findings before PRs are opened.
"""
import json
import anthropic
from memory.state import CortexState, AgentFinding
from config.settings import settings

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

EVALUATOR_SYSTEM = """You are the Evaluator for Argus Cortex. Your job is to independently assess findings from specialized agents and decide which are worth opening a PR for.

Scoring criteria:
- APPROVE if: finding is specific (has file + line), fix is actionable, impact is clear
- REJECT if: finding is vague, duplicated, incorrect, or the fix would break functionality

Return JSON in this exact format:
{
  "approved": [<list of finding indices that pass>],
  "rejected": [<list of finding indices that fail>],
  "notes": "overall evaluation summary"
}"""


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

    response = client.messages.create(
        model=settings.orchestrator_model,
        max_tokens=2048,
        system=EVALUATOR_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Evaluate these {len(findings)} findings and decide which to approve for PRs:\n\n{findings_text}",
            }
        ],
    )

    raw = response.content[0].text
    start = raw.find("{")
    end = raw.rfind("}") + 1
    result = json.loads(raw[start:end])

    approved_indices = result.get("approved", [])
    rejected_indices = result.get("rejected", [])

    return {
        "approved_findings": [findings[i] for i in approved_indices if i < len(findings)],
        "rejected_findings": [findings[i] for i in rejected_indices if i < len(findings)],
        "evaluation_notes": result.get("notes", ""),
    }
