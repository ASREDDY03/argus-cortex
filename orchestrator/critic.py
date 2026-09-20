"""
Critic node — runs after Test Writer, before PR Creator.

Takes approved findings and scores each one for application risk:
  low    — safe to apply, straightforward fix
  medium — needs reviewer attention, possible edge cases
  high   — could break functionality if applied incorrectly

Risk scores are written back to approved_findings and surfaced in the PR body
so reviewers know what to scrutinize vs what to rubber-stamp.
"""
import json
import logging

import anthropic

from memory.state import CortexState
from config.settings import settings

logger = logging.getLogger(__name__)

CRITIC_TOOL = {
    "name": "submit_risk_scores",
    "description": "Submit application risk scores for each approved finding.",
    "input_schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "description": "Zero-based index into the findings list"
                        },
                        "risk": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "Application risk level"
                        },
                        "risk_note": {
                            "type": "string",
                            "description": "One sentence explaining the risk. Empty string for low risk."
                        }
                    },
                    "required": ["index", "risk", "risk_note"]
                }
            }
        },
        "required": ["scores"]
    }
}

CRITIC_SYSTEM = """You are a code review critic. You receive a list of approved code fixes and assess the risk of applying each one.

Risk levels:
  low    — fix is mechanical and safe (adding a null check, removing a debug log, fixing a typo)
  medium — fix changes logic or control flow; needs reviewer attention and testing
  high   — fix touches auth, security boundaries, database transactions, or shared state; could break things if misapplied

For each finding, return its index and a risk level. For medium and high, add a one-sentence risk_note explaining what could go wrong.

Be conservative: when in doubt between two levels, pick the higher one."""


def run_critic(state: CortexState) -> dict:
    """LangGraph node: Critic."""
    approved = state.get("approved_findings", [])
    if not approved:
        return {}

    # Strip code diffs — critic only needs description + suggested_fix + file
    slim = [
        {k: v for k, v in f.items() if k not in ("old_code", "new_code")}
        for f in approved
    ]

    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model=settings.orchestrator_model,
            max_tokens=1024,
            tools=[CRITIC_TOOL],
            tool_choice={"type": "tool", "name": "submit_risk_scores"},
            system=CRITIC_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Score the application risk for each of these {len(slim)} approved findings:\n\n{json.dumps(slim, indent=2)}"
            }]
        )

        usage = response.usage
        tokens_in  = getattr(usage, "input_tokens", 0)
        tokens_out = getattr(usage, "output_tokens", 0)

        scores: list[dict] = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_risk_scores":
                scores = block.input.get("scores", [])
                break

        # Write risk and risk_note back onto approved_findings
        scored = list(approved)
        for s in scores:
            idx = s.get("index", -1)
            if 0 <= idx < len(scored):
                scored[idx] = {**scored[idx], "risk": s.get("risk", "low"), "risk_note": s.get("risk_note", "")}

        return {
            "approved_findings": scored,
            "orch_tokens_in": tokens_in,
            "orch_tokens_out": tokens_out,
        }

    except Exception as exc:
        logger.warning("[critic] Failed to score risks: %s", exc)
        return {}
