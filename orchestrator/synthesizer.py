"""
Synthesizer — claude-sonnet-4-6
Runs after all Generator agents fan in, before the Evaluator.

Responsibilities:
  1. Spot cross-cutting issues — patterns that appear across multiple agents/domains
     (e.g. timeouts misconfigured in both Spring Boot AND docker-compose)
  2. Identify coverage gaps — files or areas that received zero findings
     (may mean they're clean, or that the agent didn't look hard enough)
  3. Recommend targeted re-runs — which agents should look again and where
     (surfaced to the human; no automatic retry loop yet)

The Evaluator receives the synthesis output as context so cross-cutting
patterns can influence how individual findings are scored.
"""
import json
import anthropic
from langsmith import traceable
from memory.state import CortexState
from tools.retry import retry_api
from config.settings import settings

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

# Domain map — tells the Synthesizer what each agent is responsible for.
# Used to detect gaps (agent ran but reported nothing on a file it owns).
AGENT_DOMAINS = {
    "springboot_agent":     ["JenkinsService.java", "JenkinsController.java"],
    "ml_agent":             ["ml-service/app.py"],
    "react_agent":          ["JenkinsDashboardComponent.jsx", "JenkinsService.js"],
    "infra_agent":          ["docker-compose.yml", "nginx.conf", "Jenkinsfile"],
    "observability_agent":  ["prometheus.yml", "alert.rules.yml", "alertmanager.yml"],
}

SYNTHESIZER_SYSTEM = """You are the Synthesizer for Argus Cortex, an AI agent network that audits the Argus Agent DevOps system.

You receive all findings produced by all generator agents and your job is to reason across them — not to re-evaluate individual findings, but to look at the big picture.

Your three tasks:
1. CROSS-CUTTING ISSUES — identify patterns that appear across multiple agents or domains.
   Example: "Three agents flagged missing timeout handling — this is a systemic issue across Spring Boot, ML service, and nginx config."
   Example: "Both infra_agent and springboot_agent found hardcoded credentials — likely same root cause."

2. COVERAGE GAPS — identify files or areas that an agent is responsible for but produced zero findings on.
   This may mean the area is clean (good!) or the agent didn't look hard enough (worth flagging).
   Do NOT flag an area as a gap just because it has few findings — only flag it if you suspect under-coverage.

3. RETRY RECOMMENDATIONS — if a gap looks suspicious (an agent reported nothing on a complex file),
   recommend a targeted re-run: which agent, which file, what specific angle to investigate.
   Keep recommendations concrete and specific. If coverage looks fine, leave retry_agents empty.

Call submit_synthesis with your analysis."""

SYSTEM_BLOCK = [{"type": "text", "text": SYNTHESIZER_SYSTEM, "cache_control": {"type": "ephemeral"}}]

SYNTHESIS_TOOL = {
    "name": "submit_synthesis",
    "description": "Submit cross-cutting analysis and coverage assessment across all agent findings.",
    "input_schema": {
        "type": "object",
        "properties": {
            "cross_cutting_issues": {
                "type": "array",
                "description": "Patterns that span multiple agents or domains. Each entry is a concrete observation, not a vague statement.",
                "items": {"type": "string"},
            },
            "coverage_gaps": {
                "type": "array",
                "description": "Files or areas an agent owns but produced no findings on, where that looks suspicious rather than clean.",
                "items": {"type": "string"},
            },
            "retry_agents": {
                "type": "object",
                "description": "Recommended targeted re-runs keyed by agent name. Value is the specific focus instruction. Empty object if coverage is adequate.",
                "additionalProperties": {"type": "string"},
            },
            "synthesis_notes": {
                "type": "string",
                "description": "2-4 sentence summary of overall coverage quality and the most important cross-cutting observation.",
            },
        },
        "required": ["cross_cutting_issues", "coverage_gaps", "retry_agents", "synthesis_notes"],
    },
}


@retry_api
def _call_synthesizer(messages: list) -> dict:
    """Calls the synthesizer with forced tool_use — returns parsed synthesis dict."""
    response = client.messages.create(
        model=settings.orchestrator_model,
        max_tokens=2048,
        system=SYSTEM_BLOCK,
        messages=messages,
        tools=[SYNTHESIS_TOOL],
        tool_choice={"type": "tool", "name": "submit_synthesis"},
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    return {}


@traceable(run_type="chain", name="synthesizer")
def run_synthesizer(state: CortexState) -> dict:
    """LangGraph node: Synthesizer."""
    findings = state.get("findings", [])
    agents_that_ran = state.get("agents_to_run", list(AGENT_DOMAINS.keys()))

    if not findings:
        return {
            "cross_cutting_issues": [],
            "coverage_gaps": [],
            "retry_agents": {},
            "synthesis_notes": "No findings to synthesize.",
        }

    # Build coverage summary: for each agent that ran, list which of its files
    # appear in findings vs. which don't
    coverage_summary = []
    files_with_findings = {f.get("file", "") for f in findings}

    for agent in agents_that_ran:
        owned_files = AGENT_DOMAINS.get(agent, [])
        covered = [f for f in owned_files if any(f in fw for fw in files_with_findings)]
        uncovered = [f for f in owned_files if not any(f in fw for fw in files_with_findings)]
        coverage_summary.append(
            f"{agent}: covered={covered or 'none'}, no findings on={uncovered or 'none'}"
        )

    findings_text = json.dumps(findings, indent=2)
    coverage_text = "\n".join(coverage_summary)

    prompt = (
        f"Agents that ran: {', '.join(agents_that_ran)}\n\n"
        f"Coverage summary (files with findings vs. files with none):\n{coverage_text}\n\n"
        f"All findings ({len(findings)} total):\n{findings_text}"
    )

    messages = [{"role": "user", "content": prompt}]
    result = _call_synthesizer(messages)

    return {
        "cross_cutting_issues": result.get("cross_cutting_issues", []),
        "coverage_gaps": result.get("coverage_gaps", []),
        "retry_agents": result.get("retry_agents", {}),
        "synthesis_notes": result.get("synthesis_notes", ""),
    }
