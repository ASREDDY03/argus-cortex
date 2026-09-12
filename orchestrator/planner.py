"""
Planner Agent — claude-sonnet-4-6
Receives the high-level goal and produces a structured plan.
"""
import anthropic
from langsmith import traceable
from memory.state import CortexState
from tools.retry import retry_api, parse_json_with_retry
from config.settings import settings

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

AVAILABLE_AGENTS = [
    "springboot_agent",
    "ml_agent",
    "react_agent",
    "infra_agent",
    "observability_agent",
]

PLANNER_SYSTEM = """You are the Planner for Argus Cortex, an AI agent network that audits and improves the Argus Agent DevOps system.

Argus Agent is a Spring Boot + React + Python ML + Docker Compose system that monitors Jenkins pipelines.

Your job:
1. Read the user's goal
2. Decide which specialized agents to activate
3. Return a structured JSON plan

Available agents:
- springboot_agent: Reviews Java/Spring Boot backend (JenkinsService.java, JenkinsController.java)
- ml_agent: Reviews Python ML service (anomaly detection, model persistence, Flask API)
- react_agent: Reviews React frontend (JenkinsDashboardComponent.jsx, JenkinsService.js)
- infra_agent: Reviews Docker Compose, Kubernetes, Nginx, Jenkinsfile
- observability_agent: Reviews Prometheus rules, Grafana dashboards, Alertmanager config

Always return valid JSON in this exact format:
{
  "plan": ["step 1", "step 2", ...],
  "agents_to_run": ["agent_name", ...],
  "focus": {
    "agent_name": "specific focus area for this agent"
  }
}"""

SYSTEM_BLOCK = [{"type": "text", "text": PLANNER_SYSTEM, "cache_control": {"type": "ephemeral"}}]


@retry_api
def _call_planner(messages: list) -> str:
    response = client.messages.create(
        model=settings.orchestrator_model,
        max_tokens=1024,
        system=SYSTEM_BLOCK,
        messages=messages,
    )
    return response.content[0].text


@traceable(run_type="chain", name="planner")
def run_planner(state: CortexState) -> dict:
    """LangGraph node: Planner."""
    goal = state["goal"]
    messages = [{"role": "user", "content": f"Goal: {goal}\n\nCreate a plan and select the right agents."}]

    raw = _call_planner(messages)

    def reprompt() -> str:
        return _call_planner(messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "Return ONLY valid JSON in the required format. No explanation."},
        ])

    plan_json = parse_json_with_retry(raw, kind="object", reprompt_fn=reprompt)

    return {
        "plan": plan_json.get("plan", []),
        "agents_to_run": plan_json.get("agents_to_run", AVAILABLE_AGENTS),
        "agent_focus": plan_json.get("focus", {}),
    }
