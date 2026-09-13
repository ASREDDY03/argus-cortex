"""
Planner Agent — claude-sonnet-4-6
Receives the high-level goal and produces a structured plan.
Uses tool_use for schema-valid output. Full prompt/response traced to LangSmith.
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from memory.state import CortexState
from tools.file_discovery import summarise_discovery
from config.settings import settings

AVAILABLE_AGENTS = [
    "springboot_agent",
    "ml_agent",
    "react_agent",
    "infra_agent",
    "observability_agent",
    "jenkins_agent",
]

PLANNER_SYSTEM = """You are the Planner for Argus Cortex, an AI agent network that audits and improves the Argus Agent DevOps system.

Argus Agent is a Spring Boot + React + Python ML + Docker Compose system that monitors Jenkins pipelines.

Available agents:
- springboot_agent: Reviews Java/Spring Boot backend (JenkinsService.java, JenkinsController.java)
- ml_agent: Reviews Python ML service (anomaly detection, model persistence, Flask API)
- react_agent: Reviews React frontend (JenkinsDashboardComponent.jsx, JenkinsService.js)
- infra_agent: Reviews Docker Compose, Kubernetes manifests, Nginx config
- observability_agent: Reviews Prometheus rules, Grafana dashboards, Alertmanager config
- jenkins_agent: Reviews AND rewrites the Jenkinsfile — finds issues (missing tests, no parallel stages, hardcoded values) AND generates a complete production-ready replacement pipeline with parallel stages, ML service, health checks, and post{} cleanup

Read the user's goal, decide which agents to activate, and call create_plan with your decision."""

PLAN_TOOL = {
    "name": "create_plan",
    "description": "Create a structured audit plan and select which agents to activate.",
    "input_schema": {
        "type": "object",
        "properties": {
            "plan": {
                "type": "array",
                "description": "Ordered list of high-level steps for this audit run.",
                "items": {"type": "string"},
            },
            "agents_to_run": {
                "type": "array",
                "description": "Names of agents to activate for this goal.",
                "items": {
                    "type": "string",
                    "enum": AVAILABLE_AGENTS,
                },
            },
            "focus": {
                "type": "object",
                "description": "Per-agent focus instructions keyed by agent name.",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["plan", "agents_to_run", "focus"],
    },
}

_llm = ChatAnthropic(
    model=settings.orchestrator_model,
    api_key=settings.anthropic_api_key,
    max_tokens=1024,
    max_retries=3,
)
_llm_with_tools = _llm.bind_tools(
    [PLAN_TOOL],
    tool_choice={"type": "tool", "name": "create_plan"},
)


def _call_planner(messages: list) -> dict:
    """Invoke the planner — auto-traced to LangSmith with full prompt + response."""
    response = _llm_with_tools.invoke(messages)
    if response.tool_calls:
        return response.tool_calls[0]["args"]
    return {}


@traceable(run_type="chain", name="planner")
def run_planner(state: CortexState) -> dict:
    """LangGraph node: Planner."""
    goal = state["goal"]

    discovered = state.get("agent_files", {})
    files_context = (
        f"\n\nDiscovered files in repo:\n{summarise_discovery(discovered)}"
        if discovered else ""
    )

    messages = [
        SystemMessage(content=[{
            "type": "text",
            "text": PLANNER_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }]),
        HumanMessage(content=(
            f"Goal: {goal}{files_context}\n\nCreate a plan and select the right agents."
        )),
    ]

    plan = _call_planner(messages)

    return {
        "plan": plan.get("plan", []),
        "agents_to_run": plan.get("agents_to_run", AVAILABLE_AGENTS),
        "agent_focus": plan.get("focus", {}),
    }
