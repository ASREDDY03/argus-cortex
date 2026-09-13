from memory.state import CortexState
from agents.base import BaseAgent


class ObservabilityAgent(BaseAgent):
    name = "observability_agent"
    domain = "Observability — Prometheus scrape configs, alert rules, Alertmanager, Grafana dashboards"
    files_to_review = [
        "Devops/prometheus.yml",
        "Devops/alert.rules.yml",
        "Devops/alertmanager.yml",
    ]


def run_observability_agent(state: CortexState) -> dict:
    """LangGraph node: Observability Generator agent."""
    if "observability_agent" not in state.get("agents_to_run", []):
        return {"findings": []}
    agent = ObservabilityAgent()
    discovered = state.get("agent_files", {}).get("observability_agent", [])
    if discovered:
        agent.files_to_review = discovered
    focus = state.get("agent_focus", {}).get("observability_agent", "")
    findings = agent.analyze(state["goal"], focus=focus)
    return {"findings": findings}
