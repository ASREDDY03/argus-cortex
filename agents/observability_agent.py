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
    return ObservabilityAgent().run_node(state)
