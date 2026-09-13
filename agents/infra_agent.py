from memory.state import CortexState
from agents.base import BaseAgent


class InfraAgent(BaseAgent):
    name = "infra_agent"
    domain = "Infrastructure — Docker Compose, Kubernetes manifests, Nginx config"
    files_to_review = [
        "Devops/docker-compose.yml",
        "Devops/nginx/nginx.conf",
    ]


def run_infra_agent(state: CortexState) -> dict:
    """LangGraph node: Infra Generator agent."""
    return InfraAgent().run_node(state)
