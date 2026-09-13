from memory.state import CortexState
from agents.base import BaseAgent


class InfraAgent(BaseAgent):
    name = "infra_agent"
    domain = "Infrastructure — Docker Compose, Kubernetes manifests, Nginx config, Jenkinsfile"
    files_to_review = [
        "Devops/docker-compose.yml",
        "Devops/nginx/nginx.conf",
        "Devops/Jenkinsfile",
    ]


def run_infra_agent(state: CortexState) -> dict:
    """LangGraph node: Infra Generator agent."""
    if "infra_agent" not in state.get("agents_to_run", []):
        return {"findings": []}
    agent = InfraAgent()
    discovered = state.get("agent_files", {}).get("infra_agent", [])
    if discovered:
        agent.files_to_review = discovered
    focus = state.get("agent_focus", {}).get("infra_agent", "")
    findings = agent.analyze(state["goal"], focus=focus)
    return {"findings": findings}
