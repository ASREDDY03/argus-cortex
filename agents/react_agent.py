from memory.state import CortexState
from agents.base import BaseAgent


class ReactAgent(BaseAgent):
    name = "react_agent"
    domain = "React 18 frontend — Jenkins dashboard, Grafana monitoring tab, job detail drawer"
    files_to_review = [
        "Devops/react-frontend/src/components/JenkinsDashboardComponent.jsx",
        "Devops/react-frontend/src/services/JenkinsService.js",
    ]


def run_react_agent(state: CortexState) -> dict:
    """LangGraph node: React Generator agent."""
    return ReactAgent().run_node(state)
