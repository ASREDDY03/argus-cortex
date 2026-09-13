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
    if "react_agent" not in state.get("agents_to_run", []):
        return {"findings": []}
    agent = ReactAgent()
    discovered = state.get("agent_files", {}).get("react_agent", [])
    if discovered:
        agent.files_to_review = discovered
    focus = state.get("agent_focus", {}).get("react_agent", "")
    findings = agent.analyze(state["goal"], focus=focus)
    return {"findings": findings}
