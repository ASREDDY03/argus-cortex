from memory.state import CortexState
from agents.base import BaseAgent


class MLAgent(BaseAgent):
    name = "ml_agent"
    domain = "Python Flask ML service — Isolation Forest anomaly detection, SQLite persistence, model lifecycle"
    files_to_review = [
        "Devops/ml-service/app.py",
    ]


def run_ml_agent(state: CortexState) -> dict:
    """LangGraph node: ML Generator agent."""
    if "ml_agent" not in state.get("agents_to_run", []):
        return {"findings": []}
    agent = MLAgent()
    discovered = state.get("agent_files", {}).get("ml_agent", [])
    if discovered:
        agent.files_to_review = discovered
    focus = state.get("agent_focus", {}).get("ml_agent", "")
    findings = agent.analyze(state["goal"], focus=focus)
    return {"findings": findings}
