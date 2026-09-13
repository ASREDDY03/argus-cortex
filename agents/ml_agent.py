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
    return MLAgent().run_node(state)
