from memory.state import CortexState
from agents.base import BaseAgent


class SpringBootAgent(BaseAgent):
    name = "springboot_agent"
    domain = "Java Spring Boot backend — REST API, Jenkins polling, ML + LLM orchestration"
    files_to_review = [
        "Devops/springboot-backend/src/main/java/com/example/crudspring/services/JenkinsService.java",
        "Devops/springboot-backend/src/main/java/com/example/crudspring/controllers/JenkinsController.java",
    ]


def run_springboot_agent(state: CortexState) -> dict:
    """LangGraph node: SpringBoot Generator agent."""
    if "springboot_agent" not in state.get("agents_to_run", []):
        return {"findings": []}
    agent = SpringBootAgent()
    findings = agent.analyze(state["goal"])
    return {"findings": findings}
