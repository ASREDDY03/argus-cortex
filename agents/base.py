"""
Base class for all Generator agents.
Each agent uses claude-haiku-4-5 and reads code from the Argus Agent repo.
"""
import os
import json
import anthropic
from pathlib import Path
from memory.state import AgentFinding
from config.settings import settings

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


class BaseAgent:
    name: str = "base_agent"
    domain: str = ""
    files_to_review: list[str] = []

    AGENT_SYSTEM = """You are a specialized code review agent. Analyze the provided code and return findings as JSON.

Each finding must follow this format:
{
  "agent": "<your agent name>",
  "file": "<relative file path>",
  "line": <line number or null>,
  "severity": "<critical|high|medium|low>",
  "category": "<bug|security|performance|duplication|style>",
  "description": "<specific description of the issue>",
  "suggested_fix": "<concrete fix with code if possible>",
  "pr_ready": <true if you can provide a specific code change>
}

Return a JSON array of findings. Be specific. No vague suggestions."""

    def read_file(self, relative_path: str) -> str:
        """Read a file from the Argus Agent repo."""
        full_path = Path(settings.argus_repo_path) / relative_path
        if not full_path.exists():
            return f"[File not found: {relative_path}]"
        return full_path.read_text(encoding="utf-8")

    def analyze(self, goal: str) -> list[AgentFinding]:
        """Run the agent analysis. Override files_to_review in subclasses."""
        file_contents = {}
        for f in self.files_to_review:
            file_contents[f] = self.read_file(f)

        files_block = "\n\n".join(
            f"=== {path} ===\n{content}"
            for path, content in file_contents.items()
        )

        response = client.messages.create(
            model=settings.agent_model,
            max_tokens=4096,
            system=self.AGENT_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": f"Goal: {goal}\n\nDomain: {self.domain}\n\nReview these files:\n\n{files_block}",
                }
            ],
        )

        raw = response.content[0].text
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1:
            return []

        findings = json.loads(raw[start:end])
        # Ensure agent name is set
        for f in findings:
            f["agent"] = self.name
        return findings
