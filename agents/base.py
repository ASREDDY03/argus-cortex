"""
Base class for all Generator agents.
Each agent uses claude-haiku-4-5, reads code from the Argus Agent repo,
and queries long-term memory to avoid re-reporting known issues.

Prompt caching strategy:
  - System prompt     → cached (static across all runs)
  - File contents     → cached (large, rarely change — biggest cost saving)
  - Goal/focus/known  → NOT cached (changes every run)
"""
import json
import anthropic
from pathlib import Path
from langsmith import traceable
from memory.state import AgentFinding
from memory.long_term import get_past_findings_for_files
from config.settings import settings

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


class BaseAgent:
    name: str = "base_agent"
    domain: str = ""
    files_to_review: list[str] = []

    AGENT_SYSTEM = """You are a specialized code review agent for the Argus Agent DevOps system.

Analyze the provided code and return findings as a JSON array.

Each finding must follow this exact format:
{
  "agent": "<your agent name>",
  "file": "<relative file path>",
  "line": <line number or null>,
  "severity": "<critical|high|medium|low>",
  "category": "<bug|security|performance|duplication|style>",
  "description": "<specific description of the issue>",
  "suggested_fix": "<concrete fix with code snippet if possible>",
  "pr_ready": <true if you can provide a specific code change, false otherwise>
}

Rules:
- Be specific — include file + line number whenever possible
- Do NOT re-report issues listed in the KNOWN ISSUES section below
- Focus on NEW issues not previously found
- No vague suggestions like "improve error handling" — say exactly what to change
- Return ONLY the JSON array, no markdown, no explanation"""

    def read_file(self, relative_path: str) -> str:
        full_path = Path(settings.argus_repo_path) / relative_path
        if not full_path.exists():
            return f"[File not found: {relative_path}]"
        return full_path.read_text(encoding="utf-8")

    def _format_known_issues(self, past: list[dict]) -> str:
        if not past:
            return "None."
        lines = []
        for p in past:
            status = f"PR: {p['pr_url']}" if p.get("pr_url") else "reported, no PR yet"
            lines.append(f"- [{p['severity']}] {p['file']} — {p['description']} ({status})")
        return "\n".join(lines)

    @traceable(run_type="chain")
    def analyze(self, goal: str, focus: str = "") -> list[AgentFinding]:
        """Run agent analysis with long-term memory context and prompt caching."""

        # Read files
        file_contents: dict[str, str] = {}
        for f in self.files_to_review:
            file_contents[f] = self.read_file(f)

        # Query long-term memory
        past_findings = get_past_findings_for_files(self.files_to_review)
        known_issues_text = self._format_known_issues(past_findings)

        files_block = "\n\n".join(
            f"=== {path} ===\n{content}"
            for path, content in file_contents.items()
        )

        focus_line = f"Focus specifically on: {focus}\n\n" if focus else ""

        dynamic_block = (
            f"Goal: {goal}\n\n"
            f"Domain: {self.domain}\n\n"
            f"{focus_line}"
            f"KNOWN ISSUES (already reported — do NOT repeat these):\n"
            f"{known_issues_text}\n\n"
            f"Review the files above for NEW issues only."
        )

        response = client.messages.create(
            model=settings.agent_model,
            max_tokens=4096,
            # Cache the system prompt — static, same on every agent call
            system=[
                {
                    "type": "text",
                    "text": self.AGENT_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        # Cache the file contents — large, rarely change between runs
                        {
                            "type": "text",
                            "text": files_block,
                            "cache_control": {"type": "ephemeral"},
                        },
                        # NOT cached — goal, focus, known issues change every run
                        {
                            "type": "text",
                            "text": dynamic_block,
                        },
                    ],
                }
            ],
        )

        raw = response.content[0].text.strip()
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return []

        findings = json.loads(raw[start:end])
        for f in findings:
            f["agent"] = self.name
        return findings
