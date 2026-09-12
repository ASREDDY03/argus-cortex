"""
Base class for all Generator agents.
Each agent uses claude-haiku-4-5, reads code from the Argus Agent repo,
and queries long-term memory to avoid re-reporting known issues.

Prompt caching strategy:
  - System prompt     → cached (static across all runs)
  - File contents     → cached (large, rarely change — biggest cost saving)
  - Goal/focus/known  → NOT cached (changes every run)

Retry strategy:
  - API calls    → retried up to 3x with exponential backoff (rate limits, connection errors)
  - JSON parsing → retried once with a stricter re-prompt if LLM returns malformed output
"""
import anthropic
from pathlib import Path
from langsmith import traceable
from memory.state import AgentFinding
from memory.long_term import get_past_findings_for_files
from tools.retry import retry_api, parse_json_with_retry
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
  "suggested_fix": "<one-line summary of the fix>",
  "old_code": "<the exact lines to replace, copy-pasted from the file — empty string if not applicable>",
  "new_code": "<the replacement lines — empty string if not applicable>",
  "pr_ready": <true if old_code and new_code are both non-empty and the change is safe to apply>
}

Rules:
- Be specific — include file + line number whenever possible
- old_code must be the EXACT text from the file (copy paste it) — it will be used for find-and-replace
- old_code must be unique in the file — include enough surrounding lines to make it unique
- Do NOT re-report issues listed in the KNOWN ISSUES section below
- Focus on NEW issues not previously found
- Return ONLY the JSON array, no markdown, no explanation"""

    JSON_REPAIR_PROMPT = "Your previous response was not valid JSON. Return ONLY a valid JSON array of findings. No markdown, no explanation, just the JSON array."

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

    @retry_api
    def _call_api(self, system: list, messages: list) -> str:
        """Single API call — decorated with retry_api for automatic backoff on failures."""
        response = client.messages.create(
            model=settings.agent_model,
            max_tokens=4096,
            system=system,
            messages=messages,
        )
        return response.content[0].text.strip()

    @traceable(run_type="chain")
    def analyze(self, goal: str, focus: str = "") -> list[AgentFinding]:
        """Run agent analysis with long-term memory, prompt caching, and retry logic."""

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

        system = [
            {
                "type": "text",
                "text": self.AGENT_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": files_block,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": dynamic_block,
                    },
                ],
            }
        ]

        # API call with retry
        raw = self._call_api(system, messages)

        # JSON parsing with retry — re-prompts if output is malformed
        def reprompt() -> str:
            repair_messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": self.JSON_REPAIR_PROMPT},
            ]
            return self._call_api(system, repair_messages)

        findings = parse_json_with_retry(raw, kind="array", reprompt_fn=reprompt)

        for f in findings:
            f["agent"] = self.name
        return findings
