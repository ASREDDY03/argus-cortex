"""
Base class for all Generator agents.
Each agent uses claude-haiku-4-5, reads code from the Argus Agent repo,
and queries long-term memory to avoid re-reporting known issues.

Prompt caching strategy:
  - System prompt     → cached (static across all runs)
  - File contents     → cached (large, rarely change — biggest cost saving)
  - Goal/focus/known  → NOT cached (changes every run)

LangSmith tracing:
  - @traceable wraps analyze() as a named chain span
  - ChatAnthropic auto-traces every LLM call inside it (prompt, response, tokens, cost)
  - Result: full nested trace visible in LangSmith UI
"""
from pathlib import Path
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from memory.state import AgentFinding
from memory.long_term import get_past_findings_for_files
from orchestrator.deduplicator import deduplicate
from config.settings import settings

# Tool schema — forces the model to return findings in a guaranteed structure.
FINDINGS_TOOL = {
    "name": "report_findings",
    "description": "Report all code review findings discovered in the analyzed files.",
    "input_schema": {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "description": "List of findings. Empty array if no issues found.",
                "items": {
                    "type": "object",
                    "properties": {
                        "file":          {"type": "string",  "description": "Relative file path"},
                        "line":          {"type": ["integer", "null"], "description": "Line number, or null if not applicable"},
                        "severity":      {"type": "string",  "enum": ["critical", "high", "medium", "low"]},
                        "category":      {"type": "string",  "enum": ["bug", "security", "performance", "duplication", "style", "enhancement"]},
                        "description":   {"type": "string",  "description": "Specific description of the issue"},
                        "suggested_fix": {"type": "string",  "description": "One-line summary of the fix"},
                        "old_code":      {"type": "string",  "description": "Exact lines to replace, copy-pasted from the file. Empty string if not applicable."},
                        "new_code":      {"type": "string",  "description": "Replacement lines. Empty string if not applicable."},
                        "pr_ready":      {"type": "boolean", "description": "True only if old_code and new_code are both non-empty and the change is safe to apply"},
                    },
                    "required": ["file", "line", "severity", "category", "description", "suggested_fix", "old_code", "new_code", "pr_ready"],
                },
            }
        },
        "required": ["findings"],
    },
}

AGENT_SYSTEM = """You are a specialized code review agent for the Argus Agent DevOps system.

Analyze the provided code and report all findings using the report_findings tool.

Rules:
- Be specific — include file + line number whenever possible
- old_code must be the EXACT text from the file (copy-paste it) — it will be used for find-and-replace
- old_code must be unique in the file — include enough surrounding lines to make it unique
- Do NOT re-report issues listed in the KNOWN ISSUES section
- Focus on NEW issues not previously found"""

# Module-level model — reused across all agent instances.
# max_retries=3 handles rate limits and transient errors automatically.
_llm = ChatAnthropic(
    model=settings.agent_model,
    api_key=settings.anthropic_api_key,
    max_tokens=2048,
    max_retries=3,
)
_llm_with_tools = _llm.bind_tools(
    [FINDINGS_TOOL],
    tool_choice={"type": "tool", "name": "report_findings"},
)


class BaseAgent:
    name: str = "base_agent"
    domain: str = ""
    files_to_review: list[str] = []
    system_prompt: str = AGENT_SYSTEM   # subclasses can override for domain-specific instructions

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
            pr_url = p.get("pr_url")
            pr_state = p.get("pr_state")
            close_reason = (p.get("close_reason") or "").strip()
            if pr_url:
                if pr_state == "merged":
                    status = "FIXED — PR merged"
                elif pr_state == "closed":
                    if close_reason:
                        status = f"PR closed — reviewer note: \"{close_reason[:150]}\""
                    else:
                        status = f"PR closed without merge"
                else:
                    status = f"PR open: {pr_url}"
            else:
                status = "reported, no PR yet"
            lines.append(f"- [{p['severity']}] {p['file']} — {p['description']} ({status})")
        return "\n".join(lines)

    def _call_api(self, messages: list) -> tuple[dict, int, int]:
        """Invoke the LangChain model — returns (result, tokens_in, tokens_out)."""
        response = _llm_with_tools.invoke(messages)
        usage = response.usage_metadata or {}
        tokens_in  = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)
        if response.tool_calls:
            return response.tool_calls[0]["args"], tokens_in, tokens_out
        return {"findings": []}, tokens_in, tokens_out

    @traceable(run_type="chain")
    def analyze(self, goal: str, focus: str = "") -> tuple[list[AgentFinding], int, int]:
        """Run agent analysis with long-term memory, prompt caching, and structured output."""

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

        messages = [
            SystemMessage(content=[{
                "type": "text",
                "text": self.system_prompt,
                "cache_control": {"type": "ephemeral"},
            }]),
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": files_block,
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": dynamic_block,
                },
            ]),
        ]

        result, tokens_in, tokens_out = self._call_api(messages)
        findings = result.get("findings", [])
        for f in findings:
            f["agent"] = self.name
        return findings, tokens_in, tokens_out

    def run_node(self, state: "CortexState") -> dict:
        """
        Standard LangGraph node implementation shared by all agents.
        Handles early-return, file discovery, focus, and token tracking.
        Each agent's run_* function just calls: return AgentClass().run_node(state)
        """
        from memory.state import CortexState  # avoid circular at module level
        if self.name not in state.get("agents_to_run", []):
            return {"findings": [], "suppressed_count": 0, "agent_tokens_in": 0, "agent_tokens_out": 0}

        agent_files_map = state.get("agent_files", {})
        discovered = agent_files_map.get(self.name, [])

        # Discovery ran but found nothing for this agent — skip to save tokens
        if agent_files_map and not discovered:
            return {"findings": [], "suppressed_count": 0, "agent_tokens_in": 0, "agent_tokens_out": 0}

        if discovered:
            self.files_to_review = discovered
        focus = state.get("agent_focus", {}).get(self.name, "")
        findings, tokens_in, tokens_out = self.analyze(state["goal"], focus=focus)

        # Deduplicate against past findings — suppress issues already reported
        # with an open PR or approved in a prior run (only suppress active issues;
        # merged/closed PRs mean the issue may be a regression worth re-reporting).
        past = get_past_findings_for_files([f["file"] for f in findings if f.get("file")])
        findings, suppressed = deduplicate(findings, past)

        return {
            "findings": findings,
            "suppressed_count": suppressed,
            "agent_tokens_in": tokens_in,
            "agent_tokens_out": tokens_out,
        }
