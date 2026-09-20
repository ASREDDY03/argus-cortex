"""
Test Writer node - runs after Human Review, before PR Creator.

For each approved finding with actual code changes (old_code + new_code),
generates a test that verifies the fix. Tests are committed to the same
branch as the fix so the PR includes both the patch and the coverage.

Framework detection by file extension:
  .java          -> JUnit 5
  .py            -> pytest
  .tsx/.ts/.jsx/.js -> Jest
  other          -> skipped
"""
import logging
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from memory.state import CortexState
from config.settings import settings

logger = logging.getLogger(__name__)

_FRAMEWORKS = {
    ".java": "JUnit 5",
    ".py":   "pytest",
    ".tsx":  "Jest + React Testing Library",
    ".ts":   "Jest",
    ".jsx":  "Jest + React Testing Library",
    ".js":   "Jest",
}

TEST_WRITER_TOOL = {
    "name": "write_tests",
    "description": "Write tests that verify the applied code fixes.",
    "input_schema": {
        "type": "object",
        "properties": {
            "tests": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "test_file_path": {
                            "type": "string",
                            "description": "Relative path for the test file in the repo",
                        },
                        "content": {
                            "type": "string",
                            "description": "Complete, runnable test file content",
                        },
                        "framework": {
                            "type": "string",
                            "description": "Test framework used, e.g. JUnit 5, pytest, Jest",
                        },
                        "source_file": {
                            "type": "string",
                            "description": "Source file this test covers",
                        },
                    },
                    "required": ["test_file_path", "content", "framework", "source_file"],
                },
            }
        },
        "required": ["tests"],
    },
}

_llm = ChatAnthropic(
    model=settings.orchestrator_model,
    api_key=settings.anthropic_api_key,
    max_tokens=4096,
    max_retries=3,
)
_llm_with_tools = _llm.bind_tools(
    [TEST_WRITER_TOOL],
    tool_choice={"type": "tool", "name": "write_tests"},
)


def _test_file_path(source_path: str) -> str | None:
    """Derive the test file path from the source file path."""
    p = Path(source_path)
    ext = p.suffix.lower()

    if ext == ".java":
        # src/main/java/com/example/Foo.java -> src/test/java/com/example/FooCortexTest.java
        parts = list(p.parts)
        try:
            parts[parts.index("main")] = "test"
        except ValueError:
            pass
        return str(Path(*parts[:-1]) / (p.stem + "CortexTest" + ext))

    if ext == ".py":
        # app/services/auth.py -> tests/test_auth_cortex.py
        return f"tests/test_{p.stem}_cortex.py"

    if ext in (".tsx", ".ts", ".jsx", ".js"):
        # src/components/Dashboard.tsx -> src/components/__tests__/Dashboard.cortex.test.tsx
        return str(p.parent / "__tests__" / f"{p.stem}.cortex.test{ext}")

    return None


def _group_by_file(findings: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for f in findings:
        file = f.get("file", "")
        if file:
            groups.setdefault(file, []).append(f)
    return groups


@traceable(run_type="chain", name="test_writer")
def _generate_for_file(
    source_file: str,
    findings: list[dict],
    framework: str,
    test_path: str,
) -> tuple[list[dict], int, int]:
    """Ask Claude haiku to write tests for one source file."""
    findings_block = "\n\n".join(
        f"Fix {i + 1}: {f.get('description', '')}\n"
        f"Before:\n{f.get('old_code', '').strip()}\n"
        f"After:\n{f.get('new_code', '').strip()}"
        for i, f in enumerate(findings)
    )

    system = (
        f"You are a test engineer writing {framework} tests.\n"
        "Write tests that verify each applied fix works correctly.\n"
        "Each test must be focused, use realistic values, and clearly guard against the bug that was fixed.\n"
        "Do not test unrelated functionality.\n"
        "Return one complete, runnable test file."
    )

    human = (
        f"Source file: {source_file}\n"
        f"Test file to create: {test_path}\n"
        f"Framework: {framework}\n\n"
        f"These fixes were applied to {source_file}:\n\n"
        f"{findings_block}\n\n"
        f"Write {framework} tests that verify each fix using the write_tests tool."
    )

    try:
        response = _llm_with_tools.invoke([
            SystemMessage(content=system),
            HumanMessage(content=human),
        ])
        usage = response.usage_metadata or {}
        tin  = usage.get("input_tokens", 0)
        tout = usage.get("output_tokens", 0)
        if response.tool_calls:
            tests = response.tool_calls[0]["args"].get("tests", [])
            return tests, tin, tout
    except Exception as exc:
        logger.warning("[test_writer] Generation failed for %s: %s", source_file, exc)

    return [], 0, 0


def run_test_writer(state: CortexState) -> dict:
    """LangGraph node: Test Writer."""
    approved = state.get("approved_findings", [])

    patchable = [
        f for f in approved
        if f.get("old_code") and f.get("new_code") and f.get("pr_ready")
    ]

    if not patchable:
        return {"generated_tests": [], "orch_tokens_in": 0, "orch_tokens_out": 0}

    grouped = _group_by_file(patchable)
    all_tests: list[dict] = []
    total_in = total_out = 0

    for source_file, findings in grouped.items():
        ext = Path(source_file).suffix.lower()
        framework = _FRAMEWORKS.get(ext)
        test_path = _test_file_path(source_file)

        if not framework or not test_path:
            logger.info("[test_writer] No framework for %s, skipping.", source_file)
            continue

        tests, tin, tout = _generate_for_file(source_file, findings, framework, test_path)
        all_tests.extend(tests)
        total_in  += tin
        total_out += tout

    return {
        "generated_tests": all_tests,
        "orch_tokens_in":  total_in,
        "orch_tokens_out": total_out,
    }
