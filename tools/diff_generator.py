"""
Diff generator — applies agent patches to files and produces unified diffs.

Strategy:
  1. Read the original file content
  2. Find old_code in the file (exact string match)
  3. Replace with new_code
  4. Generate a unified diff for display in the PR body
  5. Return the patched content for committing to GitHub
"""
import difflib
from pathlib import Path
from config.settings import settings


class PatchResult:
    def __init__(self, file_path: str, success: bool, patched_content: str, unified_diff: str, error: str = ""):
        self.file_path = file_path
        self.success = success
        self.patched_content = patched_content
        self.unified_diff = unified_diff
        self.error = error


def apply_patch(finding: dict) -> PatchResult:
    """
    Apply a single finding's old_code → new_code patch to the target file.

    Returns PatchResult with:
      - success: True if patch applied cleanly
      - patched_content: full file content after patch
      - unified_diff: diff string for PR body display
      - error: reason for failure if success=False
    """
    file_path = finding.get("file", "")
    old_code = finding.get("old_code", "").strip()
    new_code = finding.get("new_code", "").strip()

    if not old_code or not new_code:
        return PatchResult(file_path, False, "", "", "old_code or new_code is empty")

    full_path = Path(settings.argus_repo_path) / file_path
    if not full_path.exists():
        return PatchResult(file_path, False, "", "", f"File not found: {file_path}")

    original = full_path.read_text(encoding="utf-8")

    # Exact match first
    if old_code not in original:
        # Try stripped line-by-line match (handles minor whitespace differences)
        normalised = _normalise_whitespace(original, old_code)
        if normalised is None:
            return PatchResult(file_path, False, "", "", "old_code not found in file — may have changed since analysis")
        original = normalised

    patched = original.replace(old_code, new_code, 1)

    diff = _unified_diff(original, patched, file_path)

    return PatchResult(file_path, True, patched, diff)


def apply_patches(findings: list[dict]) -> list[PatchResult]:
    """
    Apply all pr_ready patches. Groups by file so multiple patches
    to the same file are applied sequentially on the updated content.
    """
    # Group findings by file
    by_file: dict[str, list[dict]] = {}
    for f in findings:
        if f.get("pr_ready") and f.get("old_code") and f.get("new_code"):
            by_file.setdefault(f["file"], []).append(f)

    results = []
    for file_path, file_findings in by_file.items():
        full_path = Path(settings.argus_repo_path) / file_path
        if not full_path.exists():
            for f in file_findings:
                results.append(PatchResult(file_path, False, "", "", "File not found"))
            continue

        current_content = full_path.read_text(encoding="utf-8")

        for finding in file_findings:
            old_code = finding.get("old_code", "").strip()
            new_code = finding.get("new_code", "").strip()

            if old_code not in current_content:
                results.append(PatchResult(file_path, False, "", "", "old_code not found (may conflict with previous patch)"))
                continue

            patched = current_content.replace(old_code, new_code, 1)
            diff = _unified_diff(current_content, patched, file_path)
            results.append(PatchResult(file_path, True, patched, diff))
            current_content = patched  # next patch applies to already-patched content

    return results


def _unified_diff(original: str, patched: str, file_path: str) -> str:
    orig_lines = original.splitlines(keepends=True)
    patched_lines = patched.splitlines(keepends=True)
    diff = difflib.unified_diff(
        orig_lines,
        patched_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm="",
    )
    return "".join(diff)


def _normalise_whitespace(content: str, target: str) -> str | None:
    """
    Try to find target in content ignoring leading/trailing whitespace per line.
    Returns content with target replaced by exact version found, or None if not found.
    """
    target_lines = [l.strip() for l in target.splitlines()]
    content_lines = content.splitlines()

    for i in range(len(content_lines) - len(target_lines) + 1):
        window = [l.strip() for l in content_lines[i:i + len(target_lines)]]
        if window == target_lines:
            exact = "\n".join(content_lines[i:i + len(target_lines)])
            return content.replace(exact, target, 1)

    return None
