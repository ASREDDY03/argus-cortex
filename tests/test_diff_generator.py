"""
Tests for tools/diff_generator.py — uses tmp_path for file I/O, patches settings.
"""
import pytest
from pathlib import Path
from unittest.mock import patch
from tools.diff_generator import apply_patch, apply_patches, _unified_diff, _normalise_whitespace


# ── _unified_diff ──────────────────────────────────────────────────────────────

class TestUnifiedDiff:
    def test_produces_diff_header(self):
        diff = _unified_diff("old line\n", "new line\n", "src/app.py")
        assert "--- a/src/app.py" in diff
        assert "+++ b/src/app.py" in diff

    def test_shows_removed_line(self):
        diff = _unified_diff("old line\n", "new line\n", "app.py")
        assert "-old line" in diff

    def test_shows_added_line(self):
        diff = _unified_diff("old line\n", "new line\n", "app.py")
        assert "+new line" in diff

    def test_empty_diff_for_identical_content(self):
        diff = _unified_diff("same\n", "same\n", "app.py")
        assert diff == ""


# ── _normalise_whitespace ──────────────────────────────────────────────────────

class TestNormaliseWhitespace:
    def test_finds_target_with_different_indentation(self):
        content = "def foo():\n    return 1\n"
        target = "def foo():\n  return 1\n"   # 2-space indent instead of 4
        result = _normalise_whitespace(content, target)
        assert result is not None

    def test_returns_none_when_not_found(self):
        content = "def foo():\n    return 1\n"
        target = "def bar():\n    return 2\n"
        assert _normalise_whitespace(content, target) is None


# ── apply_patch ────────────────────────────────────────────────────────────────

class TestApplyPatch:
    def _finding(self, file, old_code, new_code):
        return {"file": file, "old_code": old_code, "new_code": new_code, "pr_ready": True}

    def test_success_exact_match(self, tmp_path):
        (tmp_path / "app.py").write_text("x = 1\ny = 2\n")
        finding = self._finding("app.py", "x = 1", "x = 10")
        with patch("tools.diff_generator.settings") as mock_settings:
            mock_settings.argus_repo_path = str(tmp_path)
            result = apply_patch(finding)
        assert result.success
        assert "x = 10" in result.patched_content
        assert result.unified_diff != ""

    def test_failure_empty_old_code(self, tmp_path):
        finding = self._finding("app.py", "", "x = 10")
        with patch("tools.diff_generator.settings") as mock_settings:
            mock_settings.argus_repo_path = str(tmp_path)
            result = apply_patch(finding)
        assert not result.success
        assert "empty" in result.error

    def test_failure_file_not_found(self, tmp_path):
        finding = self._finding("missing.py", "x = 1", "x = 10")
        with patch("tools.diff_generator.settings") as mock_settings:
            mock_settings.argus_repo_path = str(tmp_path)
            result = apply_patch(finding)
        assert not result.success
        assert "not found" in result.error.lower()

    def test_failure_old_code_not_in_file(self, tmp_path):
        (tmp_path / "app.py").write_text("x = 1\n")
        finding = self._finding("app.py", "y = 99", "y = 100")
        with patch("tools.diff_generator.settings") as mock_settings:
            mock_settings.argus_repo_path = str(tmp_path)
            result = apply_patch(finding)
        assert not result.success


# ── apply_patches ──────────────────────────────────────────────────────────────

class TestApplyPatches:
    def test_applies_multiple_patches_sequentially(self, tmp_path):
        (tmp_path / "app.py").write_text("a = 1\nb = 2\nc = 3\n")
        findings = [
            {"file": "app.py", "old_code": "a = 1", "new_code": "a = 10", "pr_ready": True},
            {"file": "app.py", "old_code": "b = 2", "new_code": "b = 20", "pr_ready": True},
        ]
        with patch("tools.diff_generator.settings") as mock_settings:
            mock_settings.argus_repo_path = str(tmp_path)
            results = apply_patches(findings)
        assert all(r.success for r in results)
        assert len(results) == 2

    def test_skips_findings_without_pr_ready(self, tmp_path):
        (tmp_path / "app.py").write_text("x = 1\n")
        findings = [
            {"file": "app.py", "old_code": "x = 1", "new_code": "x = 10", "pr_ready": False},
        ]
        with patch("tools.diff_generator.settings") as mock_settings:
            mock_settings.argus_repo_path = str(tmp_path)
            results = apply_patches(findings)
        assert results == []

    def test_handles_missing_file(self, tmp_path):
        findings = [
            {"file": "ghost.py", "old_code": "x = 1", "new_code": "x = 10", "pr_ready": True},
        ]
        with patch("tools.diff_generator.settings") as mock_settings:
            mock_settings.argus_repo_path = str(tmp_path)
            results = apply_patches(findings)
        assert len(results) == 1
        assert not results[0].success
