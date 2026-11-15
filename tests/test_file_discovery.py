"""
Tests for tools/file_discovery.py — uses tmp_path for directory I/O.
"""
import pytest
from pathlib import Path
from tools.file_discovery import _is_excluded, _is_test_file, discover_files


# ── _is_excluded ───────────────────────────────────────────────────────────────

class TestIsExcluded:
    def test_excludes_node_modules(self):
        assert _is_excluded(Path("react-frontend/node_modules/lodash/index.js"))

    def test_excludes_target_directory(self):
        assert _is_excluded(Path("springboot-backend/target/classes/App.class"))

    def test_excludes_pycache(self):
        assert _is_excluded(Path("ml-service/__pycache__/app.cpython-310.pyc"))

    def test_excludes_venv(self):
        assert _is_excluded(Path("ml-service/.venv/lib/python3.10/site.py"))

    def test_keeps_normal_source_file(self):
        assert not _is_excluded(Path("springboot-backend/src/main/java/App.java"))

    def test_keeps_top_level_file(self):
        assert not _is_excluded(Path("docker-compose.yml"))

    def test_excludes_git_directory(self):
        assert _is_excluded(Path(".git/HEAD"))


# ── _is_test_file ──────────────────────────────────────────────────────────────

class TestIsTestFile:
    def test_identifies_java_test(self):
        assert _is_test_file(Path("JenkinsServiceTest.java"))

    def test_identifies_jest_spec(self):
        assert _is_test_file(Path("JenkinsService.test.js"))

    def test_identifies_pytest_file(self):
        assert _is_test_file(Path("test_app.py"))

    def test_identifies_spec_file(self):
        assert _is_test_file(Path("JenkinsSpec.java"))

    def test_normal_source_is_not_test(self):
        assert not _is_test_file(Path("JenkinsService.java"))

    def test_app_py_is_not_test(self):
        assert not _is_test_file(Path("app.py"))

    def test_component_jsx_is_not_test(self):
        assert not _is_test_file(Path("JenkinsDashboardComponent.jsx"))


# ── discover_files ─────────────────────────────────────────────────────────────

class TestDiscoverFiles:
    def test_returns_empty_dict_for_nonexistent_path(self):
        result = discover_files("/nonexistent/path/that/does/not/exist")
        assert result == {}

    def test_returns_empty_dict_for_empty_string(self):
        result = discover_files("")
        assert result == {}

    def test_classifies_java_files(self, tmp_path):
        java_dir = tmp_path / "Devops" / "springboot-backend" / "src" / "main" / "java"
        java_dir.mkdir(parents=True)
        (java_dir / "JenkinsService.java").write_text("public class JenkinsService {}")
        result = discover_files(str(tmp_path))
        assert "springboot_agent" in result
        assert any("JenkinsService.java" in f for f in result["springboot_agent"])

    def test_classifies_python_ml_files(self, tmp_path):
        ml_dir = tmp_path / "Devops" / "ml-service"
        ml_dir.mkdir(parents=True)
        (ml_dir / "app.py").write_text("from flask import Flask")
        result = discover_files(str(tmp_path))
        assert "ml_agent" in result
        assert any("app.py" in f for f in result["ml_agent"])

    def test_excludes_node_modules(self, tmp_path):
        nm_dir = tmp_path / "Devops" / "react-frontend" / "node_modules" / "react" / "src"
        nm_dir.mkdir(parents=True)
        (nm_dir / "index.js").write_text("module.exports = {}")
        result = discover_files(str(tmp_path))
        for files in result.values():
            assert not any("node_modules" in f for f in files)
