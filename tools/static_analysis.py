"""
Static analysis runner — executes deterministic tools against the Argus Agent repo
before generator agents run. Results are injected as verified facts into agent context,
giving agents CVE IDs and specific line numbers that LLMs alone cannot produce.

Tools run (silently skipped if not installed):
  trivy   — CVE scan on Dockerfiles, docker-compose image references, requirements.txt, pom.xml
  bandit  — Python security lints (CWE violations with line numbers)
  eslint  -- JS/JSX lint for XSS-class rules (no-eval, no-dangerouslySetInnerHTML)

Each tool returns a plain-text summary sized for prompt injection (not raw JSON).
"""
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_FINDINGS = 20   # cap per tool so we don't blow up prompts


def _run(cmd: list[str], cwd: str, timeout: int = 60) -> tuple[int, str, str]:
    """Run a subprocess, return (returncode, stdout, stderr). Never raises."""
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except FileNotFoundError:
        return -1, "", f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return -1, "", f"{cmd[0]}: timed out after {timeout}s"
    except Exception as exc:
        return -1, "", str(exc)


def run_trivy(repo_path: str) -> str:
    """
    Run trivy fs scan on the repo. Returns a plain-text summary of CVEs found,
    or empty string if trivy is not installed or no issues found.
    """
    rc, out, err = _run(
        ["trivy", "fs", "--format", "json", "--scanners", "vuln,secret", "--quiet", "."],
        cwd=repo_path,
        timeout=120,
    )
    if rc == -1:
        logger.debug("[trivy] %s", err)
        return ""
    try:
        data = json.loads(out)
    except Exception:
        return ""

    lines = []
    for result in data.get("Results", []):
        target = result.get("Target", "")
        for vuln in (result.get("Vulnerabilities") or [])[:_MAX_FINDINGS]:
            vid    = vuln.get("VulnerabilityID", "")
            pkg    = vuln.get("PkgName", "")
            sev    = vuln.get("Severity", "")
            title  = vuln.get("Title") or vuln.get("Description", "")[:80]
            fixed  = vuln.get("FixedVersion", "not yet fixed")
            lines.append(f"  [{sev}] {target} — {pkg}: {vid} — {title} (fix: {fixed})")
        for secret in (result.get("Secrets") or [])[:_MAX_FINDINGS]:
            lines.append(
                f"  [SECRET] {target} line {secret.get('StartLine', '?')} — "
                f"{secret.get('Title', secret.get('RuleID', ''))}"
            )

    if not lines:
        return ""
    return "Trivy scan results (verified CVEs and secrets):\n" + "\n".join(lines[:_MAX_FINDINGS])


def run_bandit(repo_path: str) -> str:
    """
    Run bandit on Python files. Returns plain-text summary of security issues.
    """
    rc, out, _ = _run(
        ["bandit", "-r", ".", "-f", "json", "-q", "--exclude", ".venv,venv,.ml"],
        cwd=repo_path,
        timeout=60,
    )
    if rc == -1:
        return ""
    try:
        data = json.loads(out)
    except Exception:
        return ""

    issues = data.get("results", [])[:_MAX_FINDINGS]
    if not issues:
        return ""

    lines = ["Bandit security scan (Python):"]
    for issue in issues:
        fname = issue.get("filename", "").replace(repo_path, "").lstrip("/")
        lines.append(
            f"  [{issue.get('issue_severity', '')}] {fname}:{issue.get('line_number', '?')} — "
            f"{issue.get('issue_text', '')} ({issue.get('test_id', '')})"
        )
    return "\n".join(lines)


def run_eslint(repo_path: str) -> str:
    """
    Run eslint on React frontend JS/JSX files for XSS-class rules.
    """
    frontend_path = Path(repo_path) / "Devops" / "react-frontend"
    if not frontend_path.exists():
        # try root level
        frontend_path = Path(repo_path)

    rc, out, _ = _run(
        [
            "npx", "eslint", "--format", "json",
            "--rule", '{"no-eval": "error", "no-implied-eval": "error"}',
            "--ext", ".js,.jsx,.ts,.tsx",
            str(frontend_path),
        ],
        cwd=repo_path,
        timeout=60,
    )
    if rc == -1:
        return ""

    try:
        results = json.loads(out)
    except Exception:
        return ""

    lines = ["ESLint scan (XSS-class rules):"]
    found = 0
    for file_result in results:
        fname = file_result.get("filePath", "").replace(repo_path, "").lstrip("/")
        for msg in file_result.get("messages", [])[:5]:
            lines.append(
                f"  [{'ERROR' if msg.get('severity') == 2 else 'WARN'}] "
                f"{fname}:{msg.get('line', '?')} — {msg.get('message', '')} ({msg.get('ruleId', '')})"
            )
            found += 1
            if found >= _MAX_FINDINGS:
                break
        if found >= _MAX_FINDINGS:
            break

    if found == 0:
        return ""
    return "\n".join(lines)


def run_all(repo_path: str) -> dict[str, str]:
    """
    Run all static analysis tools. Returns {tool_name: summary_text}.
    Keys with empty string values mean the tool was skipped or found nothing.
    """
    if not repo_path or not Path(repo_path).exists():
        return {}
    return {
        "trivy":  run_trivy(repo_path),
        "bandit": run_bandit(repo_path),
        "eslint": run_eslint(repo_path),
    }
