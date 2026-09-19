"""
Goal Suggester — pre-planner layer that runs BEFORE the orchestrator.

Gathers context from:
  - Recent git commits in the Argus Agent repo
  - Open GitHub issues on argus-agent
  - Discovered file structure (what domains exist)
  - Recent run history (avoid repeating already-covered goals)

Then asks Claude (sonnet) to suggest 5 specific, actionable audit goals
for the user to choose from.
"""
import json
import logging
import subprocess

import anthropic

from config.settings import settings

logger = logging.getLogger(__name__)


# ── Context gatherers ─────────────────────────────────────────────────────────

def _recent_commits(repo_path: str, n: int = 15) -> list[str]:
    """Return the last N commit subject lines from the argus-agent local repo."""
    try:
        result = subprocess.run(
            ["git", "log", f"--pretty=format:%s", f"-{n}"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return [l.strip() for l in result.stdout.splitlines() if l.strip()]
    except Exception as exc:
        logger.debug("git log failed: %s", exc)
    return []


def _changed_files(repo_path: str, n: int = 20) -> list[str]:
    """Return file paths that changed in the last N commits."""
    try:
        result = subprocess.run(
            ["git", "log", f"--name-only", "--pretty=format:", f"-{n}"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            seen: set[str] = set()
            files = []
            for line in result.stdout.splitlines():
                line = line.strip()
                if line and line not in seen:
                    seen.add(line)
                    files.append(line)
            return files[:20]
    except Exception as exc:
        logger.debug("git log --name-only failed: %s", exc)
    return []


def _open_issues(token: str, org: str, repo: str) -> list[str]:
    """Return open GitHub issue titles (excludes PRs)."""
    try:
        import httpx
        resp = httpx.get(
            f"https://api.github.com/repos/{org}/{repo}/issues",
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
            },
            params={"state": "open", "per_page": 15},
            timeout=10,
        )
        if resp.status_code == 200:
            return [
                issue["title"]
                for issue in resp.json()
                if "pull_request" not in issue
            ]
    except Exception as exc:
        logger.debug("GitHub issues fetch failed: %s", exc)
    return []


# ── Main entry point ──────────────────────────────────────────────────────────

def suggest_goals(
    agent_files: dict[str, list[str]],
    recent_runs: list[dict] | None = None,
) -> list[str]:
    """
    Ask Claude to suggest 5 specific, actionable audit goals based on available context.

    Args:
        agent_files:  dict from discover_files() — {agent_name: [file_paths]}
        recent_runs:  recent run rows from get_run_history() — used to avoid repeats

    Returns:
        List of 5 goal strings. Falls back to hardcoded defaults on any error.
    """
    context_parts: list[str] = []

    # 1. Recent commits + changed files
    if settings.argus_repo_path:
        commits = _recent_commits(settings.argus_repo_path)
        if commits:
            context_parts.append(
                "Recent commits to Argus Agent:\n"
                + "\n".join(f"  - {c}" for c in commits)
            )
        changed = _changed_files(settings.argus_repo_path)
        if changed:
            context_parts.append(
                "Files changed recently:\n"
                + "\n".join(f"  - {f}" for f in changed)
            )

    # 2. Open GitHub issues
    if settings.github_token:
        issues = _open_issues(settings.github_token, settings.github_org, settings.argus_repo)
        if issues:
            context_parts.append(
                "Open GitHub issues on argus-agent:\n"
                + "\n".join(f"  - {i}" for i in issues)
            )

    # 3. Discovered file structure
    if agent_files:
        from pathlib import Path
        lines = []
        for agent, files in agent_files.items():
            if files:
                names = [Path(f).name for f in files[:4]]
                suffix = f" (+{len(files) - 4} more)" if len(files) > 4 else ""
                lines.append(f"  {agent}: {', '.join(names)}{suffix}")
        if lines:
            context_parts.append("Discovered source files by domain:\n" + "\n".join(lines))

    # 4. Recent run goals (avoid repetition)
    if recent_runs:
        goals_done = [r["goal"] for r in recent_runs[:6] if r.get("goal")]
        if goals_done:
            context_parts.append(
                "Goals already run recently (avoid duplicating unless evidence of regression):\n"
                + "\n".join(f"  - {g}" for g in goals_done)
            )

    context = "\n\n".join(context_parts) if context_parts else "No external context available."

    prompt = f"""You are the pre-planner for Argus Cortex — an AI pipeline that audits \
Argus Agent (Spring Boot backend, React 18 frontend, Python Flask ML service, \
Docker Compose, Jenkins CI/CD).

Based on the context below, suggest exactly 5 specific, actionable audit goals.
Rules:
- Each goal is ONE sentence a developer can read in 3 seconds and know exactly what will be audited.
- Cover diverse areas: security, performance, code quality, infrastructure, CI/CD, dependencies.
- Prioritize areas with recent changes or known issues.
- Avoid repeating recently-run goals unless the context shows a likely regression.
- Goals should be concrete (e.g. "Audit Spring Boot auth layer for JWT expiry and RBAC gaps") \
not generic ("Review the code").

Context:
{context}

Return ONLY a valid JSON array of exactly 5 strings. No markdown, no explanation.
Example format: ["Goal one", "Goal two", "Goal three", "Goal four", "Goal five"]"""

    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model=settings.orchestrator_model,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        goals = json.loads(raw)
        if isinstance(goals, list) and len(goals) >= 1:
            return goals[:5]
    except Exception as exc:
        logger.warning("Goal suggester Claude call failed: %s", exc)

    # ── Hardcoded fallback ────────────────────────────────────────────────────
    return [
        "Full security audit — OWASP Top 10, JWT handling, and secret exposure across all services",
        "Review Spring Boot service layer for N+1 queries, missing transactions, and error handling gaps",
        "Audit React frontend for XSS vectors, stale state bugs, and accessibility regressions",
        "Check Docker Compose and Nginx config for exposed ports, missing health checks, and resource limits",
        "Dependency audit — scan Maven, pip, and npm for known CVEs and outdated packages",
    ]
