"""
GitHub tool — applies all agent patches to ONE shared branch on Argus Agent repo,
then opens ONE consolidated draft PR.

Flow:
  1. Create single branch: cortex/run/<run-id>
  2. Commit all patched files from all agents to that branch
  3. Open one draft PR with full diff breakdown by agent
  4. User reviews everything together and merges when satisfied
"""
import logging
import time
import base64
from datetime import datetime
from github import Github, GithubException
from tools.diff_generator import apply_patches, PatchResult
from config.settings import settings

logger = logging.getLogger(__name__)


def get_repo():
    return Github(settings.github_token).get_repo(
        f"{settings.github_org}/{settings.argus_repo}"
    )


def sync_pr_states() -> list[dict]:
    """
    Check GitHub for the current state of every open Cortex PR and write the
    outcome back to long-term memory.

    Returns a list of result dicts, one per PR checked:
      {
        "pr_url":    str   — full GitHub PR URL
        "pr_number": int   — PR number
        "state":     str   — 'open' | 'merged' | 'closed'
        "title":     str   — PR title
        "error":     str   — non-empty if the PR could not be fetched
      }

    PR states written to SQLite:
      'open'   — PR is still open / under review
      'merged' — PR was merged (issue fixed — deduplicator won't suppress regressions)
      'closed' — PR was closed without merge (fix rejected)
    """
    from memory.long_term import get_findings_with_open_prs, update_pr_state_by_url

    rows = get_findings_with_open_prs()
    if not rows:
        return []

    unique_urls: list[str] = list({r["pr_url"] for r in rows})

    try:
        repo = get_repo()
    except Exception as e:
        logger.warning(f"[sync] Could not connect to GitHub: {e}")
        return [{"pr_url": url, "pr_number": 0, "state": "", "title": "", "error": str(e)}
                for url in unique_urls]

    results: list[dict] = []
    for url in unique_urls:
        try:
            pr_number = int(url.rstrip("/").split("/")[-1])
            pr = repo.get_pull(pr_number)

            if pr.merged:
                state = "merged"
            elif pr.state == "closed":
                state = "closed"
            else:
                state = "open"

            update_pr_state_by_url(url, state)
            logger.info(f"[sync] PR #{pr_number} → {state}")
            results.append({
                "pr_url":    url,
                "pr_number": pr_number,
                "state":     state,
                "title":     pr.title,
                "error":     "",
            })
        except Exception as e:
            logger.warning(f"[sync] Could not check PR {url}: {e}")
            results.append({
                "pr_url":    url,
                "pr_number": 0,
                "state":     "",
                "title":     "",
                "error":     str(e),
            })

    return results


def wait_for_ci(repo, branch_name: str) -> dict:
    """
    Poll GitHub Actions workflow runs on branch_name until all complete.

    Returns:
      {
        "skipped":  bool  — True if no workflows found (no CI configured)
        "passed":   bool  — True if all checks concluded with 'success'
        "summary":  str   — e.g. "3/3 checks passed" or "1/3 checks failed: build"
        "checks":   list  — [{"name": str, "conclusion": str, "url": str}]
      }

    Waits up to settings.ci_timeout seconds total.
    Polls every 20s after an initial 30s grace period for runs to appear.
    """
    timeout = settings.ci_timeout
    deadline = time.time() + timeout

    # Give GitHub Actions up to 30s to register the workflow run after the push
    logger.info(f"[ci] Waiting for CI runs to appear on branch '{branch_name}'...")
    runs = []
    appear_deadline = time.time() + 30
    while time.time() < appear_deadline:
        runs = list(repo.get_workflow_runs(branch=branch_name, event="push"))
        if runs:
            break
        time.sleep(5)

    if not runs:
        logger.info("[ci] No workflow runs found — CI not configured, proceeding without validation.")
        return {"skipped": True, "passed": True, "summary": "No CI configured — skipped", "checks": []}

    logger.info(f"[ci] Found {len(runs)} workflow run(s) — polling for completion (timeout: {timeout}s)...")

    # Poll until all runs complete or timeout
    while time.time() < deadline:
        run_ids = [r.id for r in runs]
        runs = [repo.get_workflow_run(rid) for rid in run_ids]  # refresh from GitHub

        pending = [r for r in runs if r.status not in ("completed", "cancelled")]
        if not pending:
            break

        remaining = int(deadline - time.time())
        logger.info(f"[ci] {len(pending)} run(s) still in progress — {remaining}s remaining...")
        time.sleep(20)

    # Collect final results
    checks = []
    for run in runs:
        run = repo.get_workflow_run(run.id)  # final refresh
        checks.append({
            "name":       run.name,
            "conclusion": run.conclusion or "timed_out",
            "url":        run.html_url,
        })

    passed_count = sum(1 for c in checks if c["conclusion"] == "success")
    all_passed = passed_count == len(checks)

    failed_names = [c["name"] for c in checks if c["conclusion"] != "success"]
    if all_passed:
        summary = f"{passed_count}/{len(checks)} CI check(s) passed"
    else:
        summary = f"{passed_count}/{len(checks)} passed — failed: {', '.join(failed_names)}"

    logger.info(f"[ci] {summary}")
    return {"skipped": False, "passed": all_passed, "summary": summary, "checks": checks}


def create_consolidated_pr(
    run_id: str,
    all_findings: list[dict],
    goal: str,
    human_notes: str = "",
) -> dict:
    """
    Commit ALL approved patches from ALL agents onto one branch,
    validate CI passes, then open a single draft PR.

    Returns:
      {
        "pr_url":     str | None  — PR URL if opened, None otherwise
        "ci_passed":  bool | None — True/False/None (None = no CI configured)
        "ci_summary": str         — human-readable CI result
        "ci_checks":  list        — per-check results
      }
    """
    pr_ready = [f for f in all_findings if f.get("pr_ready", False)]
    if not pr_ready:
        return {"pr_url": None, "ci_passed": None, "ci_summary": "", "ci_checks": []}

    repo = get_repo()
    short_id = run_id[:8]
    branch_name = f"cortex/run/{short_id}"
    base_branch = repo.default_branch
    base_sha = repo.get_branch(base_branch).commit.sha

    # Create single shared branch for this run
    repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base_sha)

    # Apply all patches locally (sequential per file — handles multi-agent edits to same file)
    patch_results = apply_patches(pr_ready)
    successful = [r for r in patch_results if r.success]

    # Commit each patched file once (last patch wins per file — apply_patches handles ordering)
    committed: list[str] = []
    files_to_commit = _final_content_per_file(successful)

    for file_path, content in files_to_commit.items():
        try:
            existing = repo.get_contents(file_path, ref=base_branch)
            repo.update_file(
                path=file_path,
                message=f"cortex: improve {file_path.split('/')[-1]}",
                content=content,
                sha=existing.sha,
                branch=branch_name,
            )
            committed.append(file_path)
        except GithubException:
            pass

    # If nothing was committed, add a summary file so the PR has content
    if not committed:
        _commit_summary(repo, branch_name, short_id, pr_ready, goal)

    # ── CI Validation ─────────────────────────────────────────────────────────
    ci_result = {"skipped": True, "passed": True, "summary": "CI validation disabled", "checks": []}

    if settings.ci_validation:
        ci_result = wait_for_ci(repo, branch_name)

    ci_passed = None if ci_result["skipped"] else ci_result["passed"]

    if not ci_result["passed"]:
        logger.warning(f"[ci] CI failed on branch '{branch_name}' — PR not opened.")
        return {
            "pr_url":     None,
            "ci_passed":  False,
            "ci_summary": ci_result["summary"],
            "ci_checks":  ci_result["checks"],
        }
    # ──────────────────────────────────────────────────────────────────────────

    body = _build_pr_body(run_id, all_findings, patch_results, goal, human_notes, ci_result)

    agents_involved = sorted({f.get("agent", "") for f in all_findings})
    title = f"[Argus Cortex] {len(pr_ready)} improvement(s) across {len(agents_involved)} domain(s) — {short_id}"

    pr = repo.create_pull(
        title=title,
        body=body,
        head=branch_name,
        base=base_branch,
        draft=True,
    )
    return {
        "pr_url":     pr.html_url,
        "ci_passed":  ci_passed,
        "ci_summary": ci_result["summary"],
        "ci_checks":  ci_result["checks"],
    }


def _final_content_per_file(results: list[PatchResult]) -> dict[str, str]:
    """Last successful patch per file — apply_patches already applied them sequentially."""
    by_file: dict[str, str] = {}
    for r in results:
        if r.patched_content:
            by_file[r.file_path] = r.patched_content
    return by_file


def _commit_summary(repo, branch_name, short_id, findings, goal):
    lines = [f"# Argus Cortex Run {short_id}\nGoal: {goal}\n"]
    for f in findings:
        lines.append(f"- [{f.get('severity')}] {f.get('file')} — {f.get('description')}")
    try:
        repo.create_file(
            path=f".cortex/runs/{short_id}.md",
            message=f"cortex: run {short_id} summary",
            content="\n".join(lines),
            branch=branch_name,
        )
    except GithubException:
        pass


def _build_pr_body(
    run_id: str,
    findings: list[dict],
    patch_results: list[PatchResult],
    goal: str,
    human_notes: str,
    ci_result: dict | None = None,
) -> str:
    severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
    pr_ready = [f for f in findings if f.get("pr_ready")]
    successful_patches = sum(1 for r in patch_results if r.success)
    agents = sorted({f.get("agent", "") for f in findings})

    lines = [
        f"## Argus Cortex — Consolidated Improvement PR",
        f"",
        f"**Goal:** {goal}",
        f"**Run ID:** `{run_id[:8]}`",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Agents:** {', '.join(f'`{a}`' for a in agents)}",
        f"**Total findings:** {len(findings)} | **Patches applied:** {successful_patches}",
    ]

    if human_notes:
        lines += [f"**Reviewer notes:** {human_notes}", ""]

    # CI validation result
    if ci_result and not ci_result.get("skipped"):
        ci_emoji = "✅" if ci_result.get("passed") else "❌"
        lines += [f"**CI:** {ci_emoji} {ci_result.get('summary', '')}"]
        for check in ci_result.get("checks", []):
            c_emoji = "✅" if check["conclusion"] == "success" else "❌"
            lines.append(f"  - {c_emoji} [{check['name']}]({check['url']}) — {check['conclusion']}")
        lines += [""]

    lines += ["", "---", ""]

    # Group findings by agent for readability
    by_agent: dict[str, list] = {}
    for f in findings:
        by_agent.setdefault(f.get("agent", "unknown"), []).append(f)

    patch_map = {r.file_path: r for r in patch_results}

    for agent_name, agent_findings in by_agent.items():
        lines += [f"## `{agent_name}`", ""]

        for i, f in enumerate(agent_findings, 1):
            emoji = severity_emoji.get(f.get("severity", "low"), "⚪")
            file_path = f.get("file", "")
            patch = patch_map.get(file_path)

            lines += [
                f"### {i}. {emoji} `{f.get('severity', '').upper()}` — {f.get('category', '')}",
                f"**File:** `{file_path}`" + (f" (line {f['line']})" if f.get("line") else ""),
                f"**Issue:** {f.get('description', '')}",
                "",
            ]

            if patch and patch.success and patch.unified_diff:
                lines += [
                    "**Diff:**",
                    "```diff",
                    patch.unified_diff,
                    "```",
                    "",
                ]
            elif f.get("old_code") and f.get("new_code"):
                lines += [
                    "**Intended change** _(could not apply automatically)_:",
                    "```diff",
                    "\n".join(f"- {l}" for l in f["old_code"].splitlines()),
                    "\n".join(f"+ {l}" for l in f["new_code"].splitlines()),
                    "```",
                    "",
                ]
            else:
                lines += [f"**Fix:** {f.get('suggested_fix', '')}", ""]

    lines += [
        "---",
        "",
        "> Draft PR generated by [Argus Cortex](https://github.com/ASREDDY03/argus-cortex).",
        "> Verify all changes, then merge when satisfied.",
    ]

    return "\n".join(lines)
