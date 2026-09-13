"""
GitHub tool — applies all agent patches to ONE shared branch on Argus Agent repo,
then opens ONE consolidated draft PR.

Flow:
  1. Create single branch: cortex/run/<run-id>
  2. Commit all patched files from all agents to that branch
  3. Open one draft PR with full diff breakdown by agent
  4. User reviews everything together and merges when satisfied
"""
import base64
from datetime import datetime
from github import Github, GithubException
from tools.diff_generator import apply_patches, PatchResult
from config.settings import settings


def get_repo():
    return Github(settings.github_token).get_repo(
        f"{settings.github_org}/{settings.argus_repo}"
    )


def create_consolidated_pr(
    run_id: str,
    all_findings: list[dict],
    goal: str,
    human_notes: str = "",
) -> str | None:
    """
    Commit ALL approved patches from ALL agents onto one branch.
    Opens a single draft PR. Returns PR URL or None.
    """
    pr_ready = [f for f in all_findings if f.get("pr_ready", False)]
    if not pr_ready:
        return None

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

    body = _build_pr_body(run_id, all_findings, patch_results, goal, human_notes)

    agents_involved = sorted({f.get("agent", "") for f in all_findings})
    title = f"[Argus Cortex] {len(pr_ready)} improvement(s) across {len(agents_involved)} domain(s) — {short_id}"

    pr = repo.create_pull(
        title=title,
        body=body,
        head=branch_name,
        base=base_branch,
        draft=True,
    )
    return pr.html_url


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
