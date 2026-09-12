"""
GitHub tool — applies code patches and opens PRs on the Argus Agent repo.

PR strategy:
- One PR per agent domain
- Branch: cortex/<agent>/<short-uuid>
- Commits actual patched files (not just summary markdown)
- PR body includes unified diffs for each change
- Draft PRs only — nothing merges without human approval
"""
import uuid
import base64
from datetime import datetime
from collections import defaultdict
from github import Github, GithubException
from tools.diff_generator import apply_patches, PatchResult
from config.settings import settings


def get_github_client() -> Github:
    return Github(settings.github_token)


def get_repo():
    g = get_github_client()
    return g.get_repo(f"{settings.github_org}/{settings.argus_repo}")


def read_file_from_github(file_path: str) -> tuple[str, str]:
    """Read a file from the Argus Agent GitHub repo. Returns (content, sha)."""
    repo = get_repo()
    try:
        file = repo.get_contents(file_path)
        content = base64.b64decode(file.content).decode("utf-8")
        return content, file.sha
    except GithubException as e:
        return f"[Error reading {file_path}: {e}]", ""


def create_pr_for_agent(agent_name: str, findings: list[dict], goal: str) -> str | None:
    """
    Create a draft PR with actual code changes for a specific agent's findings.
    Returns the PR URL or None if no pr_ready findings.
    """
    pr_ready = [f for f in findings if f.get("pr_ready", False)]
    if not pr_ready:
        return None

    # Apply patches locally first
    patch_results = apply_patches(pr_ready)
    successful_patches = [r for r in patch_results if r.success]

    repo = get_repo()
    short_id = str(uuid.uuid4())[:8]
    branch_name = f"cortex/{agent_name}/{short_id}"
    base_branch = repo.default_branch
    base_sha = repo.get_branch(base_branch).commit.sha

    # Create branch
    repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base_sha)

    # Commit patched files grouped by file path
    committed_files: list[str] = []
    files_by_path = _group_patches_by_file(successful_patches)

    for file_path, final_content in files_by_path.items():
        try:
            existing = repo.get_contents(file_path, ref=base_branch)
            repo.update_file(
                path=file_path,
                message=f"cortex({agent_name}): fix {file_path.split('/')[-1]}",
                content=final_content,
                sha=existing.sha,
                branch=branch_name,
            )
            committed_files.append(file_path)
        except GithubException as e:
            # If file doesn't exist on GitHub yet, create it
            try:
                repo.create_file(
                    path=file_path,
                    message=f"cortex({agent_name}): add {file_path.split('/')[-1]}",
                    content=final_content,
                    branch=branch_name,
                )
                committed_files.append(file_path)
            except GithubException:
                pass

    # If no patches applied, still commit findings summary
    if not committed_files:
        _commit_summary_fallback(repo, branch_name, agent_name, short_id, pr_ready, goal)

    # Build PR body with diffs
    body = _build_pr_body(agent_name, pr_ready, patch_results, goal)

    pr = repo.create_pull(
        title=f"[Argus Cortex] {agent_name}: {len(pr_ready)} fix(es) — {short_id}",
        body=body,
        head=branch_name,
        base=base_branch,
        draft=True,
    )

    return pr.html_url


def _group_patches_by_file(results: list[PatchResult]) -> dict[str, str]:
    """Return {file_path: final_patched_content} — last patch per file wins (they were applied sequentially)."""
    by_file: dict[str, str] = {}
    for r in results:
        if r.success and r.patched_content:
            by_file[r.file_path] = r.patched_content
    return by_file


def _commit_summary_fallback(repo, branch_name, agent_name, short_id, findings, goal):
    """Fallback: commit a markdown summary if no file patches succeeded."""
    summary_path = f".cortex/findings/{agent_name}/{short_id}.md"
    lines = [f"# {agent_name} Findings\nGoal: {goal}\nDate: {datetime.utcnow().isoformat()}\n"]
    for f in findings:
        lines.append(f"- [{f.get('severity')}] {f.get('file')} — {f.get('description')}")
    try:
        repo.create_file(
            path=summary_path,
            message=f"cortex: {agent_name} findings summary — {short_id}",
            content="\n".join(lines),
            branch=branch_name,
        )
    except GithubException:
        pass


def _build_pr_body(agent_name: str, findings: list[dict], patch_results: list[PatchResult], goal: str) -> str:
    severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}

    successful = sum(1 for r in patch_results if r.success)
    failed = len(patch_results) - successful

    lines = [
        f"## Argus Cortex — `{agent_name}`",
        f"",
        f"**Goal:** {goal}",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Findings:** {len(findings)} | **Patches applied:** {successful} | **Failed:** {failed}",
        f"",
        f"---",
        f"",
        f"## Changes",
        f"",
    ]

    # Map patch results back to findings by index
    patch_map = {r.file_path: r for r in patch_results}

    for i, f in enumerate(findings):
        emoji = severity_emoji.get(f.get("severity", "low"), "⚪")
        file_path = f.get("file", "")
        patch = patch_map.get(file_path)

        lines += [
            f"### {i + 1}. {emoji} `{f.get('severity', '').upper()}` — {f.get('category', '')}",
            f"",
            f"**File:** `{file_path}`" + (f" (line {f['line']})" if f.get("line") else ""),
            f"**Issue:** {f.get('description', '')}",
            f"",
        ]

        if patch and patch.success and patch.unified_diff:
            lines += [
                f"**Diff:**",
                f"```diff",
                patch.unified_diff,
                f"```",
                f"",
            ]
        elif f.get("old_code") and f.get("new_code"):
            # Patch failed (old_code not found) — still show intended change
            lines += [
                f"**Intended change** _(patch could not be applied automatically)_:",
                f"```diff",
                f"- {chr(10).join('- ' + l for l in f['old_code'].splitlines())}",
                f"+ {chr(10).join('+ ' + l for l in f['new_code'].splitlines())}",
                f"```",
                f"",
            ]
        else:
            lines += [
                f"**Suggested fix:** {f.get('suggested_fix', '')}",
                f"",
            ]

    lines += [
        "---",
        "",
        "> Draft PR generated by [Argus Cortex](https://github.com/ASREDDY03/argus-cortex). Review before merging.",
    ]

    return "\n".join(lines)
