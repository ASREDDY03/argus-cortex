"""
Finding deduplication utility — called by BaseAgent.run_node() before returning.

Problem it solves:
  Repeated Cortex runs on the same codebase produce duplicate findings for
  issues that already have an open PR or were approved in a prior run.
  These clutter human review and waste evaluator tokens.

How it works:
  1. Accept a list of fresh findings from one agent pass.
  2. Accept past findings from long-term memory for the same files.
  3. For each fresh finding, compute Jaccard similarity of description word
     sets against every past finding for the same file + category.
  4. If similarity ≥ SIMILARITY_THRESHOLD AND the past finding's issue is
     still active (open PR, or approved but no PR yet) — suppress the fresh
     finding as a known duplicate.
  5. Return (kept_findings, suppressed_count).

Suppression only happens for ACTIVE issues (open PRs, approved-pending).
If a past PR was merged or closed, the issue is considered resolved and the
agent's new finding is kept — it may be a regression or re-introduced bug.

Similarity threshold is conservative (0.55) to avoid false positives:
two different bugs in the same file and category should both be reported.

Called from agents/base.py — no graph node needed; each agent deduplicates
its own output before the findings reducer accumulates them into state.
"""
import logging
import re

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.55


def _word_set(text: str) -> set[str]:
    """Lowercase alphanumeric tokens from a string, ignoring stop words."""
    stops = {"the", "a", "an", "in", "is", "it", "of", "to", "and", "or", "for", "on", "at"}
    return set(re.findall(r"[a-z0-9]+", text.lower())) - stops


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _is_active(past: dict) -> bool:
    """
    True when the past finding represents an issue that is still open/pending.
    We suppress current findings that duplicate active past ones.
    We do NOT suppress if the past PR was merged or closed — the agent may
    have found a regression.
    """
    pr_state = past.get("pr_state")
    pr_url = past.get("pr_url")
    # Has a PR that is still open (or state not yet synced from GitHub)
    if pr_url and pr_state in (None, "open"):
        return True
    # Was approved but no PR opened yet (pending human action in dashboard)
    if not pr_url and past.get("approved"):
        return True
    return False


def deduplicate(
    fresh_findings: list[dict],
    past_findings: list[dict],
) -> tuple[list[dict], int]:
    """
    Filter fresh_findings against past_findings.

    Returns:
        (kept, suppressed_count)
        kept            — findings that are genuinely new
        suppressed_count — how many were dropped as known duplicates
    """
    if not past_findings:
        return fresh_findings, 0

    # Index past by (file, category) for O(1) lookup per finding
    past_index: dict[tuple[str, str], list[dict]] = {}
    for p in past_findings:
        key = (p.get("file", ""), p.get("category", ""))
        past_index.setdefault(key, []).append(p)

    kept: list[dict] = []
    suppressed = 0

    for finding in fresh_findings:
        file = finding.get("file", "")
        category = finding.get("category", "")
        desc_words = _word_set(finding.get("description", ""))

        duplicate_of = None
        for p in past_index.get((file, category), []):
            if not _is_active(p):
                continue
            sim = _jaccard(desc_words, _word_set(p.get("description", "")))
            if sim >= SIMILARITY_THRESHOLD:
                duplicate_of = p
                break

        if duplicate_of:
            suppressed += 1
            pr_ref = duplicate_of.get("pr_url") or "approved, no PR yet"
            logger.info(
                f"[dedup] Suppressed: {file!r} / {category!r} — "
                f"sim={_jaccard(desc_words, _word_set(duplicate_of.get('description', ''))):.2f}, "
                f"active issue: {pr_ref}"
            )
        else:
            kept.append(finding)

    if suppressed:
        logger.info(
            f"[dedup] {len(fresh_findings)} in → {len(kept)} kept, "
            f"{suppressed} suppressed as known duplicates."
        )

    return kept, suppressed
