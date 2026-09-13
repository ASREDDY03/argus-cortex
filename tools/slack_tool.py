"""
Slack notifications via Incoming Webhooks + Block Kit.

Two notification points:
  notify_run_start()    — webhook-triggered run began (pusher, goal, run_id)
  notify_run_complete() — any run finished (findings, PR link, CI status, retry info)

Uses stdlib urllib only — no new dependencies.
Silently no-ops if SLACK_WEBHOOK_URL is not set in .env.
"""
import json
import logging
import urllib.request
from datetime import datetime

from config.settings import settings

logger = logging.getLogger(__name__)

_SEVERITY_EMOJI = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}


def _post(blocks: list, fallback_text: str) -> bool:
    """POST a Block Kit payload to the Slack Incoming Webhook URL."""
    if not settings.slack_webhook_url:
        return False
    payload = json.dumps({"text": fallback_text, "blocks": blocks}).encode()
    try:
        req = urllib.request.Request(
            settings.slack_webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        logger.warning(f"[slack] Failed to send notification: {e}")
        return False


def notify_run_start(pusher: str, goal: str, run_id: str = "") -> bool:
    """
    Notify Slack that a webhook-triggered run has started.
    Call from webhook_server.py right after _spawn_run().
    """
    short_id = run_id[:8] if run_id else "—"
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "🔄 Argus Cortex — Run Triggered"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Pushed by:* {pusher}"},
                {"type": "mrkdwn", "text": f"*Run ID:* `{short_id}`"},
            ],
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Goal:* {goal}"},
        },
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"Started {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"},
            ],
        },
    ]
    return _post(blocks, f"Argus Cortex run triggered by {pusher}")


def notify_run_complete(
    goal: str,
    run_id: str,
    approved_findings: list[dict],
    agents_used: list[str],
    pr_urls: list[str],
    ci_summary: str = "",
    retry_count: int = 0,
) -> bool:
    """
    Notify Slack that a run has finished.
    Call from main.py at the end of run(), after PRs are opened.
    """
    short_id = run_id[:8]
    n = len(approved_findings)

    # Severity breakdown
    counts: dict[str, int] = {}
    for f in approved_findings:
        sev = f.get("severity", "low")
        counts[sev] = counts.get(sev, 0) + 1

    sev_parts = [
        f"{_SEVERITY_EMOJI[s]} {counts[s]} {s}"
        for s in ("critical", "high", "medium", "low")
        if counts.get(s)
    ]
    severity_line = "  ".join(sev_parts) if sev_parts else "_none_"

    # Agents line (max 4 shown inline)
    sorted_agents = sorted(agents_used)
    agent_text = ", ".join(f"`{a}`" for a in sorted_agents[:4])
    if len(sorted_agents) > 4:
        agent_text += f" _(+{len(sorted_agents) - 4} more)_"

    # PR links
    pr_text = (
        "  ".join(f"<{url}|View PR →>" for url in pr_urls)
        if pr_urls
        else "_No PR opened_"
    )

    header_text = (
        f"✅ Argus Cortex — {n} finding(s) approved"
        if n
        else "ℹ️ Argus Cortex — Run complete (no findings)"
    )

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": header_text},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Run ID:* `{short_id}`"},
                {"type": "mrkdwn", "text": f"*Findings approved:* {n}"},
            ],
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Goal:* {goal}"},
        },
    ]

    if n:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Severity breakdown:*\n{severity_line}"},
        })

    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": f"*Agents:* {agent_text}"},
    })

    if retry_count > 0:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Retry:* ↻ {retry_count} pass — synthesizer flagged coverage gaps"},
        })

    if ci_summary:
        ci_emoji = "✅" if "passed" in ci_summary.lower() else ("⏭️" if "skip" in ci_summary.lower() else "❌")
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*CI:* {ci_emoji} {ci_summary}"},
        })

    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": f"*PR:* {pr_text}"},
    })

    blocks += [
        {"type": "divider"},
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"Completed {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"},
            ],
        },
    ]

    return _post(blocks, f"Argus Cortex run {short_id} — {n} finding(s) approved")
