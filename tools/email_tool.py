"""
Email digest tool — sends a summary of Cortex findings via SMTP.

Configure in .env:
  SMTP_HOST=smtp.gmail.com
  SMTP_PORT=587
  SMTP_USER=you@gmail.com
  SMTP_PASSWORD=app-password
  DIGEST_TO=team@company.com,lead@company.com
  DIGEST_FROM=cortex@company.com   # optional, defaults to SMTP_USER

Silent no-op if SMTP_HOST is not set.
"""
import logging
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config.settings import settings

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    return bool(settings.smtp_host and settings.smtp_user and settings.digest_to)


def send_digest(
    subject: str,
    html_body: str,
    text_body: str,
) -> bool:
    """
    Send an email via SMTP. Returns True on success, False on failure or if disabled.
    Silent no-op if SMTP_HOST / SMTP_USER / DIGEST_TO are not configured.
    """
    if not _enabled():
        logger.debug("[email] SMTP not configured — skipping digest.")
        return False

    from_addr = settings.digest_from or settings.smtp_user
    to_addrs  = [a.strip() for a in settings.digest_to.split(",") if a.strip()]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = from_addr
    msg["To"]      = ", ".join(to_addrs)

    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body,  "html"))

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(settings.smtp_user, settings.smtp_password)
            server.sendmail(from_addr, to_addrs, msg.as_string())
        logger.info(f"[email] Digest sent to {', '.join(to_addrs)}")
        return True
    except Exception as e:
        logger.error(f"[email] Failed to send digest: {e}")
        return False


def send_run_digest(
    goal: str,
    run_id: str,
    approved_findings: list[dict],
    pr_urls: list[str],
    cost_usd: float = 0.0,
    agents_used: list[str] | None = None,
) -> bool:
    """Send a digest email summarising a single Cortex run."""
    if not _enabled():
        return False

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}

    # Count by severity
    sev_counts: dict[str, int] = {}
    for f in approved_findings:
        s = f.get("severity", "low")
        sev_counts[s] = sev_counts.get(s, 0) + 1

    sev_lines_html = "".join(
        f"<li>{severity_emoji.get(s, '⚪')} <strong>{c} {s}</strong></li>"
        for s, c in sorted(sev_counts.items(), key=lambda x: ["critical","high","medium","low"].index(x[0]) if x[0] in ["critical","high","medium","low"] else 99)
    )
    sev_lines_text = "\n".join(
        f"  {severity_emoji.get(s, '-')} {c} {s}"
        for s, c in sev_counts.items()
    )

    pr_lines_html = "".join(f'<li><a href="{u}">{u}</a></li>' for u in pr_urls) or "<li>None</li>"
    pr_lines_text = "\n".join(f"  {u}" for u in pr_urls) or "  None"

    agents_text = ", ".join(agents_used) if agents_used else "unknown"

    subject = f"[Argus Cortex] {len(approved_findings)} finding(s) — {run_id[:8]} — {now}"

    html_body = f"""
<html><body style="font-family:sans-serif;color:#1e293b;max-width:600px;margin:auto">
<h2 style="color:#0f172a">Argus Cortex Run Summary</h2>
<p><strong>Goal:</strong> {goal}</p>
<p><strong>Run ID:</strong> <code>{run_id[:8]}</code> &nbsp;·&nbsp; <strong>Generated:</strong> {now}</p>
<p><strong>Agents:</strong> {agents_text}</p>
<hr style="border:1px solid #e2e8f0">

<h3>Approved Findings ({len(approved_findings)})</h3>
<ul>{sev_lines_html}</ul>

<h3>PRs Opened ({len(pr_urls)})</h3>
<ul>{pr_lines_html}</ul>

<p style="color:#64748b">Cost: ${cost_usd:.4f} USD</p>
<hr style="border:1px solid #e2e8f0">
<p style="font-size:.8rem;color:#94a3b8">
  Sent by <a href="https://github.com/ASREDDY03/argus-cortex">Argus Cortex</a>.
  Configure recipients via <code>DIGEST_TO</code> in .env.
</p>
</body></html>
"""

    text_body = f"""Argus Cortex Run Summary
========================
Goal:    {goal}
Run ID:  {run_id[:8]}
Date:    {now}
Agents:  {agents_text}

Approved Findings ({len(approved_findings)}):
{sev_lines_text}

PRs Opened ({len(pr_urls)}):
{pr_lines_text}

Cost: ${cost_usd:.4f} USD
"""

    return send_digest(subject, html_body, text_body)


def send_weekly_digest(stats: dict) -> bool:
    """
    Send a weekly aggregate digest using stats from get_weekly_stats() +
    get_stats(). Called by `python main.py send-digest`.
    """
    if not _enabled():
        return False

    now  = datetime.utcnow().strftime("%Y-%m-%d")
    runs = stats.get("runs", {})
    findings = stats.get("findings", {})
    pr_outcomes = stats.get("pr_outcomes", {})
    by_sev = stats.get("by_severity", [])
    trend  = stats.get("trend", {})

    sev_rows_html = "".join(
        f"<tr><td>{r['severity']}</td><td>{r['total']}</td><td>{r['approved']}</td></tr>"
        for r in by_sev
    )
    sev_rows_text = "\n".join(
        f"  {r['severity']:10} {r['total']} total, {r['approved']} approved"
        for r in by_sev
    )

    last_7d  = trend.get("last_7d", 0.0)
    prev_7d  = trend.get("prev_7d", 0.0)
    delta    = last_7d - prev_7d
    delta_str = f"+${delta:.4f}" if delta >= 0 else f"-${abs(delta):.4f}"

    subject = f"[Argus Cortex] Weekly Digest — {now}"

    html_body = f"""
<html><body style="font-family:sans-serif;color:#1e293b;max-width:600px;margin:auto">
<h2>Argus Cortex — Weekly Digest</h2>
<p><strong>Week ending:</strong> {now}</p>
<hr style="border:1px solid #e2e8f0">

<h3>Runs</h3>
<p>{runs.get('total_runs', 0)} total &nbsp;·&nbsp;
   {runs.get('completed_runs', 0)} completed &nbsp;·&nbsp;
   ${runs.get('total_cost', 0):.4f} total cost</p>

<h3>Findings</h3>
<p>{findings.get('total', 0)} total &nbsp;·&nbsp;
   {findings.get('total_approved', 0)} approved &nbsp;·&nbsp;
   {findings.get('total_prs_opened', 0)} PRs opened</p>
<table border="1" cellpadding="4" style="border-collapse:collapse;font-size:.85rem">
<tr><th>Severity</th><th>Total</th><th>Approved</th></tr>
{sev_rows_html}
</table>

<h3>PR Outcomes</h3>
<p>
  ✔ Merged: {pr_outcomes.get('merged', 0)} &nbsp;·&nbsp;
  ● Open: {pr_outcomes.get('open', 0)} &nbsp;·&nbsp;
  ✘ Closed: {pr_outcomes.get('closed', 0)}
</p>

<h3>Cost Trend</h3>
<p>Last 7d: ${last_7d:.4f} &nbsp;·&nbsp; Prior 7d: ${prev_7d:.4f} &nbsp;·&nbsp; Delta: {delta_str}</p>

<hr style="border:1px solid #e2e8f0">
<p style="font-size:.8rem;color:#94a3b8">
  Sent by <a href="https://github.com/ASREDDY03/argus-cortex">Argus Cortex</a>.
</p>
</body></html>
"""

    text_body = f"""Argus Cortex — Weekly Digest ({now})
=====================================

Runs: {runs.get('total_runs', 0)} total, {runs.get('completed_runs', 0)} completed, ${runs.get('total_cost', 0):.4f} cost

Findings: {findings.get('total', 0)} total, {findings.get('total_approved', 0)} approved, {findings.get('total_prs_opened', 0)} PRs
{sev_rows_text}

PR Outcomes:
  Merged: {pr_outcomes.get('merged', 0)}
  Open:   {pr_outcomes.get('open', 0)}
  Closed: {pr_outcomes.get('closed', 0)}

Cost Trend:
  Last 7d:  ${last_7d:.4f}
  Prior 7d: ${prev_7d:.4f}
  Delta:    {delta_str}
"""

    return send_digest(subject, html_body, text_body)
