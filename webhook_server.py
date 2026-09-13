"""
Argus Cortex — GitHub Webhook Server

Listens for push events from the Argus Agent repo and auto-triggers a Cortex run.

Flow:
  Argus Agent push → GitHub → ngrok tunnel → POST /webhook
      → verify HMAC signature
      → filter to default branch only
      → build goal from changed files
      → spawn: python main.py run "<goal>" --auto  (background subprocess)
      → return 200 immediately

Setup:
  1. Install ngrok:        brew install ngrok
  2. Start tunnel:         ngrok http 8000
  3. Copy the ngrok URL (e.g. https://abc123.ngrok.io)
  4. Go to Argus Agent repo → Settings → Webhooks → Add webhook
       Payload URL:    https://abc123.ngrok.io/webhook
       Content type:   application/json
       Secret:         <any string — must match WEBHOOK_SECRET in .env>
       Events:         Just the push event
  5. Start this server:    python webhook_server.py

Runs are logged to:  logs/webhook_<timestamp>.log
"""
import hashlib
import hmac
import logging
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from config.settings import settings
from memory.long_term import get_run_history, init_db
from tools.slack_tool import notify_run_start

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Scheduler ─────────────────────────────────────────────────────────────────

_scheduler = BackgroundScheduler(timezone="UTC")


def _run_scheduled():
    """Fired by APScheduler — spawns a Cortex run with the configured goal."""
    goal = settings.schedule_goal
    logger.info(f"[scheduler] Firing scheduled run: {goal}")
    _spawn_run(goal, pusher="scheduler")
    notify_run_start(pusher="scheduler", goal=goal)


def _start_scheduler():
    if not settings.schedule_cron:
        logger.info("[scheduler] No SCHEDULE_CRON set — scheduled runs disabled.")
        return
    try:
        trigger = CronTrigger.from_crontab(settings.schedule_cron, timezone="UTC")
        _scheduler.add_job(_run_scheduled, trigger, id="cortex_scheduled_run", replace_existing=True)
        _scheduler.start()
        next_run = _scheduler.get_job("cortex_scheduled_run").next_run_time
        logger.info(f"[scheduler] Active — cron: '{settings.schedule_cron}' | next run: {next_run}")
    except Exception as e:
        logger.error(f"[scheduler] Failed to start — check SCHEDULE_CRON format: {e}")


def _stop_scheduler():
    if _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("[scheduler] Stopped.")


@asynccontextmanager
async def lifespan(app):
    _start_scheduler()
    yield
    _stop_scheduler()


app = FastAPI(title="Argus Cortex Webhook", docs_url=None, redoc_url=None, lifespan=lifespan)


# ── Signature verification ────────────────────────────────────────────────────

def _verify_signature(payload: bytes, signature: str | None) -> bool:
    """
    Verify GitHub's HMAC-SHA256 webhook signature.
    Header format: 'sha256=<hex_digest>'
    Returns True if signature is valid or WEBHOOK_SECRET is not configured.
    """
    if not settings.webhook_secret:
        logger.warning("WEBHOOK_SECRET not set — accepting all requests (insecure)")
        return True
    if not signature:
        return False
    expected = "sha256=" + hmac.new(
        settings.webhook_secret.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


# ── Goal builder ─────────────────────────────────────────────────────────────

def _build_goal(payload: dict) -> str:
    """
    Build a specific Cortex goal from the push event payload.
    The Planner uses this to focus on what actually changed.
    """
    pusher = payload.get("pusher", {}).get("name", "unknown")
    ref = payload.get("ref", "").replace("refs/heads/", "")
    commits = payload.get("commits", [])

    # Collect all changed files across every commit in this push
    changed: list[str] = []
    for commit in commits:
        changed.extend(commit.get("modified", []))
        changed.extend(commit.get("added", []))
        changed.extend(commit.get("removed", []))

    # De-duplicate while preserving order
    seen: set[str] = set()
    unique_changed = [f for f in changed if not (f in seen or seen.add(f))]  # type: ignore[func-returns-value]

    commit_count = len(commits)
    commit_word = "commit" if commit_count == 1 else "commits"

    if unique_changed:
        file_list = ", ".join(Path(f).name for f in unique_changed[:4])
        suffix = f" (+{len(unique_changed) - 4} more)" if len(unique_changed) > 4 else ""
        return (
            f"Review {commit_count} {commit_word} pushed by {pusher} to {ref}. "
            f"Changed files: {file_list}{suffix}. "
            f"Focus on the modified code for bugs, security issues, and regressions."
        )

    return (
        f"Review {commit_count} {commit_word} pushed by {pusher} to {ref}. "
        f"Full audit for bugs, security issues, and regressions."
    )


# ── Background run ────────────────────────────────────────────────────────────

def _spawn_run(goal: str, pusher: str):
    """
    Spawn a Cortex run as a background subprocess so the webhook returns immediately.
    Output is written to logs/webhook_<timestamp>.log
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = Path("logs") / f"webhook_{timestamp}.log"

    cmd = [sys.executable, "main.py", "run", goal, "--auto"]
    logger.info(f"Spawning run for push by {pusher} → log: {log_path}")

    with open(log_path, "w") as log_file:
        log_file.write(f"Push by: {pusher}\nGoal: {goal}\nStarted: {timestamp}\n\n")
        log_file.flush()
        subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            cwd=Path(__file__).parent,
        )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "argus-cortex-webhook"}


@app.post("/run")
async def manual_run(
    request: Request,
    authorization: str | None = Header(None),
):
    """
    Manually trigger a Cortex run from anywhere — lab, phone, curl.

    Auth: Bearer token matching WEBHOOK_SECRET in .env
      curl -X POST https://<your-ngrok>/run \\
           -H "Authorization: Bearer <WEBHOOK_SECRET>" \\
           -H "Content-Type: application/json" \\
           -d '{"goal": "security audit"}'

    If WEBHOOK_SECRET is not set, the endpoint is open (development only).
    """
    if settings.webhook_secret:
        expected = f"Bearer {settings.webhook_secret}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Unauthorized — provide Bearer <WEBHOOK_SECRET>")

    body = {}
    try:
        body = await request.json()
    except Exception:
        pass

    goal = body.get("goal") or "Full audit of Argus Agent — security, CI/CD, and code quality"
    pusher = body.get("pusher", "manual")

    _spawn_run(goal, pusher)
    notify_run_start(pusher=pusher, goal=goal)

    logger.info(f"Manual run triggered by {pusher}: {goal}")
    return JSONResponse({"triggered": True, "goal": goal, "pusher": pusher})


@app.get("/runs")
def recent_runs():
    """Show the last 10 Cortex runs from long-term memory."""
    init_db()
    return {"runs": get_run_history(limit=10)}


@app.get("/schedule")
def schedule_status():
    """Show current scheduler state and next run time."""
    if not settings.schedule_cron:
        return {"enabled": False, "reason": "SCHEDULE_CRON not set in .env"}
    if not _scheduler.running:
        return {"enabled": False, "reason": "Scheduler failed to start — check logs"}
    job = _scheduler.get_job("cortex_scheduled_run")
    next_run = str(job.next_run_time) if job and job.next_run_time else "unknown"
    return {
        "enabled": True,
        "cron": settings.schedule_cron,
        "goal": settings.schedule_goal,
        "next_run_utc": next_run,
    }


@app.post("/webhook", status_code=status.HTTP_200_OK)
async def github_webhook(
    request: Request,
    x_github_event: str | None = Header(None),
    x_hub_signature_256: str | None = Header(None),
):
    payload_bytes = await request.body()

    # 1. Verify signature
    if not _verify_signature(payload_bytes, x_hub_signature_256):
        logger.warning("Webhook rejected — invalid signature")
        raise HTTPException(status_code=403, detail="Invalid signature")

    # 2. Only handle push events
    if x_github_event != "push":
        logger.info(f"Ignoring event: {x_github_event}")
        return {"ignored": True, "event": x_github_event}

    payload = await request.json()

    # 3. Only trigger on the default branch
    ref = payload.get("ref", "")
    default_branch = f"refs/heads/{payload.get('repository', {}).get('default_branch', 'main')}"
    if ref != default_branch:
        logger.info(f"Ignoring push to non-default branch: {ref}")
        return {"ignored": True, "reason": f"push to {ref}, not {default_branch}"}

    # 4. Build goal and spawn run
    pusher = payload.get("pusher", {}).get("name", "unknown")
    goal = _build_goal(payload)

    _spawn_run(goal, pusher)
    notify_run_start(pusher=pusher, goal=goal)

    logger.info(f"Run triggered for push by {pusher}")
    return JSONResponse({"triggered": True, "goal": goal, "pusher": pusher})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = settings.webhook_port
    schedule_line = (
        f"  Schedule:    {settings.schedule_cron} (UTC)"
        if settings.schedule_cron
        else "  Schedule:    disabled (set SCHEDULE_CRON in .env)"
    )

    print(f"""
╔══════════════════════════════════════════════════════╗
║           Argus Cortex — Webhook Server              ║
╠══════════════════════════════════════════════════════╣
║  Listening on:  http://localhost:{port}                 ║
║  Health check:  http://localhost:{port}/health          ║
║  Recent runs:   http://localhost:{port}/runs            ║
║  Schedule:      http://localhost:{port}/schedule        ║
╠══════════════════════════════════════════════════════╣
║  Scheduled runs ({schedule_line.strip()})
╠══════════════════════════════════════════════════════╣
║  Webhook setup:                                      ║
║  1. ngrok http {port}                                   ║
║  2. Copy the https://xxxx.ngrok.io URL               ║
║  3. Argus Agent repo → Settings → Webhooks           ║
║     Payload URL:  https://xxxx.ngrok.io/webhook      ║
║     Content type: application/json                   ║
║     Secret:       <your WEBHOOK_SECRET from .env>    ║
║     Events:       Just the push event                ║
╚══════════════════════════════════════════════════════╝
""")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
