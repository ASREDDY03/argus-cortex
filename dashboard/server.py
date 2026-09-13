"""
Argus Cortex — Web Dashboard

Human review UI for approving/rejecting agent findings and triggering PRs.

Routes:
  GET  /                          — home: pending reviews + run history
  GET  /review/<run_id>           — finding cards with approve/reject checkboxes
  POST /review/<run_id>/submit    — submit decisions, resume LangGraph in background
  GET  /review/<run_id>/result    — PR links (auto-refreshes while processing)

Start:
  python dashboard/server.py

Then run Cortex with --web flag:
  python main.py run "full audit" --web
  → prints: Review at http://localhost:8001/review/<run_id>
"""
import json
import logging
import threading
import uvicorn
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from config.settings import settings
from memory.long_term import (
    get_pending_review, list_pending_reviews, get_run_history,
    update_pending_review_status, complete_pending_review,
    mark_approved, mark_rejected, mark_pr_opened, save_findings,
)
from orchestrator.graph import build_graph
from tools.slack_tool import notify_run_complete

logger = logging.getLogger(__name__)

app = FastAPI(title="Argus Cortex Dashboard", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


# ── Template helpers ──────────────────────────────────────────────────────────

def _ago(dt_str: str | None) -> str:
    """Human-readable relative time from ISO datetime string."""
    if not dt_str:
        return "—"
    try:
        dt = datetime.fromisoformat(dt_str)
        diff = int((datetime.utcnow() - dt).total_seconds())
        if diff < 60:
            return f"{diff}s ago"
        if diff < 3600:
            return f"{diff // 60}m ago"
        if diff < 86400:
            return f"{diff // 3600}h ago"
        return f"{diff // 86400}d ago"
    except Exception:
        return dt_str[:16]


app.state.ago = _ago


# ── Background graph resume ───────────────────────────────────────────────────

def _resume_graph(run_id: str, thread_id: str, human_approved_indices: list[int], human_notes: str, goal: str, agents_run: list[str], cost_usd: float):
    """Resume the LangGraph checkpoint in a background thread after web review."""
    try:
        update_pending_review_status(run_id, "processing")
        graph = build_graph()
        config = {"configurable": {"thread_id": thread_id}}
        resume_state = {
            "human_approved_indices": human_approved_indices,
            "human_notes": human_notes,
        }
        final_state = graph.invoke(resume_state, config=config)
        pr_urls = final_state.get("pr_urls", [])

        complete_pending_review(run_id, pr_urls)

        # Mark PRs in long-term memory
        for url in pr_urls:
            mark_pr_opened(run_id, "web_review", url)

        # Slack notification
        review = get_pending_review(run_id)
        approved_findings = json.loads(review.get("approved_findings", "[]")) if review else []
        selected = [approved_findings[i] for i in human_approved_indices if i < len(approved_findings)]
        ci_line = next(
            (l for l in final_state.get("summary", "").splitlines() if l.startswith("CI:")), ""
        )
        notify_run_complete(
            goal=goal,
            run_id=run_id,
            approved_findings=selected,
            agents_used=agents_run,
            pr_urls=pr_urls,
            ci_summary=ci_line.removeprefix("CI:").strip(),
            cost_usd=cost_usd,
        )

    except Exception as e:
        logger.error(f"[dashboard] Graph resume failed for {run_id}: {e}", exc_info=True)
        update_pending_review_status(run_id, "failed")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    all_reviews = list_pending_reviews()
    for r in all_reviews:
        try:
            r["finding_count"] = len(json.loads(r.get("approved_findings") or "[]"))
        except Exception:
            r["finding_count"] = 0
    pending = [r for r in all_reviews if r["status"] == "pending"]
    in_progress = [r for r in all_reviews if r["status"] == "processing"]
    history = get_run_history(limit=15)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "pending": pending,
        "in_progress": in_progress,
        "history": history,
        "ago": _ago,
    })


@app.get("/review/{run_id}", response_class=HTMLResponse)
async def review_page(request: Request, run_id: str):
    review = get_pending_review(run_id)
    if not review:
        return RedirectResponse("/")
    if review["status"] not in ("pending",):
        return RedirectResponse(f"/review/{run_id}/result")

    findings = json.loads(review["approved_findings"])
    # Sort by severity for display
    findings_sorted = sorted(findings, key=lambda f: _SEVERITY_ORDER.get(f.get("severity", "low"), 9))

    return templates.TemplateResponse("review.html", {
        "request": request,
        "review": review,
        "findings": findings_sorted,
        "ago": _ago,
    })


@app.post("/review/{run_id}/submit")
async def submit_review(
    request: Request,
    run_id: str,
    human_notes: str = Form(default=""),
):
    review = get_pending_review(run_id)
    if not review or review["status"] != "pending":
        return RedirectResponse(f"/review/{run_id}/result", status_code=303)

    form = await request.form()
    findings = json.loads(review["approved_findings"])

    # Checkboxes: "approve_0", "approve_1", ... are present when checked
    human_approved_indices = [
        i for i in range(len(findings))
        if form.get(f"approve_{i}") == "on"
    ]

    if not human_approved_indices:
        # Nothing selected — mark cancelled
        update_pending_review_status(run_id, "failed")
        return RedirectResponse(f"/review/{run_id}/result", status_code=303)

    agents_run = [a.strip() for a in (review.get("agents_run") or "").split(",") if a.strip()]
    cost_usd = float(review.get("cost_usd") or 0.0)

    # Save approved/rejected to long-term memory
    selected = [findings[i] for i in human_approved_indices if i < len(findings)]
    rejected = [findings[i] for i in range(len(findings)) if i not in human_approved_indices]
    save_findings(run_id, findings)
    mark_approved(run_id, selected)
    mark_rejected(run_id, rejected)

    # Resume graph in background — web response returns immediately
    t = threading.Thread(
        target=_resume_graph,
        args=(run_id, review["thread_id"], human_approved_indices, human_notes,
              review["goal"], agents_run, cost_usd),
        daemon=True,
    )
    t.start()

    return RedirectResponse(f"/review/{run_id}/result", status_code=303)


@app.get("/review/{run_id}/result", response_class=HTMLResponse)
async def result_page(request: Request, run_id: str):
    review = get_pending_review(run_id)
    if not review:
        return RedirectResponse("/")

    pr_urls = json.loads(review.get("pr_urls") or "[]")
    status = review["status"]
    processing = status == "processing"

    return templates.TemplateResponse("result.html", {
        "request": request,
        "review": review,
        "pr_urls": pr_urls,
        "processing": processing,
        "failed": status == "failed",
        "ago": _ago,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = settings.dashboard_port
    print(f"""
╔══════════════════════════════════════════════════════╗
║         Argus Cortex — Web Dashboard                 ║
╠══════════════════════════════════════════════════════╣
║  Open in browser:  http://localhost:{port}              ║
╠══════════════════════════════════════════════════════╣
║  To send findings here instead of CLI review:        ║
║    python main.py run "goal" --web                   ║
╚══════════════════════════════════════════════════════╝
""")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
