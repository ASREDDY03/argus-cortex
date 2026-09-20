"""
Long-term memory — SQLite backed.

Two tables:
  runs     — every cortex run (goal, timestamp, thread_id)
  findings — every finding ever produced, with PR status

Agents query this before analyzing so they:
  1. Don't re-report already-known issues
  2. Get context on what was fixed vs still open
"""
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = Path("checkpoints/memory.db")


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist. Migrates existing DBs forward."""
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id           TEXT PRIMARY KEY,
                thread_id    TEXT NOT NULL,
                goal         TEXT NOT NULL,
                agents_run   TEXT,
                status       TEXT DEFAULT 'running',
                tokens_in    INTEGER DEFAULT 0,
                tokens_out   INTEGER DEFAULT 0,
                cost_usd     REAL DEFAULT 0.0,
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
                finished_at  DATETIME
            );

            CREATE TABLE IF NOT EXISTS pending_reviews (
                run_id            TEXT PRIMARY KEY,
                thread_id         TEXT NOT NULL,
                goal              TEXT NOT NULL,
                approved_findings TEXT NOT NULL,
                agents_run        TEXT,
                cost_usd          REAL DEFAULT 0.0,
                status            TEXT DEFAULT 'pending',
                pr_urls           TEXT,
                created_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at      DATETIME
            );

            CREATE TABLE IF NOT EXISTS findings (
                id              TEXT PRIMARY KEY,
                run_id          TEXT NOT NULL,
                agent           TEXT NOT NULL,
                file            TEXT NOT NULL,
                line            INTEGER,
                severity        TEXT,
                category        TEXT,
                description     TEXT,
                suggested_fix   TEXT,
                pr_ready        INTEGER DEFAULT 0,
                approved        INTEGER DEFAULT 0,
                rejected        INTEGER DEFAULT 0,
                pr_url          TEXT,
                pr_state        TEXT,
                created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
                finding_id      TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );

            CREATE TABLE IF NOT EXISTS pr_outcomes (
                pr_url          TEXT PRIMARY KEY,
                pr_number       INTEGER,
                state           TEXT NOT NULL,
                title           TEXT,
                close_reason    TEXT,
                review_comments TEXT,
                merged_by       TEXT,
                synced_at       DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS file_reviews (
                file_path    TEXT PRIMARY KEY,
                content_sha  TEXT NOT NULL,
                reviewed_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
                run_id       TEXT
            );
        """)
        # Migrations: add columns to existing DBs that pre-date them
        for migration in [
            "ALTER TABLE findings ADD COLUMN pr_state TEXT",
            "ALTER TABLE runs ADD COLUMN tokens_in INTEGER DEFAULT 0",
            "ALTER TABLE runs ADD COLUMN tokens_out INTEGER DEFAULT 0",
            "ALTER TABLE runs ADD COLUMN cost_usd REAL DEFAULT 0.0",
            "ALTER TABLE findings ADD COLUMN finding_id TEXT",
        ]:
            try:
                conn.execute(migration)
            except Exception:
                pass  # column already exists


def start_run(thread_id: str, goal: str) -> str:
    """Record a new run. Returns run_id."""
    init_db()
    run_id = str(uuid.uuid4())
    with _conn() as conn:
        conn.execute(
            "INSERT INTO runs (id, thread_id, goal) VALUES (?, ?, ?)",
            (run_id, thread_id, goal),
        )
    return run_id


def finish_run(run_id: str, agents_run: list[str], tokens_in: int = 0, tokens_out: int = 0, cost_usd: float = 0.0):
    with _conn() as conn:
        conn.execute(
            """UPDATE runs
               SET status='completed', agents_run=?, finished_at=?,
                   tokens_in=?, tokens_out=?, cost_usd=?
               WHERE id=?""",
            (",".join(agents_run), datetime.utcnow().isoformat(), tokens_in, tokens_out, cost_usd, run_id),
        )


def save_findings(run_id: str, findings: list[dict]):
    """Persist findings for a run."""
    init_db()
    with _conn() as conn:
        for f in findings:
            conn.execute(
                """INSERT INTO findings
                   (id, run_id, agent, file, line, severity, category,
                    description, suggested_fix, pr_ready, finding_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    run_id,
                    f.get("agent", ""),
                    f.get("file", ""),
                    f.get("line"),
                    f.get("severity", ""),
                    f.get("category", ""),
                    f.get("description", ""),
                    f.get("suggested_fix", ""),
                    1 if f.get("pr_ready") else 0,
                    f.get("finding_id", ""),
                ),
            )


def mark_approved(run_id: str, findings: list[dict]):
    with _conn() as conn:
        for f in findings:
            conn.execute(
                """UPDATE findings SET approved=1
                   WHERE run_id=? AND agent=? AND file=? AND description=?""",
                (run_id, f.get("agent"), f.get("file"), f.get("description")),
            )


def mark_rejected(run_id: str, findings: list[dict]):
    with _conn() as conn:
        for f in findings:
            conn.execute(
                """UPDATE findings SET rejected=1
                   WHERE run_id=? AND agent=? AND file=? AND description=?""",
                (run_id, f.get("agent"), f.get("file"), f.get("description")),
            )


def mark_pr_opened(run_id: str, agent: str, pr_url: str):
    with _conn() as conn:
        conn.execute(
            "UPDATE findings SET pr_url=? WHERE run_id=? AND agent=?",
            (pr_url, run_id, agent),
        )


def upsert_pr_outcome(
    pr_url: str,
    pr_number: int,
    state: str,
    title: str = "",
    close_reason: str = "",
    review_comments: str = "",
    merged_by: str = "",
):
    """
    Insert or update a PR outcome record.
    Called by sync_pr_states after fetching review comments from GitHub.
    state: 'open' | 'merged' | 'closed'
    """
    with _conn() as conn:
        conn.execute(
            """INSERT INTO pr_outcomes
               (pr_url, pr_number, state, title, close_reason, review_comments, merged_by, synced_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(pr_url) DO UPDATE SET
                 state=excluded.state,
                 title=excluded.title,
                 close_reason=excluded.close_reason,
                 review_comments=excluded.review_comments,
                 merged_by=excluded.merged_by,
                 synced_at=excluded.synced_at""",
            (pr_url, pr_number, state, title, close_reason, review_comments, merged_by,
             datetime.utcnow().isoformat()),
        )


def get_past_findings_for_files(file_paths: list[str]) -> list[dict]:
    """
    Query past findings for specific files.
    Agents use this to avoid re-reporting already known issues.
    Joins pr_outcomes so agents see WHY a PR was closed, not just that it was.
    """
    init_db()
    if not file_paths:
        return []

    placeholders = ",".join("?" * len(file_paths))
    with _conn() as conn:
        rows = conn.execute(
            f"""SELECT f.agent, f.file, f.line, f.severity, f.category,
                       f.description, f.pr_url, f.pr_state,
                       po.close_reason
                FROM findings f
                LEFT JOIN pr_outcomes po ON f.pr_url = po.pr_url
                WHERE f.file IN ({placeholders})
                ORDER BY f.created_at DESC
                LIMIT 50""",
            file_paths,
        ).fetchall()
    return [dict(r) for r in rows]


def get_findings_with_open_prs() -> list[dict]:
    """
    Return all findings that have a PR URL but whose outcome is not yet final.
    Used by sync_pr_states() to know which PRs to check on GitHub.
    """
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            """SELECT DISTINCT pr_url
               FROM findings
               WHERE pr_url IS NOT NULL
                 AND (pr_state IS NULL OR pr_state = 'open')"""
        ).fetchall()
    return [dict(r) for r in rows]


def update_pr_state_by_url(pr_url: str, state: str):
    """
    Set pr_state for every finding linked to this PR URL.
    state: 'open' | 'merged' | 'closed'
    """
    with _conn() as conn:
        conn.execute(
            "UPDATE findings SET pr_state=? WHERE pr_url=?",
            (state, pr_url),
        )


REVIEW_TTL_DAYS = 7  # force re-review even for unchanged files after this many days


def get_stale_files(file_shas: dict[str, str]) -> list[str]:
    """
    Return file paths that need re-review: SHA changed, never reviewed, or
    reviewed more than REVIEW_TTL_DAYS ago (catches stale-but-unchanged files).
    file_shas: {relative_file_path: sha256_hex}
    """
    init_db()
    stale = []
    with _conn() as conn:
        for file_path, current_sha in file_shas.items():
            row = conn.execute(
                "SELECT content_sha, reviewed_at FROM file_reviews WHERE file_path = ?",
                (file_path,)
            ).fetchone()
            if row is None:
                stale.append(file_path)
                continue
            if row["content_sha"] != current_sha:
                stale.append(file_path)
                continue
            # Force re-review if last review is older than TTL
            try:
                last = datetime.fromisoformat(row["reviewed_at"])
                age_days = (datetime.utcnow() - last).days
                if age_days >= REVIEW_TTL_DAYS:
                    stale.append(file_path)
            except Exception:
                stale.append(file_path)
    return stale


def update_file_shas(file_shas: dict[str, str], run_id: str = ""):
    """
    Record current SHA for each reviewed file.
    Called after agent analysis so next run can skip unchanged files.
    """
    now = datetime.utcnow().isoformat()
    with _conn() as conn:
        for file_path, sha in file_shas.items():
            conn.execute(
                """INSERT INTO file_reviews (file_path, content_sha, reviewed_at, run_id)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(file_path) DO UPDATE SET
                     content_sha = excluded.content_sha,
                     reviewed_at = excluded.reviewed_at,
                     run_id      = excluded.run_id""",
                (file_path, sha, now, run_id),
            )


def get_stats() -> dict:
    """
    Aggregate statistics across all runs and findings.
    Used by `python main.py stats`.
    """
    init_db()
    with _conn() as conn:

        # ── Runs ──────────────────────────────────────────────────────────────
        runs_row = conn.execute("""
            SELECT
                COUNT(*)                                                      AS total_runs,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END)        AS completed_runs,
                COALESCE(SUM(cost_usd), 0)                                   AS total_cost,
                COALESCE(AVG(CASE WHEN status = 'completed' AND cost_usd > 0
                                  THEN cost_usd END), 0)                     AS avg_cost,
                COALESCE(SUM(tokens_in),  0)                                 AS total_tokens_in,
                COALESCE(SUM(tokens_out), 0)                                 AS total_tokens_out
            FROM runs
        """).fetchone()

        # ── Findings overall ─────────────────────────────────────────────────
        findings_row = conn.execute("""
            SELECT
                COUNT(*)                                                      AS total,
                COALESCE(SUM(approved), 0)                                   AS total_approved,
                COALESCE(SUM(rejected), 0)                                   AS total_rejected,
                COALESCE(SUM(pr_ready), 0)                                   AS total_pr_ready,
                SUM(CASE WHEN pr_url IS NOT NULL THEN 1 ELSE 0 END)          AS total_prs_opened
            FROM findings
        """).fetchone()

        # ── By severity ───────────────────────────────────────────────────────
        by_severity = conn.execute("""
            SELECT severity,
                   COUNT(*)           AS total,
                   SUM(approved)      AS approved,
                   SUM(rejected)      AS rejected
            FROM findings
            WHERE severity IS NOT NULL AND severity != ''
            GROUP BY severity
            ORDER BY CASE severity
                WHEN 'critical' THEN 0 WHEN 'high' THEN 1
                WHEN 'medium'   THEN 2 ELSE 3 END
        """).fetchall()

        # ── By agent ─────────────────────────────────────────────────────────
        by_agent = conn.execute("""
            SELECT agent,
                   COUNT(*)           AS total,
                   SUM(approved)      AS approved,
                   SUM(rejected)      AS rejected
            FROM findings
            WHERE agent IS NOT NULL AND agent != ''
            GROUP BY agent
            ORDER BY total DESC
        """).fetchall()

        # ── By category ──────────────────────────────────────────────────────
        by_category = conn.execute("""
            SELECT category,
                   COUNT(*)           AS total,
                   SUM(approved)      AS approved
            FROM findings
            WHERE category IS NOT NULL AND category != ''
            GROUP BY category
            ORDER BY total DESC
        """).fetchall()

        # ── PR outcomes ───────────────────────────────────────────────────────
        pr_rows = conn.execute("""
            SELECT COALESCE(pr_state, 'open') AS state, COUNT(*) AS count
            FROM findings
            WHERE pr_url IS NOT NULL
            GROUP BY pr_state
        """).fetchall()

        # ── Top flagged files ─────────────────────────────────────────────────
        top_files = conn.execute("""
            SELECT file,
                   COUNT(*)           AS total,
                   SUM(approved)      AS approved
            FROM findings
            WHERE file IS NOT NULL AND file != ''
            GROUP BY file
            ORDER BY total DESC
            LIMIT 8
        """).fetchall()

        # ── Cost: last 7 days vs prior 7 days ────────────────────────────────
        trend_row = conn.execute("""
            SELECT
                COALESCE(SUM(CASE WHEN created_at >= date('now','-7 days')
                                  THEN cost_usd END), 0)  AS last_7d,
                COALESCE(SUM(CASE WHEN created_at >= date('now','-14 days')
                                   AND created_at <  date('now','-7 days')
                                  THEN cost_usd END), 0)  AS prev_7d,
                COUNT(CASE WHEN created_at >= date('now','-7 days') THEN 1 END) AS runs_last_7d
            FROM runs
            WHERE status = 'completed'
        """).fetchone()

    pr_outcomes: dict[str, int] = {}
    for row in pr_rows:
        pr_outcomes[row["state"]] = row["count"]

    return {
        "runs":        dict(runs_row),
        "findings":    dict(findings_row),
        "by_severity": [dict(r) for r in by_severity],
        "by_agent":    [dict(r) for r in by_agent],
        "by_category": [dict(r) for r in by_category],
        "pr_outcomes": pr_outcomes,
        "top_files":   [dict(r) for r in top_files],
        "trend":       dict(trend_row),
    }


def get_outcome_summary() -> dict:
    """
    Aggregate PR outcome stats across all runs.
    Used by goal_suggester to inform suggestions, and by the outcome-report command.
    """
    init_db()
    with _conn() as conn:
        by_category = conn.execute("""
            SELECT f.category,
                   COUNT(DISTINCT f.pr_url)                                    AS total_prs,
                   SUM(CASE WHEN po.state = 'merged' THEN 1 ELSE 0 END)       AS merged,
                   SUM(CASE WHEN po.state = 'closed' THEN 1 ELSE 0 END)       AS closed
            FROM findings f
            JOIN pr_outcomes po ON f.pr_url = po.pr_url
            WHERE f.pr_url IS NOT NULL AND f.category IS NOT NULL AND f.category != ''
            GROUP BY f.category
            ORDER BY total_prs DESC
        """).fetchall()

        by_agent = conn.execute("""
            SELECT f.agent,
                   COUNT(DISTINCT f.pr_url)                                    AS total_prs,
                   SUM(CASE WHEN po.state = 'merged' THEN 1 ELSE 0 END)       AS merged,
                   SUM(CASE WHEN po.state = 'closed' THEN 1 ELSE 0 END)       AS closed
            FROM findings f
            JOIN pr_outcomes po ON f.pr_url = po.pr_url
            WHERE f.pr_url IS NOT NULL AND f.agent IS NOT NULL AND f.agent != ''
            GROUP BY f.agent
            ORDER BY total_prs DESC
        """).fetchall()

        close_reasons = conn.execute("""
            SELECT pr_url, title, close_reason, synced_at
            FROM pr_outcomes
            WHERE state = 'closed' AND close_reason IS NOT NULL AND close_reason != ''
            ORDER BY synced_at DESC
            LIMIT 10
        """).fetchall()

        rejected_files = conn.execute("""
            SELECT f.file,
                   COUNT(*) AS closed_count
            FROM findings f
            JOIN pr_outcomes po ON f.pr_url = po.pr_url
            WHERE po.state = 'closed'
            GROUP BY f.file
            ORDER BY closed_count DESC
            LIMIT 8
        """).fetchall()

        totals = conn.execute("""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN state = 'merged' THEN 1 ELSE 0 END) AS merged,
                   SUM(CASE WHEN state = 'closed' THEN 1 ELSE 0 END) AS closed,
                   SUM(CASE WHEN state = 'open'   THEN 1 ELSE 0 END) AS open
            FROM pr_outcomes
        """).fetchone()

    return {
        "totals":         dict(totals) if totals else {},
        "by_category":    [dict(r) for r in by_category],
        "by_agent":       [dict(r) for r in by_agent],
        "close_reasons":  [dict(r) for r in close_reasons],
        "rejected_files": [dict(r) for r in rejected_files],
    }


def get_run_history(limit: int = 10) -> list[dict]:
    """Return recent runs for display."""
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ── Dashboard: pending_reviews CRUD ──────────────────────────────────────────

def save_pending_review(
    run_id: str,
    thread_id: str,
    goal: str,
    approved_findings: list[dict],
    agents_run: list[str],
    cost_usd: float = 0.0,
):
    """Save an evaluator-approved set of findings awaiting human review in the dashboard."""
    init_db()
    with _conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO pending_reviews
               (run_id, thread_id, goal, approved_findings, agents_run, cost_usd)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (run_id, thread_id, goal, json.dumps(approved_findings),
             ",".join(agents_run), cost_usd),
        )


def get_pending_review(run_id: str) -> dict | None:
    """Return a single pending review by run_id, or None if not found."""
    init_db()
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM pending_reviews WHERE run_id = ?", (run_id,)
        ).fetchone()
    return dict(row) if row else None


def list_pending_reviews() -> list[dict]:
    """Return all reviews ordered by most recent, for the dashboard home page."""
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM pending_reviews ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
    return [dict(r) for r in rows]


def update_pending_review_status(run_id: str, status: str):
    """Update status: 'pending' | 'processing' | 'completed' | 'failed'."""
    with _conn() as conn:
        conn.execute(
            "UPDATE pending_reviews SET status = ? WHERE run_id = ?",
            (status, run_id),
        )


def complete_pending_review(run_id: str, pr_urls: list[str]):
    """Mark a review as completed and store the resulting PR URLs."""
    with _conn() as conn:
        conn.execute(
            """UPDATE pending_reviews
               SET status = 'completed', pr_urls = ?, completed_at = ?
               WHERE run_id = ?""",
            (json.dumps(pr_urls), datetime.utcnow().isoformat(), run_id),
        )


def get_run_findings(run_id: str) -> list[dict]:
    """Return all findings for a specific run."""
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM findings WHERE run_id = ? ORDER BY severity DESC",
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_weekly_stats() -> dict:
    """Return aggregated stats for the past 7 days."""
    init_db()
    with _conn() as conn:
        runs = conn.execute(
            "SELECT COUNT(*) FROM runs WHERE created_at >= datetime('now', '-7 days')"
        ).fetchone()[0]
        findings = conn.execute(
            "SELECT COUNT(*) FROM findings WHERE created_at >= datetime('now', '-7 days')"
        ).fetchone()[0]
        approved = conn.execute(
            "SELECT COUNT(*) FROM findings WHERE approved=1 AND created_at >= datetime('now', '-7 days')"
        ).fetchone()[0]
        prs = conn.execute(
            "SELECT COUNT(*) FROM findings WHERE pr_url IS NOT NULL AND created_at >= datetime('now', '-7 days')"
        ).fetchone()[0]
    return {
        "runs": runs,
        "findings": findings,
        "approved": approved,
        "prs_opened": prs,
    }
