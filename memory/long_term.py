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
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );
        """)
        # Migrations: add columns to existing DBs that pre-date them
        for migration in [
            "ALTER TABLE findings ADD COLUMN pr_state TEXT",
            "ALTER TABLE runs ADD COLUMN tokens_in INTEGER DEFAULT 0",
            "ALTER TABLE runs ADD COLUMN tokens_out INTEGER DEFAULT 0",
            "ALTER TABLE runs ADD COLUMN cost_usd REAL DEFAULT 0.0",
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
                    description, suggested_fix, pr_ready)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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


def get_past_findings_for_files(file_paths: list[str]) -> list[dict]:
    """
    Query past findings for specific files.
    Agents use this to avoid re-reporting already known issues.
    Includes pr_state so agents know which issues are fixed vs still open.
    """
    init_db()
    if not file_paths:
        return []

    placeholders = ",".join("?" * len(file_paths))
    with _conn() as conn:
        rows = conn.execute(
            f"""SELECT agent, file, line, severity, category, description, pr_url, pr_state
                FROM findings
                WHERE file IN ({placeholders})
                ORDER BY created_at DESC
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
