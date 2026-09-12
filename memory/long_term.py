"""
Long-term memory — SQLite backed.

Two tables:
  runs     — every cortex run (goal, timestamp, thread_id)
  findings — every finding ever produced, with PR status

Agents query this before analyzing so they:
  1. Don't re-report already-known issues
  2. Get context on what was fixed vs still open
"""
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
    """Create tables if they don't exist."""
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id          TEXT PRIMARY KEY,
                thread_id   TEXT NOT NULL,
                goal        TEXT NOT NULL,
                agents_run  TEXT,
                status      TEXT DEFAULT 'running',
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
                finished_at DATETIME
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
                created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );
        """)


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


def finish_run(run_id: str, agents_run: list[str]):
    with _conn() as conn:
        conn.execute(
            "UPDATE runs SET status='completed', agents_run=?, finished_at=? WHERE id=?",
            (",".join(agents_run), datetime.utcnow().isoformat(), run_id),
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
    """
    init_db()
    if not file_paths:
        return []

    placeholders = ",".join("?" * len(file_paths))
    with _conn() as conn:
        rows = conn.execute(
            f"""SELECT agent, file, line, severity, category, description, pr_url
                FROM findings
                WHERE file IN ({placeholders})
                ORDER BY created_at DESC
                LIMIT 50""",
            file_paths,
        ).fetchall()
    return [dict(r) for r in rows]


def get_run_history(limit: int = 10) -> list[dict]:
    """Return recent runs for display."""
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
