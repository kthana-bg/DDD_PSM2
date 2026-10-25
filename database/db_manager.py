import os
import sqlite3
from datetime import datetime
from typing import List, Optional

DB_PATH = os.environ.get("DRIVER_DB_PATH") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "drowsyguard.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def initialize_database():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        detector TEXT,
        sensitivity TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id))""")
    c.execute("""CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        level TEXT,
        score REAL,
        eye_closed INTEGER,
        p_closed REAL,
        ear REAL,
        perclos REAL,
        closure_s REAL,
        mar REAL,
        pitch REAL,
        yawns INTEGER,
        nods INTEGER,
        detector TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(id))""")
    c.execute("""CREATE TABLE IF NOT EXISTS drowsy_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        level TEXT,
        reasons TEXT,
        score REAL,
        perclos REAL,
        FOREIGN KEY (session_id) REFERENCES sessions(id))""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_metrics_session ON metrics(session_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON drowsy_events(session_id)")
    conn.commit()
    conn.close()


def create_user(username: str, password_hash: str, full_name: str) -> int:
    conn = get_connection()
    try:
        cur = conn.execute("INSERT INTO users (username, password_hash, full_name) VALUES (?, ?, ?)",
                           (username, password_hash, full_name))
        conn.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        return -1
    finally:
        conn.close()


def get_user_by_username(username: str) -> Optional[dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def start_session(user_id: int, detector: str, sensitivity: str) -> int:
    conn = get_connection()
    cur = conn.execute("INSERT INTO sessions (user_id, started_at, detector, sensitivity) VALUES (?, ?, ?, ?)",
                       (user_id, _now(), detector, sensitivity))
    conn.commit()
    sid = cur.lastrowid
    conn.close()
    return sid


def end_session(session_id: int):
    conn = get_connection()
    conn.execute("UPDATE sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL", (_now(), session_id))
    conn.commit()
    conn.close()


def get_session(session_id: int) -> Optional[dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_sessions(user_id: int, limit: int = 50) -> List[dict]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT s.*,
               (SELECT COUNT(*) FROM drowsy_events e WHERE e.session_id = s.id AND e.level = 'Warning') AS n_warning,
               (SELECT COUNT(*) FROM drowsy_events e WHERE e.session_id = s.id AND e.level = 'Danger') AS n_danger,
               (SELECT MAX(score) FROM metrics m WHERE m.session_id = s.id) AS peak_score,
               (SELECT COUNT(*) FROM metrics m WHERE m.session_id = s.id) AS n_samples
        FROM sessions s WHERE s.user_id = ? ORDER BY s.id DESC LIMIT ?""", (user_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_session(session_id: int, user_id: int):
    conn = get_connection()
    owner = conn.execute("SELECT user_id FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if owner and owner["user_id"] == user_id:
        conn.execute("DELETE FROM metrics WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM drowsy_events WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
    conn.close()


def save_metric(session_id: int, user_id: int, level: str, score: float, eye_closed: bool, p_closed: Optional[float],
                ear: float, perclos: float, closure_s: float, mar: float, pitch: float, yawns: int, nods: int,
                detector: str, timestamp: Optional[str] = None):
    conn = get_connection()
    conn.execute("""INSERT INTO metrics (session_id, user_id, timestamp, level, score, eye_closed, p_closed, ear,
                    perclos, closure_s, mar, pitch, yawns, nods, detector)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                 (session_id, user_id, timestamp or _now(), level, float(score), int(bool(eye_closed)),
                  None if p_closed is None else float(p_closed), float(ear), float(perclos), float(closure_s),
                  float(mar), float(pitch), int(yawns), int(nods), detector))
    conn.commit()
    conn.close()


def save_event(session_id: int, user_id: int, level: str, reasons: List[str], score: float, perclos: float,
               timestamp: Optional[str] = None):
    conn = get_connection()
    conn.execute("""INSERT INTO drowsy_events (session_id, user_id, timestamp, level, reasons, score, perclos)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                 (session_id, user_id, timestamp or _now(), level, ", ".join(reasons), float(score), float(perclos)))
    conn.commit()
    conn.close()


def get_metrics(session_id: int) -> List[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM metrics WHERE session_id = ? ORDER BY timestamp ASC, id ASC", (session_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_events(session_id: int) -> List[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM drowsy_events WHERE session_id = ? ORDER BY timestamp ASC, id ASC", (session_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


initialize_database()
