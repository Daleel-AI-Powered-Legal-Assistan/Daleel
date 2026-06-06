"""
User management database — SQLite.

Tables:
  users                — accounts with role (user / lawyer / admin)
  lawyer_verifications — pending/approved/rejected lawyer ID submissions
  uploaded_documents   — files uploaded by verified lawyers
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path("data/users.db")

# ── Helpers ──────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """SHA-256 + random salt.  Returns (hash, salt)."""
    if salt is None:
        salt = secrets.token_hex(16)
    h = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return h, salt


def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
    h, _ = _hash_password(password, salt)
    return h == stored_hash


# ── Schema ───────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every app start."""
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            email         TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            password_salt TEXT    NOT NULL,
            full_name     TEXT    NOT NULL,
            role          TEXT    NOT NULL DEFAULT 'user',
            created_at    TEXT    NOT NULL,
            is_active     INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS lawyer_verifications (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id           INTEGER NOT NULL REFERENCES users(id),
            lawyer_id_number  TEXT    NOT NULL UNIQUE,
            full_name_ar      TEXT    NOT NULL,
            id_document_path  TEXT,
            status            TEXT    NOT NULL DEFAULT 'pending',
            submitted_at      TEXT    NOT NULL,
            reviewed_by       INTEGER REFERENCES users(id),
            reviewed_at       TEXT,
            rejection_reason  TEXT
        );

        CREATE TABLE IF NOT EXISTS licensed_lawyers (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            bar_number        TEXT    UNIQUE NOT NULL,
            full_name_ar      TEXT    NOT NULL,
            email             TEXT    UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS uploaded_documents (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id           INTEGER NOT NULL REFERENCES users(id),
            filename          TEXT    NOT NULL,
            original_filename TEXT    NOT NULL,
            file_type         TEXT    NOT NULL,
            file_size_bytes   INTEGER NOT NULL,
            doc_category      TEXT    DEFAULT 'other',
            upload_date       TEXT    NOT NULL,
            processing_status TEXT    NOT NULL DEFAULT 'pending',
            chunk_count       INTEGER DEFAULT 0,
            error_message     TEXT,
            qdrant_collection TEXT    NOT NULL
        );
        """)

    # Seed admin account if none exists
    admin = get_user_by_email("admin@lawbot.jo")
    if admin is None:
        create_user(
            email="admin@lawbot.jo",
            password="admin2024",
            full_name="مدير النظام",
            role="admin",
        )
        print("[auth] Default admin created: admin@lawbot.jo")

    # Seed licensed lawyers registry (demo data for presentation)
    _seed_licensed_lawyers()


def _seed_licensed_lawyers() -> None:
    """Populate the licensed_lawyers table with demo bar-association data."""
    demo_lawyers = [
        ("730201", "عمر امين حسن الهيموني", "omar.example@demo-daleel.jo"),
        ("730202", "محمد رياض محمد المصري", "mohd.example@demo-daleel.jo"),
        ("730203", "حسن حسين عمر العمري", "hasan.example@demo-daleel.jo"),
        ("730204", "عبدالله محمد عبدالله الخوالدة", "abd.example@demo-daleel.jo"),
    ]
    with _conn() as c:
        for bar_num, name, email in demo_lawyers:
            existing = c.execute(
                "SELECT 1 FROM licensed_lawyers WHERE bar_number=?",
                (bar_num,)).fetchone()
            if not existing:
                c.execute(
                    "INSERT INTO licensed_lawyers (bar_number, full_name_ar, email) VALUES (?, ?, ?)",
                    (bar_num, name, email))
        print("[auth] Licensed lawyers registry seeded (4 demo entries)")


# ── User CRUD ────────────────────────────────────────────────────

def create_user(email: str, password: str, full_name: str,
                role: str = "user") -> int:
    pw_hash, salt = _hash_password(password)
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO users (email, password_hash, password_salt,
                                  full_name, role, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (email.lower().strip(), pw_hash, salt, full_name, role, _now()),
        )
        return cur.lastrowid


def authenticate(email: str, password: str) -> dict | None:
    """Return user dict if credentials match, else None."""
    user = get_user_by_email(email)
    if user is None or not user["is_active"]:
        return None
    if _verify_password(password, user["password_hash"], user["password_salt"]):
        return user
    return None


def get_user_by_email(email: str) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE email=?",
                        (email.lower().strip(),)).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE id=?",
                        (user_id,)).fetchone()
        return dict(row) if row else None


def update_user_role(user_id: int, role: str) -> None:
    with _conn() as c:
        c.execute("UPDATE users SET role=? WHERE id=?", (role, user_id))


def email_exists(email: str) -> bool:
    with _conn() as c:
        row = c.execute("SELECT 1 FROM users WHERE email=?",
                        (email.lower().strip(),)).fetchone()
        return row is not None


# ── Lawyer Verification ─────────────────────────────────────────

LAWYER_ID_RE = re.compile(r"^\d{6}$")


def validate_lawyer_id(lid: str) -> bool:
    """Must be exactly 6 digits."""
    return bool(LAWYER_ID_RE.match(lid.strip()))


def verify_bar_registration(bar_number: str, full_name_ar: str) -> str | None:
    """Check bar number against the licensed lawyers registry.

    Returns:
        None              — match found, lawyer is licensed
        error message str — mismatch or not found
    """
    bar_number = bar_number.strip()
    full_name_ar = full_name_ar.strip()
    with _conn() as c:
        row = c.execute(
            "SELECT full_name_ar FROM licensed_lawyers WHERE bar_number=?",
            (bar_number,)).fetchone()
        if row is None:
            return "رقم النقابة غير موجود في سجل المحامين المرخصين"
        if row["full_name_ar"] != full_name_ar:
            return "الاسم لا يتطابق مع رقم النقابة المسجل في سجل النقابة"
    return None


def submit_lawyer_verification(user_id: int, lawyer_id_number: str,
                                full_name_ar: str,
                                id_document_path: str | None = None) -> int:
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO lawyer_verifications
               (user_id, lawyer_id_number, full_name_ar, id_document_path,
                status, submitted_at)
               VALUES (?, ?, ?, ?, 'pending', ?)""",
            (user_id, lawyer_id_number.strip(), full_name_ar,
             id_document_path, _now()),
        )
        return cur.lastrowid


def get_verification_by_user(user_id: int) -> dict | None:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM lawyer_verifications WHERE user_id=? ORDER BY id DESC LIMIT 1",
            (user_id,)).fetchone()
        return dict(row) if row else None


def lawyer_id_taken(lawyer_id_number: str) -> bool:
    with _conn() as c:
        row = c.execute(
            "SELECT 1 FROM lawyer_verifications WHERE lawyer_id_number=?",
            (lawyer_id_number.strip(),)).fetchone()
        return row is not None


def get_pending_verifications() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            """SELECT v.*, u.email, u.full_name
               FROM lawyer_verifications v
               JOIN users u ON v.user_id = u.id
               WHERE v.status = 'pending'
               ORDER BY v.submitted_at ASC""").fetchall()
        return [dict(r) for r in rows]


def get_all_verifications() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            """SELECT v.*, u.email, u.full_name
               FROM lawyer_verifications v
               JOIN users u ON v.user_id = u.id
               ORDER BY v.submitted_at DESC""").fetchall()
        return [dict(r) for r in rows]


def approve_verification(verification_id: int, admin_id: int) -> None:
    with _conn() as c:
        row = c.execute("SELECT user_id FROM lawyer_verifications WHERE id=?",
                        (verification_id,)).fetchone()
        if not row:
            return
        c.execute(
            """UPDATE lawyer_verifications
               SET status='approved', reviewed_by=?, reviewed_at=?
               WHERE id=?""",
            (admin_id, _now(), verification_id),
        )
        c.execute("UPDATE users SET role='lawyer' WHERE id=?", (row["user_id"],))


def reject_verification(verification_id: int, admin_id: int,
                         reason: str = "") -> None:
    with _conn() as c:
        c.execute(
            """UPDATE lawyer_verifications
               SET status='rejected', reviewed_by=?, reviewed_at=?,
                   rejection_reason=?
               WHERE id=?""",
            (admin_id, _now(), reason, verification_id),
        )


# ── Uploaded Documents ───────────────────────────────────────────

def qdrant_collection_for_user(user_id: int) -> str:
    return f"lawyer_{user_id}_docs"


def save_document(user_id: int, filename: str, original_filename: str,
                  file_type: str, file_size_bytes: int,
                  doc_category: str = "other") -> int:
    collection = qdrant_collection_for_user(user_id)
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO uploaded_documents
               (user_id, filename, original_filename, file_type,
                file_size_bytes, doc_category, upload_date,
                processing_status, qdrant_collection)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
            (user_id, filename, original_filename, file_type,
             file_size_bytes, doc_category, _now(), collection),
        )
        return cur.lastrowid


def update_document_status(doc_id: int, status: str,
                            chunk_count: int = 0,
                            error_message: str | None = None) -> None:
    with _conn() as c:
        c.execute(
            """UPDATE uploaded_documents
               SET processing_status=?, chunk_count=?, error_message=?
               WHERE id=?""",
            (status, chunk_count, error_message, doc_id),
        )


def get_user_documents(user_id: int) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            """SELECT * FROM uploaded_documents
               WHERE user_id=? ORDER BY upload_date DESC""",
            (user_id,)).fetchall()
        return [dict(r) for r in rows]


def get_document(doc_id: int) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM uploaded_documents WHERE id=?",
                        (doc_id,)).fetchone()
        return dict(row) if row else None


def delete_document(doc_id: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM uploaded_documents WHERE id=?", (doc_id,))
