"""
SQLite database helper for Brain MRI Abnormality Detection system.
Stores users, patients, analysis results, and audit logs.
"""

import os
import sqlite3
from datetime import datetime

from werkzeug.security import generate_password_hash
from utils.security import decrypt_value, encrypt_value, is_encrypted

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'brain_mri.db')


def _encrypt_sensitive(value):
    return encrypt_value(value)


def _decrypt_sensitive(value, cast=None):
    decrypted = decrypt_value(value)
    if decrypted is None:
        return None
    if cast is int:
        try:
            return int(decrypted)
        except (TypeError, ValueError):
            return None
    return decrypted


def _migrate_sensitive_data(conn):
    user_rows = conn.execute("SELECT id, full_name FROM users").fetchall()
    for row in user_rows:
        if row["full_name"] and not is_encrypted(row["full_name"]):
            conn.execute(
                "UPDATE users SET full_name = ? WHERE id = ?",
                (_encrypt_sensitive(row["full_name"]), row["id"]),
            )

    patient_rows = conn.execute("SELECT id, name, age, gender FROM patients").fetchall()
    for row in patient_rows:
        updates = {}
        if row["name"] and not is_encrypted(row["name"]):
            updates["name"] = _encrypt_sensitive(row["name"])
        if row["age"] not in (None, '') and not is_encrypted(str(row["age"])):
            updates["age"] = _encrypt_sensitive(row["age"])
        if row["gender"] and not is_encrypted(row["gender"]):
            updates["gender"] = _encrypt_sensitive(row["gender"])
        if updates:
            conn.execute(
                """
                UPDATE patients
                SET name = COALESCE(?, name), age = COALESCE(?, age), gender = COALESCE(?, gender)
                WHERE id = ?
                """,
                (updates.get("name"), updates.get("age"), updates.get("gender"), row["id"]),
            )


def _decrypt_user_row(row):
    data = dict(row)
    data["full_name"] = _decrypt_sensitive(data.get("full_name"))
    data["patient_age"] = _decrypt_sensitive(data.get("patient_age"), cast=int)
    data["patient_gender"] = _decrypt_sensitive(data.get("patient_gender"))
    return data


def _decrypt_patient_row(row):
    data = dict(row)
    data["name"] = _decrypt_sensitive(data.get("name"))
    data["age"] = _decrypt_sensitive(data.get("age"), cast=int)
    data["gender"] = _decrypt_sensitive(data.get("gender"))
    return data


def _decrypt_analysis_row(row):
    data = dict(row)
    for key in ("patient_name", "patient_age", "patient_gender"):
        if key == "patient_age":
            data[key] = _decrypt_sensitive(data.get(key), cast=int)
        else:
            data[key] = _decrypt_sensitive(data.get(key))
    return data


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(conn, table_name):
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}


def _ensure_column(conn, table_name, column_name, definition):
    if column_name not in _table_columns(conn, table_name):
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


def _seed_default_users(conn):
    default_users = [
        ("admin", "admin123", "admin"),
        ("doctor", "doctor123", "doctor"),
    ]
    for username, password, role in default_users:
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if not existing:
            conn.execute(
                """
                INSERT INTO users (username, password_hash, role, full_name)
                VALUES (?, ?, ?, ?)
                """,
                (username, generate_password_hash(password), role, role.title()),
            )


def init_db():
    """Create tables if they don't exist and apply lightweight migrations."""
    conn = get_connection()
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            username        TEXT UNIQUE NOT NULL,
            password_hash   TEXT NOT NULL,
            role            TEXT DEFAULT 'patient',
            full_name       TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS patients (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT NOT NULL,
            age             INTEGER,
            gender          TEXT,
            patient_id      TEXT UNIQUE,
            user_id         INTEGER REFERENCES users(id),
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS analyses (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id          INTEGER REFERENCES patients(id),
            image_filename      TEXT NOT NULL,
            result_label        TEXT,
            confidence          REAL,
            anomaly_score       REAL,
            reconstruction_loss REAL,
            heatmap_path        TEXT,
            mask_path           TEXT,
            overlay_path        TEXT,
            analysed_by_user_id INTEGER REFERENCES users(id),
            analysed_at         TEXT DEFAULT (datetime('now')),
            model_version       TEXT DEFAULT '1.0'
        );

        CREATE TABLE IF NOT EXISTS audit_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER REFERENCES users(id),
            action          TEXT,
            details         TEXT,
            timestamp       TEXT DEFAULT (datetime('now')),
            ip_address      TEXT
        );
        """
    )

    _ensure_column(conn, "users", "full_name", "TEXT")
    _ensure_column(conn, "patients", "user_id", "INTEGER REFERENCES users(id)")
    _ensure_column(conn, "analyses", "overlay_path", "TEXT")
    _ensure_column(conn, "analyses", "analysed_by_user_id", "INTEGER REFERENCES users(id)")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_patients_user_id_unique ON patients(user_id) WHERE user_id IS NOT NULL"
    )
    _migrate_sensitive_data(conn)
    conn.commit()

    _seed_default_users(conn)
    conn.commit()
    conn.close()
    print("[DB] Database initialized.")


def create_user(username, password_hash, role, full_name=None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO users (username, password_hash, role, full_name)
        VALUES (?, ?, ?, ?)
        """,
        (username, password_hash, role, _encrypt_sensitive(full_name or username)),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return user_id


def get_user_by_username(username):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    conn.close()
    return _decrypt_user_row(row) if row else None


def get_user_by_id(user_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    conn.close()
    return _decrypt_user_row(row) if row else None


def get_user_with_patient_profile(user_id):
    conn = get_connection()
    row = conn.execute(
        """
        SELECT u.*, p.id AS linked_patient_db_id, p.patient_id AS linked_patient_code,
               p.age AS patient_age, p.gender AS patient_gender
        FROM users u
        LEFT JOIN patients p ON p.user_id = u.id
        WHERE u.id = ?
        """,
        (user_id,),
    ).fetchone()
    conn.close()
    return _decrypt_user_row(row) if row else None


def get_all_users():
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT u.*, p.id AS linked_patient_db_id, p.patient_id AS linked_patient_code,
               p.age AS patient_age, p.gender AS patient_gender
        FROM users u
        LEFT JOIN patients p ON p.user_id = u.id
        ORDER BY u.created_at DESC
        """
    ).fetchall()
    conn.close()
    return [_decrypt_user_row(row) for row in rows]


def create_patient(name, age, gender, patient_id=None, user_id=None):
    conn = get_connection()
    cur = conn.cursor()
    if user_id:
        existing = cur.execute(
            "SELECT id FROM patients WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if existing:
            conn.close()
            return existing["id"]

    if not patient_id:
        patient_id = f"PAT{datetime.now().strftime('%Y%m%d%H%M%S')}"

    cur.execute(
        """
        INSERT INTO patients (name, age, gender, patient_id, user_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        (_encrypt_sensitive(name), _encrypt_sensitive(age), _encrypt_sensitive(gender), patient_id, user_id),
    )
    conn.commit()
    patient_db_id = cur.lastrowid
    conn.close()
    return patient_db_id


def update_patient(patient_db_id, name, age, gender):
    conn = get_connection()
    conn.execute(
        """
        UPDATE patients
        SET name = ?, age = ?, gender = ?
        WHERE id = ?
        """,
        (_encrypt_sensitive(name), _encrypt_sensitive(age), _encrypt_sensitive(gender), patient_db_id),
    )
    conn.commit()
    conn.close()


def unlink_patient_user(patient_db_id):
    conn = get_connection()
    conn.execute(
        "UPDATE patients SET user_id = NULL WHERE id = ?",
        (patient_db_id,),
    )
    conn.commit()
    conn.close()


def delete_patient(patient_db_id):
    conn = get_connection()
    conn.execute(
        "DELETE FROM patients WHERE id = ?",
        (patient_db_id,),
    )
    conn.commit()
    conn.close()


def update_user(user_id, username, role, full_name=None, password_hash=None):
    conn = get_connection()
    if password_hash:
        conn.execute(
            """
            UPDATE users
            SET username = ?, role = ?, full_name = ?, password_hash = ?
            WHERE id = ?
            """,
            (username, role, _encrypt_sensitive(full_name), password_hash, user_id),
        )
    else:
        conn.execute(
            """
            UPDATE users
            SET username = ?, role = ?, full_name = ?
            WHERE id = ?
            """,
            (username, role, _encrypt_sensitive(full_name), user_id),
        )
    conn.commit()
    conn.close()


def count_patient_analyses(patient_db_id):
    conn = get_connection()
    total = conn.execute(
        "SELECT COUNT(*) FROM analyses WHERE patient_id = ?",
        (patient_db_id,),
    ).fetchone()[0]
    conn.close()
    return total


def clear_analysis_author(user_id):
    conn = get_connection()
    conn.execute(
        "UPDATE analyses SET analysed_by_user_id = NULL WHERE analysed_by_user_id = ?",
        (user_id,),
    )
    conn.commit()
    conn.close()


def delete_user(user_id):
    conn = get_connection()
    conn.execute(
        "DELETE FROM audit_log WHERE user_id = ?",
        (user_id,),
    )
    conn.execute(
        "DELETE FROM users WHERE id = ?",
        (user_id,),
    )
    conn.commit()
    conn.close()


def get_patient(patient_db_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM patients WHERE id = ?",
        (patient_db_id,),
    ).fetchone()
    conn.close()
    return _decrypt_patient_row(row) if row else None


def get_patient_by_user_id(user_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM patients WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    conn.close()
    return _decrypt_patient_row(row) if row else None


def get_all_patients():
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM patients ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [_decrypt_patient_row(r) for r in rows]


def get_patient_options():
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT p.id, p.name, p.patient_id, p.age, p.gender, u.username
        FROM patients p
        LEFT JOIN users u ON u.id = p.user_id
        ORDER BY p.created_at DESC
        """
    ).fetchall()
    conn.close()
    return [_decrypt_patient_row(row) for row in rows]


def save_analysis(
    patient_db_id,
    image_filename,
    result_label,
    confidence,
    anomaly_score,
    recon_loss,
    heatmap_path=None,
    mask_path=None,
    overlay_path=None,
    model_version='1.0',
    analysed_by_user_id=None,
):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO analyses
            (patient_id, image_filename, result_label, confidence,
             anomaly_score, reconstruction_loss, heatmap_path, mask_path,
             overlay_path, model_version, analysed_by_user_id)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            patient_db_id,
            image_filename,
            result_label,
            confidence,
            anomaly_score,
            recon_loss,
            heatmap_path,
            mask_path,
            overlay_path,
            model_version,
            analysed_by_user_id,
        ),
    )
    conn.commit()
    analysis_id = cur.lastrowid
    conn.close()
    return analysis_id


def get_analyses(patient_db_id=None):
    conn = get_connection()
    query = """
        SELECT a.*, p.name AS patient_name, p.patient_id AS patient_code,
               u.username AS analysed_by_username
        FROM analyses a
        JOIN patients p ON a.patient_id = p.id
        LEFT JOIN users u ON a.analysed_by_user_id = u.id
    """
    params = ()
    if patient_db_id:
        query += " WHERE a.patient_id = ?"
        params = (patient_db_id,)
    query += " ORDER BY a.analysed_at DESC LIMIT 100"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [_decrypt_analysis_row(r) for r in rows]


def get_analysis_by_id(analysis_id):
    conn = get_connection()
    row = conn.execute(
        """
        SELECT a.*, p.name AS patient_name, p.age AS patient_age, p.gender AS patient_gender,
               p.patient_id AS patient_code, p.user_id AS patient_user_id,
               u.username AS analysed_by_username
        FROM analyses a
        JOIN patients p ON a.patient_id = p.id
        LEFT JOIN users u ON a.analysed_by_user_id = u.id
        WHERE a.id = ?
        """,
        (analysis_id,),
    ).fetchone()
    conn.close()
    return _decrypt_analysis_row(row) if row else None


def get_dashboard_stats(patient_db_id=None):
    conn = get_connection()
    if patient_db_id:
        total_patients = 1
        total_scans = conn.execute(
            "SELECT COUNT(*) FROM analyses WHERE patient_id = ?",
            (patient_db_id,),
        ).fetchone()[0]
        abnormal = conn.execute(
            "SELECT COUNT(*) FROM analyses WHERE patient_id = ? AND result_label != 'Normal'",
            (patient_db_id,),
        ).fetchone()[0]
    else:
        total_patients = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        total_scans = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        abnormal = conn.execute(
            "SELECT COUNT(*) FROM analyses WHERE result_label != 'Normal'"
        ).fetchone()[0]
    normal = total_scans - abnormal
    conn.close()
    return {
        'total_patients': total_patients,
        'total_scans': total_scans,
        'abnormal_count': abnormal,
        'normal_count': normal,
        'abnormal_pct': round((abnormal / total_scans * 100) if total_scans else 0, 1),
    }


def log_action(user_id, action, details='', ip_address=''):
    conn = get_connection()
    conn.execute(
        "INSERT INTO audit_log (user_id, action, details, ip_address) VALUES (?,?,?,?)",
        (user_id, action, details, ip_address),
    )
    conn.commit()
    conn.close()


if __name__ == '__main__':
    init_db()
    print("DB ready at:", DB_PATH)
