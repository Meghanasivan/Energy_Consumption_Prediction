import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "prediction_history.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def initialize_database():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_date TEXT,
            prediction_time TEXT,
            predicted_consumption REAL,
            actual_consumption REAL,
            prediction_error REAL,
            model_version TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def save_prediction(date, time, predicted, actual=None, model_version="RF_v1"):
    error = None

    if actual is not None:
        error = abs(predicted - actual)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            prediction_date,
            prediction_time,
            predicted_consumption,
            actual_consumption,
            prediction_error,
            model_version
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (date, time, predicted, actual, error, model_version))

    conn.commit()
    conn.close()