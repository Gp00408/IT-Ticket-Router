import json
import os

import pandas as pd
from sqlalchemy import text

from db import get_engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "synthetic_tickets.json")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "tickets.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def load_raw() -> pd.DataFrame:
    with open(RAW_PATH, encoding="utf-8") as f:
        records = json.load(f)
    return pd.DataFrame(records)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["ticket_text", "category", "urgency"]).copy()
    df["ticket_text"] = df["ticket_text"].str.strip()
    df = df[df["ticket_text"].str.len() > 0]
    df = df.drop_duplicates(subset=["ticket_text"])
    return df.reset_index(drop=True)


def load_to_db(df: pd.DataFrame, engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id SERIAL PRIMARY KEY,
                ticket_text TEXT NOT NULL,
                category VARCHAR(50) NOT NULL,
                urgency VARCHAR(20) NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text("TRUNCATE TABLE tickets RESTART IDENTITY"))
    df[["ticket_text", "category", "urgency"]].to_sql(
        "tickets", engine, if_exists="append", index=False
    )


def main():
    df = load_raw()
    df = clean_dataset(df)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    engine = get_engine()
    load_to_db(df, engine)
    print(f"Loaded {len(df)} tickets into the database.")


if __name__ == "__main__":
    main()
