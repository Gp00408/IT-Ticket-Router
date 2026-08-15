from sqlalchemy import text

from db import get_engine

engine = get_engine()

with engine.connect() as conn:
    result = conn.execute(text("SELECT 1"))
    print("Connection successful:", result.fetchone())
