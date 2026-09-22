"""Provisorio local: inspecciona qué ve el worker por su DATABASE_URL. BORRAR."""

import os

from dotenv import load_dotenv

load_dotenv(".env.local")

import psycopg

RID = "a94811f4-659a-422f-91bc-827796c25af6"

with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=10) as conn:
    print(
        "request:",
        conn.execute(
            "SELECT id, state FROM ai_integration.ai_generation_requests WHERE id = %s",
            (RID,),
        ).fetchall(),
    )
    print(
        "total requests:",
        conn.execute(
            "SELECT count(*) FROM ai_integration.ai_generation_requests"
        ).fetchone()[0],
    )
    print(
        "ownership table:",
        conn.execute(
            "SELECT to_regclass('app.routine_generation_ownership')"
        ).fetchone()[0],
    )
