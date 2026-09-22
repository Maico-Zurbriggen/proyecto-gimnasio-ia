from collections.abc import Iterator

import psycopg
from psycopg.rows import DictRow, dict_row

from gym_engine.config import Settings, get_settings


def get_connection(settings: Settings | None = None) -> psycopg.Connection[DictRow]:
    settings = settings or get_settings()
    return psycopg.connect(settings.database_url, row_factory=dict_row)


def connection_scope(
    settings: Settings | None = None,
) -> Iterator[psycopg.Connection[DictRow]]:
    conn = get_connection(settings)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
