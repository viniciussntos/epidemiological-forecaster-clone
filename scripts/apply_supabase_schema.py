"""Aplica o schema operacional sem depender de arquivos CSV locais."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.config import PROJECT_ROOT
from src.operational_store import create_database_engine, execute_migration


def main() -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL não foi configurada.")
    engine = create_database_engine(database_url)
    try:
        execute_migration(
            engine,
            PROJECT_ROOT / "supabase" / "migrations" / "001_operational_schema.sql",
        )
    finally:
        engine.dispose()
    print("Schema operacional aplicado com sucesso.")


if __name__ == "__main__":
    main()
