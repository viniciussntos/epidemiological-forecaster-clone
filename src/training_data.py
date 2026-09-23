"""Carregamento dos painéis de treino diretamente das tabelas operacionais."""

from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv

from src.config import PROJECT_ROOT
from src.criticality_pipeline import CRITICALITY_NUMERIC_FEATURES
from src.operational_pipeline import build_operational_base, build_operational_features
from src.operational_store import (
    CLIMATE_TABLE,
    DENGUE_WEEKLY_TABLE,
    POPULATION_TABLE,
    create_database_engine,
    read_table,
)


load_dotenv(PROJECT_ROOT / ".env")


def require_database_url(database_url: str | None = None) -> str:
    value = database_url or os.getenv("DATABASE_URL")
    if not value:
        raise RuntimeError(
            "DATABASE_URL não foi configurada. O treinamento usa exclusivamente as tabelas operacionais do Supabase."
        )
    return value


def load_operational_base_from_database(database_url: str | None = None) -> pd.DataFrame:
    engine = create_database_engine(require_database_url(database_url))
    try:
        population = read_table(engine, POPULATION_TABLE)
        climate = read_table(engine, CLIMATE_TABLE)
        dengue = read_table(engine, DENGUE_WEEKLY_TABLE)
    finally:
        engine.dispose()
    return build_operational_base(population, climate, dengue)


def build_training_panel(base: pd.DataFrame, horizon: int) -> pd.DataFrame:
    panel = build_operational_features(base, horizon)
    required = CRITICALITY_NUMERIC_FEATURES + [
        "bairro_norm",
        "categoria_criticidade_id",
        "target_epi_year",
    ]
    return panel.dropna(subset=required).reset_index(drop=True)


def load_criticality_training_panel(
    horizon: int,
    database_url: str | None = None,
) -> pd.DataFrame:
    return build_training_panel(load_operational_base_from_database(database_url), horizon)
