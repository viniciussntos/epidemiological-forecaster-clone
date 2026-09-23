"""Contrato de dados do dashboard operacional de criticidade.

O módulo combina previsões operacionais e resultados da validação temporal com
os metadados e campos derivados usados pela API e pelo Streamlit.
"""

from __future__ import annotations

import os

import pandas as pd

from src.operational_store import (
    CLIMATE_TABLE,
    DENGUE_WEEKLY_TABLE,
    METRICS_TABLE,
    POPULATION_TABLE,
    PREDICTION_TABLE,
)


RISK_LABELS_DISPLAY = ["Baixo", "Médio", "Alto", "Crítico"]
RISK_ORDER = {label: index for index, label in enumerate(RISK_LABELS_DISPLAY)}
RISK_DISPLAY = {
    "Baixo": "Baixo",
    "Medio": "Médio",
    "Médio": "Médio",
    "Alto": "Alto",
    "Critico": "Crítico",
    "Crítico": "Crítico",
}


def display_category(value: object) -> object:
    """Padroniza os rótulos internos sem acento para exibição em português."""
    if pd.isna(value):
        return pd.NA
    return RISK_DISPLAY.get(str(value), str(value))


class DashboardRepository:
    """Acesso estrito às tabelas operacionais do Supabase."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise RuntimeError(
                "DATABASE_URL não foi configurada. A API não utiliza arquivos CSV como fallback."
            )
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            from sqlalchemy import create_engine

            self._engine = create_engine(self.database_url, pool_pre_ping=True)
        return self._engine

    def _read_table(self, table: str) -> pd.DataFrame:
        return pd.read_sql_table(table, self.engine)

    def predictions(self) -> pd.DataFrame:
        frame = self._read_table(PREDICTION_TABLE)
        for column in ["categoria_prevista", "categoria_observada_alvo", "categoria_recente_observada"]:
            if column in frame:
                frame[column] = frame[column].map(display_category)
        return frame

    def population(self) -> pd.DataFrame:
        return self._read_table(POPULATION_TABLE)

    def climate(self) -> pd.DataFrame:
        return self._read_table(CLIMATE_TABLE)

    def observations(self) -> pd.DataFrame:
        """Retorna casos semanais já completos, incluindo semanas com zero casos."""
        frame = self._read_table(DENGUE_WEEKLY_TABLE)
        required = ["bairro_norm", "epi_year", "epi_week", "time_index", "casos_totais"]
        missing = set(required).difference(frame.columns)
        if missing:
            raise ValueError(f"Colunas ausentes em {DENGUE_WEEKLY_TABLE}: {sorted(missing)}")
        return frame[required]

    def metrics(self) -> pd.DataFrame:
        metrics = self._read_table(METRICS_TABLE)
        production = metrics.loc[metrics["fonte"].eq("producao")].copy()
        if not production.empty:
            versions = (
                production.groupby("versao_modelo")["horizonte_semanas"]
                .nunique()
                .loc[lambda values: values.eq(4)]
            )
            if not versions.empty:
                eligible = production.loc[production["versao_modelo"].isin(versions.index)].copy()
                eligible["atualizado_em"] = pd.to_datetime(eligible["atualizado_em"], errors="coerce")
                latest_version = eligible.sort_values("atualizado_em").iloc[-1]["versao_modelo"]
                return eligible.loc[eligible["versao_modelo"].eq(latest_version)].reset_index(drop=True)
        validation = metrics.loc[metrics["fonte"].eq("validacao_2021")].copy()
        if not validation.empty:
            return validation.sort_values("horizonte_semanas").reset_index(drop=True)
        raise RuntimeError(f"A tabela {METRICS_TABLE} não possui métricas completas para exibição.")


def common_origins(predictions: pd.DataFrame) -> list[tuple[int, int]]:
    """Semanas de origem com previsões simultâneas em todos os horizontes."""
    origin_sets = []
    for horizon in range(1, 5):
        rows = predictions.loc[predictions["horizonte_semanas"].eq(horizon)]
        origin_sets.append(set(zip(rows["epi_year_origem"].astype(int), rows["epi_week_origem"].astype(int))))
    return sorted(set.intersection(*origin_sets))
