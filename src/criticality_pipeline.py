"""Painel temporal para classificacao direta da criticidade em quatro categorias."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import (
    CRITICALITY_LOOKBACK_WEEKS,
    CRITICALITY_WINDOW_WEEKS,
    RISK_LABELS,
    RISK_TO_ID,
)


CASE_OBSERVATION_COLUMNS = [f"casos_obs{lag}" for lag in range(1, CRITICALITY_LOOKBACK_WEEKS + 1)]
RAIN_OBSERVATION_COLUMNS = [f"chuva_obs{lag}" for lag in range(1, CRITICALITY_LOOKBACK_WEEKS + 1)]
TEMP_OBSERVATION_COLUMNS = [f"temp_obs{lag}" for lag in range(1, CRITICALITY_LOOKBACK_WEEKS + 1)]
RECIFE_OBSERVATION_COLUMNS = [f"recife_casos_obs{lag}" for lag in range(1, CRITICALITY_LOOKBACK_WEEKS + 1)]

CRITICALITY_NUMERIC_FEATURES = [
    "populacao",
    "target_week_sin",
    "target_week_cos",
    *CASE_OBSERVATION_COLUMNS,
    "casos_soma2",
    "casos_soma4",
    "casos_soma8",
    "casos_media4",
    "casos_tendencia",
    "casos_razao_crescimento",
    "incidencia_recente_4s_100k",
    *RAIN_OBSERVATION_COLUMNS,
    "chuva_soma4",
    "chuva_soma8",
    *TEMP_OBSERVATION_COLUMNS,
    "temp_media4",
    "temp_media8",
    *RECIFE_OBSERVATION_COLUMNS,
    "recife_casos_soma4",
    "recife_casos_soma8",
    "recife_casos_tendencia",
]

CRITICALITY_SEQUENCE_FEATURES = [
    "casos_totais",
    "precipitacao_total",
    "temp_max_media",
    "week_sin",
    "week_cos",
    "recife_casos_totais",
]

CRITICALITY_STATIC_NUMERIC_FEATURES = [
    "populacao",
    "target_week_sin",
    "target_week_cos",
    "casos_soma4",
    "casos_soma8",
    "casos_tendencia",
    "incidencia_recente_4s_100k",
    "recife_casos_soma4",
    "recife_casos_tendencia",
]


def category_from_incidence(incidence: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(incidence, dtype=float)
    return np.select(
        [values < 100, values < 300, values < 500],
        RISK_LABELS[:3],
        default=RISK_LABELS[3],
    )


def _observations(grouped, column: str, prefix: str, count: int) -> dict[str, pd.Series]:
    return {f"{prefix}{lag}": grouped[column].shift(lag - 1) for lag in range(1, count + 1)}


def build_criticality_panel(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Cria somente atributos disponiveis ate a semana de origem da previsao."""
    if horizon not in range(1, 5):
        raise ValueError("O horizonte deve estar entre 1 e 4 semanas.")
    panel = panel.sort_values(["bairro_norm", "time_index"]).reset_index(drop=True).copy()

    recife_week = (
        panel.groupby("time_index", as_index=False)["casos_totais"]
        .sum()
        .rename(columns={"casos_totais": "recife_casos_totais"})
    )
    panel = panel.merge(recife_week, on="time_index", how="left", validate="many_to_one")
    grouped = panel.groupby("bairro_norm", sort=False)

    feature_blocks = [
        pd.DataFrame(_observations(grouped, "casos_totais", "casos_obs", CRITICALITY_LOOKBACK_WEEKS)),
        pd.DataFrame(_observations(grouped, "precipitacao_total", "chuva_obs", CRITICALITY_LOOKBACK_WEEKS)),
        pd.DataFrame(_observations(grouped, "temp_max_media", "temp_obs", CRITICALITY_LOOKBACK_WEEKS)),
        pd.DataFrame(_observations(grouped, "recife_casos_totais", "recife_casos_obs", CRITICALITY_LOOKBACK_WEEKS)),
    ]
    panel = pd.concat([panel, *feature_blocks], axis=1)

    panel["casos_soma2"] = panel[["casos_obs1", "casos_obs2"]].sum(axis=1)
    panel["casos_soma4"] = panel[[f"casos_obs{i}" for i in range(1, 5)]].sum(axis=1)
    panel["casos_soma8"] = panel[CASE_OBSERVATION_COLUMNS].sum(axis=1)
    panel["casos_media4"] = panel[[f"casos_obs{i}" for i in range(1, 5)]].mean(axis=1)
    earlier_case_mean = panel[["casos_obs2", "casos_obs3", "casos_obs4"]].mean(axis=1)
    panel["casos_tendencia"] = panel["casos_obs1"] - earlier_case_mean
    panel["casos_razao_crescimento"] = (panel["casos_obs1"] + 1.0) / (earlier_case_mean + 1.0)
    panel["incidencia_recente_4s_100k"] = panel["casos_soma4"] / panel["populacao"] * 100_000

    panel["chuva_soma4"] = panel[[f"chuva_obs{i}" for i in range(1, 5)]].sum(axis=1)
    panel["chuva_soma8"] = panel[RAIN_OBSERVATION_COLUMNS].sum(axis=1)
    panel["temp_media4"] = panel[[f"temp_obs{i}" for i in range(1, 5)]].mean(axis=1)
    panel["temp_media8"] = panel[TEMP_OBSERVATION_COLUMNS].mean(axis=1)

    panel["recife_casos_soma4"] = panel[[f"recife_casos_obs{i}" for i in range(1, 5)]].sum(axis=1)
    panel["recife_casos_soma8"] = panel[RECIFE_OBSERVATION_COLUMNS].sum(axis=1)
    earlier_recife_mean = panel[["recife_casos_obs2", "recife_casos_obs3", "recife_casos_obs4"]].mean(axis=1)
    panel["recife_casos_tendencia"] = panel["recife_casos_obs1"] - earlier_recife_mean

    # A categoria-alvo usa a incidencia acumulada nas quatro semanas terminadas na semana futura.
    # Recriamos o agrupamento porque novas colunas foram anexadas ao painel acima.
    grouped = panel.groupby("bairro_norm", sort=False)
    panel["casos_acumulados_4s"] = grouped["casos_totais"].rolling(
        CRITICALITY_WINDOW_WEEKS, min_periods=CRITICALITY_WINDOW_WEEKS
    ).sum().reset_index(level=0, drop=True)
    panel["casos_acumulados_4s_alvo"] = grouped["casos_acumulados_4s"].shift(-horizon)
    panel["incidencia_4s_alvo_100k"] = panel["casos_acumulados_4s_alvo"] / panel["populacao"] * 100_000
    panel["categoria_criticidade_alvo"] = category_from_incidence(panel["incidencia_4s_alvo_100k"])
    panel.loc[panel["incidencia_4s_alvo_100k"].isna(), "categoria_criticidade_alvo"] = pd.NA
    panel["categoria_criticidade_id"] = panel["categoria_criticidade_alvo"].map(RISK_TO_ID).astype("Int64")
    panel["horizonte_semanas"] = horizon
    panel["janela_incidencia_semanas"] = CRITICALITY_WINDOW_WEEKS
    panel["target_week_sin"] = np.sin(2 * math.pi * panel["target_epi_week"] / 53.0)
    panel["target_week_cos"] = np.cos(2 * math.pi * panel["target_epi_week"] / 53.0)
    return panel
