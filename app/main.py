"""API operacional do Epidemiological Forecaster."""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Carrega a configuração local antes dos módulos que consultam DATABASE_URL.
# Em produção, variáveis já definidas pelo ambiente têm prioridade.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.dashboard_data import (
    DashboardRepository,
    RISK_LABELS_DISPLAY,
    RISK_ORDER,
    common_origins,
    display_category,
)

app = FastAPI(
    title="Epidemiological Forecaster API",
    version="3.0.0",
    description="Previsões de criticidade do XGBoost para os horizontes S+1 a S+4.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def repository() -> DashboardRepository:
    return DashboardRepository()


def cache_bucket() -> int:
    return int(time.time() // int(os.getenv("API_CACHE_TTL_SECONDS", "300")))


@lru_cache(maxsize=2)
def _predictions(_bucket: int) -> pd.DataFrame:
    return repository().predictions()


def predictions() -> pd.DataFrame:
    return _predictions(cache_bucket())


@lru_cache(maxsize=2)
def _observations(_bucket: int) -> pd.DataFrame:
    return repository().observations()


def observations() -> pd.DataFrame:
    return _observations(cache_bucket())


@lru_cache(maxsize=2)
def _climate(_bucket: int) -> pd.DataFrame:
    return repository().climate()


def climate() -> pd.DataFrame:
    return _climate(cache_bucket())


def records(frame: pd.DataFrame) -> list[dict]:
    """Converte tipos NumPy, NaN e timestamps em JSON válido."""
    return json.loads(frame.to_json(orient="records", force_ascii=False, date_format="iso"))


def select_origin(frame: pd.DataFrame, year: int, week: int) -> pd.DataFrame:
    selected = frame.loc[
        frame["epi_year_origem"].astype(int).eq(year)
        & frame["epi_week_origem"].astype(int).eq(week)
    ].copy()
    if selected.empty:
        raise HTTPException(status_code=404, detail=f"Não há previsões para a origem {year}-{week:02d}.")
    if "modo_resultado" in selected and selected["modo_resultado"].eq("producao").any():
        selected = selected.loc[selected["modo_resultado"].eq("producao")].copy()
    return selected


def weighted_recent_incidence(frame: pd.DataFrame) -> float:
    population = float(frame["populacao"].sum())
    if population <= 0:
        return 0.0
    return float(frame["casos_observados_recentes_4s"].sum() / population * 100_000)


@app.get("/")
def root() -> dict:
    return {
        "status": "Epidemiological Forecaster API online",
        "modo": "produção e validação temporal",
        "documentacao": "/docs",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/options")
def options() -> dict:
    frame = predictions()
    origins = common_origins(frame)
    if not origins:
        raise HTTPException(status_code=500, detail="Nenhuma semana possui os quatro horizontes completos.")
    latest = origins[-1]
    latest_rows = select_origin(frame, latest[0], latest[1])
    latest_mode = str(latest_rows["modo_resultado"].iloc[0])
    return {
        "origens": [
            {"epi_year": year, "epi_week": week, "label": f"{year}-{week:02d}"}
            for year, week in reversed(origins)
        ],
        "origem_padrao": {"epi_year": latest[0], "epi_week": latest[1], "label": f"{latest[0]}-{latest[1]:02d}"},
        "horizontes": [1, 2, 3, 4],
        "categorias": RISK_LABELS_DISPLAY,
        "bairros": sorted(frame["bairro_norm"].dropna().astype(str).unique().tolist()),
        "modo_resultado": latest_mode,
        "nota_probabilidades": "Probabilidades não calibradas; interpretar como confiança relativa do modelo.",
    }


@app.get("/api/dashboard")
def dashboard(
    origin_year: int,
    origin_week: int = Query(ge=1, le=53),
    horizon: int = Query(default=1, ge=1, le=4),
    level: str | None = None,
) -> dict:
    all_predictions = select_origin(predictions(), origin_year, origin_week)
    selected = all_predictions.loc[all_predictions["horizonte_semanas"].astype(int).eq(horizon)].copy()
    if selected.empty:
        raise HTTPException(status_code=404, detail="Horizonte indisponível para a semana escolhida.")

    unfiltered = selected.copy()
    normalized_level = display_category(level) if level else None
    if normalized_level and normalized_level in RISK_LABELS_DISPLAY:
        selected = selected.loc[selected["categoria_prevista"].eq(normalized_level)].copy()

    distribution = (
        unfiltered["categoria_prevista"]
        .value_counts()
        .reindex(RISK_LABELS_DISPLAY, fill_value=0)
        .rename_axis("categoria")
        .reset_index(name="bairros")
    )
    severity = unfiltered["categoria_prevista"].map(RISK_ORDER).fillna(-1)
    ranking = unfiltered.assign(_severity=severity).sort_values(
        ["_severity", "prob_critico", "prob_alto", "confianca_modelo"],
        ascending=False,
    ).drop(columns="_severity").head(12)

    horizon_distribution = (
        all_predictions.groupby(["horizonte_semanas", "categoria_prevista"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=RISK_LABELS_DISPLAY, fill_value=0)
        .reset_index()
    )

    target = unfiltered.iloc[0]
    return {
        "filtros": {
            "semana_origem": f"{origin_year}-{origin_week:02d}",
            "horizonte_semanas": horizon,
            "semana_alvo": target["semana_alvo"],
            "nivel": normalized_level or "Todos",
        },
        "kpis": {
            "bairros_monitorados": int(len(unfiltered)),
            "bairros_filtrados": int(len(selected)),
            "bairros_altos": int(unfiltered["categoria_prevista"].eq("Alto").sum()),
            "bairros_alto_critico": int(unfiltered["categoria_prevista"].isin(["Alto", "Crítico"]).sum()),
            "bairros_criticos": int(unfiltered["categoria_prevista"].eq("Crítico").sum()),
            "incidencia_recente_observada_4s_100k": weighted_recent_incidence(unfiltered),
            "confianca_media": float(unfiltered["confianca_modelo"].mean()),
        },
        "distribuicao": records(distribution),
        "distribuicao_horizontes": records(horizon_distribution),
        "ranking": records(ranking),
        "bairros": records(selected),
        "atualizacao": {
            "modelo": str(target["modelo"]),
            "versao": str(target["versao_modelo"]),
            "previsoes_geradas_em": str(target["data_geracao"]),
            "dados_processados_ate": f"{origin_year}-{origin_week:02d}",
            "modo": str(target["modo_resultado"]),
        },
    }


def history_until(origin_year: int, origin_week: int, neighborhood: str | None) -> pd.DataFrame:
    frame = observations().copy()
    if neighborhood:
        frame = frame.loc[frame["bairro_norm"].eq(neighborhood)]
    else:
        frame = frame.groupby(["epi_year", "epi_week", "time_index"], as_index=False)["casos_totais"].sum()
    origin_index = frame.loc[
        frame["epi_year"].astype(int).eq(origin_year)
        & frame["epi_week"].astype(int).eq(origin_week),
        "time_index",
    ]
    if origin_index.empty:
        raise HTTPException(status_code=404, detail="Semana de origem não encontrada no conjunto disponível.")
    history = frame.loc[frame["time_index"].le(int(origin_index.iloc[0]))].sort_values("time_index").tail(16).copy()
    history["casos_acumulados_4s"] = history["casos_totais"].rolling(4, min_periods=1).sum()
    history["semana"] = history["epi_year"].astype(int).astype(str) + "-" + history["epi_week"].astype(int).astype(str).str.zfill(2)
    return history


def climate_until(origin_year: int, origin_week: int) -> pd.DataFrame:
    frame = climate().drop_duplicates(["epi_year", "epi_week"]).copy()
    origin = frame.loc[
        frame["epi_year"].astype(int).eq(origin_year)
        & frame["epi_week"].astype(int).eq(origin_week)
    ]
    if origin.empty:
        raise HTTPException(status_code=404, detail="Clima indisponível para a semana de origem.")
    time_index = int(origin.iloc[0]["time_index"])
    output = frame.loc[frame["time_index"].le(time_index)].sort_values("time_index").tail(8).copy()
    output["semana"] = output["epi_year"].astype(int).astype(str) + "-" + output["epi_week"].astype(int).astype(str).str.zfill(2)
    return output


@app.get("/api/bairro")
def neighborhood_detail(
    origin_year: int,
    origin_week: int = Query(ge=1, le=53),
    bairro: str | None = None,
) -> dict:
    origin_predictions = select_origin(predictions(), origin_year, origin_week)
    neighborhood = bairro.strip().upper() if bairro and bairro.strip() else None
    if neighborhood:
        forecast = origin_predictions.loc[origin_predictions["bairro_norm"].eq(neighborhood)].copy()
        if forecast.empty:
            raise HTTPException(status_code=404, detail=f"Bairro não encontrado: {bairro}")
        population = int(forecast.iloc[0]["populacao"])
        recent_cases = int(forecast.iloc[0]["casos_observados_recentes_4s"])
        recent_incidence = float(forecast.iloc[0]["incidencia_recente_4s_100k"])
        recent_category = str(forecast.iloc[0]["categoria_recente_observada"])
        forecast_summary = forecast.sort_values("horizonte_semanas")
        title = neighborhood.title()
    else:
        population_rows = origin_predictions.loc[origin_predictions["horizonte_semanas"].eq(1)]
        population = int(population_rows["populacao"].sum())
        recent_cases = int(population_rows["casos_observados_recentes_4s"].sum())
        recent_incidence = float(recent_cases / population * 100_000) if population else 0.0
        recent_category = display_category(
            np.select(
                [recent_incidence < 100, recent_incidence < 300, recent_incidence < 500],
                ["Baixo", "Médio", "Alto"],
                default="Crítico",
            )
        )
        grouped = origin_predictions.groupby("horizonte_semanas", as_index=False).agg(
            semana_alvo=("semana_alvo", "first"),
            modo_resultado=("modo_resultado", "first"),
            confianca_modelo=("confianca_modelo", "mean"),
            prob_baixo=("prob_baixo", "mean"),
            prob_medio=("prob_medio", "mean"),
            prob_alto=("prob_alto", "mean"),
            prob_critico=("prob_critico", "mean"),
            bairros_alto_critico=("categoria_prevista", lambda values: int(values.isin(["Alto", "Crítico"]).sum())),
            bairros_criticos=("categoria_prevista", lambda values: int(values.eq("Crítico").sum())),
        )
        grouped["categoria_prevista"] = "Resumo municipal"
        forecast_summary = grouped
        title = "Recife"

    history = history_until(origin_year, origin_week, neighborhood)
    climate_history = climate_until(origin_year, origin_week)
    return {
        "escopo": "bairro" if neighborhood else "municipio",
        "nome": title,
        "bairro_norm": neighborhood,
        "semana_origem": f"{origin_year}-{origin_week:02d}",
        "populacao": population,
        "casos_observados_recentes_4s": recent_cases,
        "incidencia_recente_4s_100k": recent_incidence,
        "categoria_recente_observada": recent_category,
        "previsoes": records(forecast_summary),
        "historico": records(history),
        "clima": records(climate_history),
    }


@app.get("/api/performance")
def performance() -> dict:
    metrics = repository().metrics().sort_values("horizonte_semanas")
    first = metrics.iloc[0]
    return {
        "modelo": "XGBoost",
        "fonte": str(first.get("fonte", "validacao_2021")),
        "versao_modelo": str(first.get("versao_modelo", "criticidade_v2")),
        "semanas_avaliadas": None if pd.isna(first.get("semanas_avaliadas")) else int(first["semanas_avaliadas"]),
        "periodo_inicio": first.get("periodo_inicio"),
        "periodo_fim": first.get("periodo_fim"),
        "metricas": records(metrics),
    }


@app.get("/predict/recife")
def legacy_prediction(
    ano: int = 2021,
    semana: int = Query(default=48, ge=1, le=53),
    horizonte: int = Query(default=1, ge=1, le=4),
) -> list[dict]:
    """Compatibilidade com o endereço usado pela primeira versão do Streamlit."""
    frame = select_origin(predictions(), ano, semana)
    frame = frame.loc[frame["horizonte_semanas"].eq(horizonte)]
    return records(frame)
