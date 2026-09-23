"""Preparacao auditavel e compartilhada dos dados dos tres modelos."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR,
    DATA_START_YEAR,
    FORECAST_HORIZON_WEEKS,
    LOOKBACK_WEEKS,
    POPULATION_CENSUS_YEAR,
    TEST_YEAR,
    TRAIN_END_YEAR,
)


TABULAR_NUMERIC_FEATURES = [
    "populacao",
    "target_week_sin",
    "target_week_cos",
    "casos_lag1",
    "casos_lag2",
    "casos_lag3",
    "casos_lag4",
    "precipitacao_total",
    "chuva_lag1",
    "chuva_lag2",
    "chuva_lag3",
    "temp_max_media",
    "temp_max_lag1",
    "temp_max_lag2",
    "temp_max_lag3",
]

LSTM_SEQUENCE_FEATURES = [
    "casos_totais",
    "precipitacao_total",
    "temp_max_media",
    "week_sin",
    "week_cos",
]

LSTM_STATIC_NUMERIC_FEATURES = ["populacao", "target_week_sin", "target_week_cos"]


# Somente equivalencias deterministicas. Rotulos ambiguos continuam sem correspondencia.
NEIGHBORHOOD_ALIASES = {
    "sitio dos pintos": "SITIO DOS PINTOS SAO BRAS",
    "graas": "GRACAS",
    "morro da conceio": "MORRO DA CONCEICAO",
    "morro da conceiao": "MORRO DA CONCEICAO",
    "areais": "AREIAS",
    "brejo de guabiraba": "BREJO DA GUABIRABA",
    "inbiribeira": "IMBIRIBEIRA",
    "cordriro": "CORDEIRO",
    "macaeira": "MACAXEIRA",
    "jdjordao": "JORDAO",
    "joana bezerra": "ILHA JOANA BEZERRA",
    "mustadinha": "MUSTARDINHA",
    "ponro de parada": "PONTO DE PARADA",
    "san martim": "SAN MARTIN",
    "sante amaro": "SANTO AMARO",
    "sanato amaro": "SANTO AMARO",
    "tejipi": "TEJIPIO",
    "vasio dagama": "VASCO DA GAMA",
}


@dataclass(frozen=True)
class PipelinePaths:
    """Arquivos brutos informados explicitamente para uma preparação offline."""

    dengue: Path
    climate: Path
    population: Path
    output_dir: Path = DATA_DIR


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value).strip().lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _parse_epi_code(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    digits = series.astype("string").str.extract(r"^(\d{4})(\d{2})$")
    year = pd.to_numeric(digits[0], errors="coerce").astype("Int64")
    week = pd.to_numeric(digits[1], errors="coerce").astype("Int64")
    valid = year.between(1900, 2100) & week.between(1, 53)
    return year.where(valid), week.where(valid)


def _epi_week_from_date(value: pd.Timestamp) -> tuple[int, int]:
    """Semana epidemiologica domingo-sabado; semana 1 contem 4 de janeiro."""
    date = value.normalize()
    candidate_year = date.year
    week_one_start = pd.Timestamp(candidate_year, 1, 4)
    week_one_start -= pd.Timedelta(days=(week_one_start.dayofweek + 1) % 7)
    if date < week_one_start:
        candidate_year -= 1
        week_one_start = pd.Timestamp(candidate_year, 1, 4)
        week_one_start -= pd.Timedelta(days=(week_one_start.dayofweek + 1) % 7)
    next_start = pd.Timestamp(candidate_year + 1, 1, 4)
    next_start -= pd.Timedelta(days=(next_start.dayofweek + 1) % 7)
    if date >= next_start:
        return candidate_year + 1, 1
    return candidate_year, int((date - week_one_start).days // 7 + 1)


def _load_population(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    selected = raw.loc[raw["ano_censo"].eq(POPULATION_CENSUS_YEAR)].copy()
    selected["bairro_norm"] = selected["bairro_norm"].astype(str).str.strip().str.upper()
    selected["populacao"] = pd.to_numeric(selected["populacao"], errors="coerce")
    selected = selected.dropna(subset=["bairro_norm", "populacao"])
    if selected["bairro_norm"].duplicated().any():
        raise ValueError("O Censo 2022 possui bairros duplicados.")
    if len(selected) != 94:
        raise ValueError(f"Esperados 94 bairros no Censo 2022; encontrados {len(selected)}.")
    return selected[["ano_censo", "bairro_norm", "populacao"]].sort_values("bairro_norm")


def _build_neighborhood_lookup(population: pd.DataFrame) -> dict[str, str]:
    lookup = {normalize_text(name): name for name in population["bairro_norm"]}
    lookup.update(NEIGHBORHOOD_ALIASES)
    return lookup


def _append_removed(audits: list[pd.DataFrame], frame: pd.DataFrame, mask: pd.Series, reason: str) -> pd.DataFrame:
    removed = frame.loc[mask].copy()
    if not removed.empty:
        removed["motivo_remocao"] = reason
        audits.append(removed)
    return frame.loc[~mask].copy()


def _load_clean_dengue(path: Path, population: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path, sep=";", low_memory=False)
    epi_year, epi_week = _parse_epi_code(raw["ds_semana_sintoma"])
    raw = pd.concat(
        [raw, pd.DataFrame({"epi_year": epi_year, "epi_week": epi_week}, index=raw.index)], axis=1
    ).copy()

    # Fallback previsto no protocolo. Nesta base consolidada, sem_pri esta completo.
    symptom_date = pd.to_datetime(raw["dt_diagnostico_sintoma"], errors="coerce")
    fallback = raw["epi_year"].isna() & symptom_date.notna()
    fallback_values = symptom_date.loc[fallback].map(_epi_week_from_date)
    if not fallback_values.empty:
        raw.loc[fallback, "epi_year"] = fallback_values.map(lambda item: item[0]).astype("Int64")
        raw.loc[fallback, "epi_week"] = fallback_values.map(lambda item: item[1]).astype("Int64")
    raw["origem_semana_sintoma"] = np.where(fallback, "dt_sin_pri_fallback", "sem_pri")

    raw = raw.loc[raw["epi_year"].between(DATA_START_YEAR, TEST_YEAR)].copy()
    audits: list[pd.DataFrame] = []

    raw = _append_removed(
        audits,
        raw,
        raw["tp_classificacao_final"].isna(),
        "classificacao_final_ausente",
    )
    raw = _append_removed(
        audits,
        raw,
        pd.to_numeric(raw["tp_classificacao_final"], errors="coerce").eq(5),
        "classificacao_final_5_descartado",
    )
    raw = _append_removed(
        audits,
        raw,
        pd.to_numeric(raw["tp_duplicidade"], errors="coerce").fillna(0).eq(1),
        "marcado_tp_duplicidade_1",
    )
    raw = _append_removed(
        audits,
        raw,
        raw["possivel_repeticao_entre_arquivos"].fillna("").astype(str).str.strip().str.upper().eq("SIM"),
        "marcado_repeticao_entre_arquivos",
    )

    raw["bairro_original"] = raw["no_bairro_residencia"].fillna("").astype(str).str.strip()
    raw["bairro_chave"] = raw["bairro_original"].map(normalize_text)
    lookup = _build_neighborhood_lookup(population)
    raw["bairro_norm"] = raw["bairro_chave"].map(lookup)

    map_audit = (
        raw.groupby(["bairro_original", "bairro_chave", "bairro_norm"], dropna=False)
        .size()
        .rename("registros")
        .reset_index()
    )
    map_audit["status"] = np.where(map_audit["bairro_norm"].notna(), "mapeado", "nao_mapeado")
    map_audit = map_audit.sort_values(["status", "registros"], ascending=[True, False])

    raw = _append_removed(audits, raw, raw["bairro_norm"].isna(), "bairro_sem_correspondencia_censo_2022")

    dedup_keys = [
        "co_unidade_notificacao",
        "nu_notificacao",
        "dt_diagnostico_sintoma",
        "ds_semana_sintoma",
        "bairro_norm",
        "tp_classificacao_final",
    ]
    duplicate_mask = raw.duplicated(subset=dedup_keys, keep="first")
    raw = _append_removed(audits, raw, duplicate_mask, "duplicidade_analitica_mesma_chave")

    columns = [
        "id_registro_consolidado",
        "ano_arquivo_origem",
        "arquivo_origem",
        "linha_origem",
        "nu_notificacao",
        "co_unidade_notificacao",
        "dt_diagnostico_sintoma",
        "ds_semana_sintoma",
        "origem_semana_sintoma",
        "epi_year",
        "epi_week",
        "bairro_original",
        "bairro_norm",
        "tp_classificacao_final",
        "tp_duplicidade",
        "possivel_repeticao_entre_arquivos",
    ]
    clean = raw[columns].copy().sort_values(["epi_year", "epi_week", "bairro_norm"])

    if audits:
        removed = pd.concat(audits, ignore_index=True, sort=False)
        audit_columns = [col for col in columns if col in removed.columns] + ["motivo_remocao"]
        removed = removed[audit_columns]
    else:
        removed = pd.DataFrame(columns=columns + ["motivo_remocao"])
    return clean, removed, map_audit


def _load_clean_climate(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    climate = pd.read_csv(path)
    climate = climate.loc[climate["epi_year"].between(DATA_START_YEAR, TEST_YEAR)].copy()
    climate = climate.sort_values(["epi_year", "epi_week"]).reset_index(drop=True)
    if climate.duplicated(["epi_year", "epi_week"]).any():
        raise ValueError("Existem semanas climaticas duplicadas.")

    expected_counts = {2015: 52, 2016: 52, 2017: 52, 2018: 52, 2019: 52, 2020: 53, 2021: 52}
    actual_counts = climate.groupby("epi_year")["epi_week"].nunique().to_dict()
    if actual_counts != expected_counts:
        raise ValueError(f"Cobertura climatica inesperada: {actual_counts}")

    training = climate.loc[climate["epi_year"].between(DATA_START_YEAR, TRAIN_END_YEAR)]
    medians = (
        training.groupby("epi_week")[["precipitacao_total", "temp_max_media"]]
        .median()
        .rename(columns={
            "precipitacao_total": "mediana_precipitacao_2015_2020",
            "temp_max_media": "mediana_temperatura_2015_2020",
        })
    )
    climate = climate.merge(medians, left_on="epi_week", right_index=True, how="left")
    climate["precipitacao_original"] = climate["precipitacao_total"]
    climate["temperatura_original"] = climate["temp_max_media"]
    climate["precipitacao_imputada"] = climate["precipitacao_total"].isna()
    climate["temperatura_imputada"] = climate["temp_max_media"].isna()
    climate["precipitacao_total"] = climate["precipitacao_total"].fillna(
        climate["mediana_precipitacao_2015_2020"]
    )
    climate["temp_max_media"] = climate["temp_max_media"].fillna(
        climate["mediana_temperatura_2015_2020"]
    )
    if climate[["precipitacao_total", "temp_max_media"]].isna().any().any():
        raise ValueError("A mediana historica nao foi suficiente para preencher o clima.")

    climate["time_index"] = np.arange(len(climate), dtype=int)
    for lag in range(1, 5):
        climate[f"chuva_lag{lag}"] = climate["precipitacao_total"].shift(lag)
        climate[f"temp_max_lag{lag}"] = climate["temp_max_media"].shift(lag)
    climate["week_sin"] = np.sin(2 * math.pi * climate["epi_week"] / 53.0)
    climate["week_cos"] = np.cos(2 * math.pi * climate["epi_week"] / 53.0)

    imputation_audit = climate.loc[
        climate["precipitacao_imputada"] | climate["temperatura_imputada"],
        [
            "epi_year",
            "epi_week",
            "precipitacao_original",
            "precipitacao_total",
            "mediana_precipitacao_2015_2020",
            "precipitacao_imputada",
            "temperatura_original",
            "temp_max_media",
            "mediana_temperatura_2015_2020",
            "temperatura_imputada",
        ],
    ].copy()
    return climate, imputation_audit


def _build_weekly_panel(
    dengue: pd.DataFrame,
    climate: pd.DataFrame,
    population: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    if horizon not in range(1, 5):
        raise ValueError("O horizonte deve estar entre 1 e 4 semanas.")
    weeks = climate[["epi_year", "epi_week", "time_index"]].copy()
    neighborhoods = population[["bairro_norm", "populacao"]].copy()
    panel = neighborhoods.merge(weeks, how="cross")

    counts = (
        dengue.groupby(["bairro_norm", "epi_year", "epi_week"], as_index=False)
        .size()
        .rename(columns={"size": "casos_totais"})
    )
    panel = panel.merge(counts, on=["bairro_norm", "epi_year", "epi_week"], how="left")
    panel["casos_totais"] = panel["casos_totais"].fillna(0).astype(int)
    panel = panel.merge(climate, on=["epi_year", "epi_week", "time_index"], how="left")
    panel = panel.sort_values(["bairro_norm", "time_index"]).reset_index(drop=True)

    # lag1 e a semana mais recente disponivel na origem da previsao.
    for lag in range(1, LOOKBACK_WEEKS + 1):
        panel[f"casos_lag{lag}"] = panel.groupby("bairro_norm")["casos_totais"].shift(lag - 1)

    grouped = panel.groupby("bairro_norm", sort=False)
    panel["casos_alvo"] = grouped["casos_totais"].shift(-horizon)
    panel["target_epi_year"] = grouped["epi_year"].shift(-horizon)
    panel["target_epi_week"] = grouped["epi_week"].shift(-horizon)
    panel["target_time_index"] = grouped["time_index"].shift(-horizon)
    panel["target_week_sin"] = np.sin(2 * math.pi * panel["target_epi_week"] / 53.0)
    panel["target_week_cos"] = np.cos(2 * math.pi * panel["target_epi_week"] / 53.0)
    panel["taxa_incidencia_alvo_100k"] = panel["casos_alvo"] / panel["populacao"] * 100_000
    panel["horizonte_semanas"] = horizon
    return panel


def prepare_all(paths: PipelinePaths, horizon: int = FORECAST_HORIZON_WEEKS) -> dict[str, Path]:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    population = _load_population(paths.population)
    dengue, removed, neighborhood_audit = _load_clean_dengue(paths.dengue, population)
    climate, imputation_audit = _load_clean_climate(paths.climate)
    panel = _build_weekly_panel(dengue, climate, population, horizon)

    suffix = f"h{horizon}"
    outputs = {
        "population": paths.output_dir / "populacao_censo_2022_processada.csv",
        "dengue": paths.output_dir / "dengue_2015_2021_processada.csv",
        "climate": paths.output_dir / "clima_2015_2021_processado.csv",
        "panel": paths.output_dir / f"painel_semanal_2015_2021_{suffix}.csv",
        "removed": paths.output_dir / "auditoria_registros_dengue_removidos.csv",
        "neighborhoods": paths.output_dir / "auditoria_mapeamento_bairros.csv",
        "imputation": paths.output_dir / "auditoria_imputacao_climatica.csv",
    }
    _write_csv(population, outputs["population"])
    _write_csv(dengue, outputs["dengue"])
    _write_csv(climate, outputs["climate"])
    _write_csv(panel, outputs["panel"])
    _write_csv(removed, outputs["removed"])
    _write_csv(neighborhood_audit, outputs["neighborhoods"])
    _write_csv(imputation_audit, outputs["imputation"])

    removal_reasons = [
        "classificacao_final_ausente",
        "classificacao_final_5_descartado",
        "marcado_tp_duplicidade_1",
        "marcado_repeticao_entre_arquivos",
        "bairro_sem_correspondencia_censo_2022",
        "duplicidade_analitica_mesma_chave",
    ]
    removal_counts = removed["motivo_remocao"].value_counts().reindex(removal_reasons, fill_value=0)
    summary = {
        "horizonte_semanas": horizon,
        "registros_dengue_mantidos": int(len(dengue)),
        "registros_dengue_removidos": int(len(removed)),
        "remocoes_por_motivo": {key: int(value) for key, value in removal_counts.items()},
        "bairros_censo": int(len(population)),
        "rotulos_bairro_nao_mapeados": int((neighborhood_audit["status"] == "nao_mapeado").sum()),
        "semanas_clima": int(len(climate)),
        "valores_precipitacao_imputados": int(climate["precipitacao_imputada"].sum()),
        "valores_temperatura_imputados": int(climate["temperatura_imputada"].sum()),
        "linhas_painel": int(len(panel)),
        "linhas_modelaveis": int(panel[TABULAR_NUMERIC_FEATURES + ["casos_alvo"]].notna().all(axis=1).sum()),
    }
    summary_path = paths.output_dir / f"resumo_preparacao_{suffix}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["summary"] = summary_path
    return outputs
