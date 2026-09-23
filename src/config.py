"""Configuracao central do experimento comparativo."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
CRITICALITY_RESULTS_DIR = RESULTS_DIR / "criticidade_v2"
CRITICALITY_MODELS_DIR = MODELS_DIR / "criticidade_v2"

# Altere para 1, 2, 3 ou 4. Tambem pode ser sobrescrito por --horizon na linha de comando.
FORECAST_HORIZON_WEEKS = 1
LOOKBACK_WEEKS = 4
CRITICALITY_LOOKBACK_WEEKS = 8
CRITICALITY_WINDOW_WEEKS = 4

DATA_START_YEAR = 2015
TRAIN_END_YEAR = 2020
TEST_YEAR = 2021
VALIDATION_YEAR = 2020
# Validacao por origem temporal: cada ano e previsto usando somente anos anteriores.
PROGRESSIVE_VALIDATION_YEARS = tuple(range(DATA_START_YEAR + 1, TRAIN_END_YEAR + 1))
XGB_OPTIMIZATION_CANDIDATES = 6
POPULATION_CENSUS_YEAR = 2022
RANDOM_SEED = 42

RISK_LABELS = ["Baixo", "Medio", "Alto", "Critico"]
RISK_TO_ID = {label: index for index, label in enumerate(RISK_LABELS)}
ID_TO_RISK = {index: label for label, index in RISK_TO_ID.items()}
