"""Valida dados, previsoes, folds e modelos dos quatro horizontes."""

import json

import numpy as np
import pandas as pd

from src.config import CRITICALITY_MODELS_DIR, CRITICALITY_RESULTS_DIR, PROJECT_ROOT
from src.training_data import build_training_panel, load_operational_base_from_database


def main() -> None:
    checks = {}
    probability_columns = ["prob_baixo", "prob_medio", "prob_alto", "prob_critico"]
    base = load_operational_base_from_database()
    for horizon in range(1, 5):
        panel = build_training_panel(base, horizon)
        test = panel.loc[panel["target_epi_year"].eq(2021)]
        checks[f"h{horizon}_amostras_teste_4888"] = len(test) == 4888
        checks[f"h{horizon}_quatro_categorias_teste"] = test["categoria_criticidade_alvo"].nunique() == 4
        checks[f"h{horizon}_distancia_alvo_correta"] = bool(
            (panel["target_time_index"] - panel["time_index"]).eq(horizon).all()
        )
        for slug in ["xgboost", "rna", "lstm"]:
            predictions = pd.read_csv(CRITICALITY_RESULTS_DIR / f"previsoes_{slug}_h{horizon}.csv")
            probabilities = predictions[probability_columns].to_numpy(dtype=float)
            checks[f"h{horizon}_{slug}_4888_previsoes"] = len(predictions) == 4888
            checks[f"h{horizon}_{slug}_probabilidades_validas"] = bool(
                np.isfinite(probabilities).all() and (probabilities >= 0).all()
                and np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
            )
            validation = pd.read_csv(CRITICALITY_RESULTS_DIR / f"validacao_progressiva_{slug}_h{horizon}.csv")
            checks[f"h{horizon}_{slug}_cinco_folds"] = validation["ano_validacao"].tolist() == [2016, 2017, 2018, 2019, 2020]
        optimization = pd.read_csv(CRITICALITY_RESULTS_DIR / f"otimizacao_xgboost_h{horizon}.csv")
        checks[f"h{horizon}_xgboost_seis_candidatos"] = len(optimization) == 6
        required_models = [
            CRITICALITY_MODELS_DIR / f"xgboost_criticidade_h{horizon}.json",
            CRITICALITY_MODELS_DIR / f"rna_criticidade_h{horizon}.pt",
            CRITICALITY_MODELS_DIR / f"lstm_criticidade_h{horizon}.pt",
        ]
        checks[f"h{horizon}_tres_modelos_salvos"] = all(path.exists() and path.stat().st_size > 0 for path in required_models)

    consolidated = pd.read_csv(CRITICALITY_RESULTS_DIR / "comparacao_todos_horizontes.csv")
    checks["comparacao_contem_20_linhas"] = len(consolidated) == 20
    checks["comparacao_contem_quatro_horizontes"] = sorted(consolidated["horizonte_semanas"].unique()) == [1, 2, 3, 4]
    checks["planilha_final_salva"] = (CRITICALITY_RESULTS_DIR / "resultados_validacao_progressiva_h1_h4.xlsx").exists()
    payload = {"status": "OK" if all(checks.values()) else "FALHA", "checks": checks}
    output = PROJECT_ROOT / "docs" / "validacao_criticidade_progressiva.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
