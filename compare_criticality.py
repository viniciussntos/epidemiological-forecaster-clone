"""Consolida os resultados dos classificadores e das referencias simples."""

import argparse

import numpy as np
import pandas as pd

from src.config import CRITICALITY_RESULTS_DIR, FORECAST_HORIZON_WEEKS, RISK_TO_ID, TEST_YEAR
from src.criticality_evaluation import evaluate_reference, save_criticality_evaluation
from src.criticality_pipeline import category_from_incidence
from src.training_data import load_criticality_training_panel


MODELS = [("xgboost", "XGBoost"), ("rna", "RNA"), ("lstm", "LSTM")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=FORECAST_HORIZON_WEEKS, choices=range(1, 5))
    args = parser.parse_args()
    panel = load_criticality_training_panel(args.horizon)
    test = panel.loc[panel["target_epi_year"].eq(TEST_YEAR)].copy()
    majority = np.zeros(len(test), dtype=int)
    persistence_labels = category_from_incidence(test["incidencia_recente_4s_100k"])
    persistence = np.asarray([RISK_TO_ID[label] for label in persistence_labels], dtype=int)
    for slug, name, predicted in [
        ("referencia_maioria", "Referencia: sempre Baixo", majority),
        ("referencia_persistencia", "Referencia: persistencia", persistence),
    ]:
        parts = evaluate_reference(name, test, predicted, args.horizon)
        save_criticality_evaluation(parts, slug, args.horizon, CRITICALITY_RESULTS_DIR)

    slugs = [slug for slug, _ in MODELS] + ["referencia_maioria", "referencia_persistencia"]
    metrics = pd.concat([pd.read_csv(CRITICALITY_RESULTS_DIR / f"metricas_{slug}_h{args.horizon}.csv") for slug in slugs], ignore_index=True)
    by_class = pd.concat([pd.read_csv(CRITICALITY_RESULTS_DIR / f"metricas_por_categoria_{slug}_h{args.horizon}.csv") for slug in slugs], ignore_index=True)
    confusion = pd.concat([pd.read_csv(CRITICALITY_RESULTS_DIR / f"matriz_confusao_{slug}_h{args.horizon}.csv") for slug in slugs], ignore_index=True)
    metrics["tipo"] = np.where(metrics["modelo"].str.startswith("Referencia"), "referencia", "modelo")
    metrics = metrics.sort_values(["tipo", "f1_macro"], ascending=[True, False])
    metrics.to_csv(CRITICALITY_RESULTS_DIR / f"comparacao_modelos_criticidade_h{args.horizon}.csv", index=False, encoding="utf-8-sig")
    by_class.to_csv(CRITICALITY_RESULTS_DIR / f"comparacao_por_categoria_h{args.horizon}.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(CRITICALITY_RESULTS_DIR / f"matrizes_confusao_h{args.horizon}.csv", index=False, encoding="utf-8-sig")
    print(metrics[["modelo", "acuracia", "acuracia_balanceada", "f1_macro", "f2_macro", "pr_auc_macro"]].to_string(index=False))


if __name__ == "__main__":
    main()
