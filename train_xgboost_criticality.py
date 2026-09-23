"""XGBoost multiclasse com busca e validacao temporal progressiva."""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import ParameterSampler

from src.config import (
    CRITICALITY_MODELS_DIR, CRITICALITY_RESULTS_DIR, FORECAST_HORIZON_WEEKS,
    RANDOM_SEED, TEST_YEAR, TRAIN_END_YEAR, XGB_OPTIMIZATION_CANDIDATES,
)
from src.criticality_evaluation import evaluate_criticality, save_criticality_evaluation
from src.criticality_modeling import class_weights, make_criticality_preprocessor, set_seed
from src.criticality_pipeline import CRITICALITY_NUMERIC_FEATURES
from src.temporal_validation import fold_metric_row, pooled_validation_metrics, progressive_masks
from src.training_data import load_criticality_training_panel


FEATURES = CRITICALITY_NUMERIC_FEATURES + ["bairro_norm"]
BASELINE = {
    "learning_rate": 0.035, "max_depth": 5, "min_child_weight": 2,
    "gamma": 0.0, "subsample": 0.85, "colsample_bytree": 0.85,
    "reg_lambda": 2.0, "reg_alpha": 0.05, "weight_power": 0.75,
}
SEARCH_SPACE = {
    "learning_rate": [0.02, 0.035, 0.05, 0.08],
    "max_depth": [3, 4, 5, 6, 7],
    "min_child_weight": [1, 2, 4, 8],
    "gamma": [0.0, 0.1, 0.5, 1.0],
    "subsample": [0.70, 0.85, 1.0],
    "colsample_bytree": [0.70, 0.85, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
    "reg_alpha": [0.0, 0.05, 0.2, 0.5, 1.0],
    "weight_power": [0.25, 0.5, 0.75, 1.0],
}


def model_for(params: dict, n_estimators: int, early_stopping_rounds=None):
    model_params = {key: value for key, value in params.items() if key != "weight_power"}
    return xgb.XGBClassifier(
        objective="multi:softprob", num_class=4, eval_metric="mlogloss",
        n_estimators=n_estimators, random_state=RANDOM_SEED, n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds, **model_params,
    )


def candidates() -> list[dict]:
    sampled = list(ParameterSampler(
        SEARCH_SPACE, n_iter=XGB_OPTIMIZATION_CANDIDATES - 1, random_state=RANDOM_SEED
    ))
    return [BASELINE, *sampled]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=FORECAST_HORIZON_WEEKS, choices=range(1, 5))
    args = parser.parse_args()
    set_seed(RANDOM_SEED)
    panel = load_criticality_training_panel(args.horizon)
    train_full = panel.loc[panel["target_epi_year"].le(TRAIN_END_YEAR)].copy()
    test = panel.loc[panel["target_epi_year"].eq(TEST_YEAR)].copy()

    fold_data = []
    for year, train_mask, validation_mask in progressive_masks(train_full):
        fold_train = train_full.loc[train_mask]
        fold_validation = train_full.loc[validation_mask]
        selector = make_criticality_preprocessor()
        fold_data.append({
            "year": year,
            "x_train": selector.fit_transform(fold_train[FEATURES]).astype(np.float32),
            "y_train": fold_train["categoria_criticidade_id"].to_numpy(dtype=int),
            "x_validation": selector.transform(fold_validation[FEATURES]).astype(np.float32),
            "y_validation": fold_validation["categoria_criticidade_id"].to_numpy(dtype=int),
        })

    optimization_rows, candidate_fold_rows = [], {}
    for candidate_id, params in enumerate(candidates(), start=1):
        actual_parts, probability_parts, trees, folds = [], [], [], []
        for fold in fold_data:
            weights = class_weights(fold["y_train"], float(params["weight_power"]))
            model = model_for(params, 600, early_stopping_rounds=40)
            model.fit(
                fold["x_train"], fold["y_train"], sample_weight=weights[fold["y_train"]],
                eval_set=[(fold["x_validation"], fold["y_validation"])], verbose=False,
            )
            probabilities = model.predict_proba(fold["x_validation"])
            best_trees = int(model.best_iteration) + 1
            trees.append(best_trees)
            actual_parts.append(fold["y_validation"])
            probability_parts.append(probabilities)
            row = fold_metric_row(fold["year"], fold["y_validation"], probabilities)
            row["arvores_selecionadas"] = best_trees
            folds.append(row)
        pooled = pooled_validation_metrics(actual_parts, probability_parts)
        native_params = {key: value.item() if hasattr(value, "item") else value for key, value in params.items()}
        optimization_rows.append({
            "candidato": candidate_id, **native_params,
            "arvores_mediana": int(np.rint(np.median(trees))),
            **{f"validacao_oof_{key}": value for key, value in pooled.items()},
        })
        candidate_fold_rows[candidate_id] = folds
        print(f"Candidato {candidate_id}/{XGB_OPTIMIZATION_CANDIDATES}: F2 macro OOF={pooled['f2_macro']:.4f}", flush=True)

    optimization = pd.DataFrame(optimization_rows).sort_values(
        ["validacao_oof_f2_macro", "validacao_oof_acuracia_balanceada", "validacao_oof_log_loss"],
        ascending=[False, False, True],
    )
    best = optimization.iloc[0]
    best_id = int(best["candidato"])
    best_params = {key: best[key] for key in SEARCH_SPACE}
    best_params["max_depth"] = int(best_params["max_depth"])
    best_params["min_child_weight"] = int(best_params["min_child_weight"])
    best_trees = max(25, int(best["arvores_mediana"]))

    final_selector = make_criticality_preprocessor()
    x_train = final_selector.fit_transform(train_full[FEATURES]).astype(np.float32)
    x_test = final_selector.transform(test[FEATURES]).astype(np.float32)
    y_train = train_full["categoria_criticidade_id"].to_numpy(dtype=int)
    final_weights = class_weights(y_train, float(best_params["weight_power"]))
    final_model = model_for(best_params, best_trees)
    final_model.fit(x_train, y_train, sample_weight=final_weights[y_train], verbose=False)
    parts = evaluate_criticality("XGBoost", test, final_model.predict_proba(x_test), args.horizon)
    parts[1]["arvores_selecionadas"] = best_trees
    parts[1]["potencia_peso_classes"] = float(best_params["weight_power"])
    parts[1]["validacao_oof_f2_macro"] = float(best["validacao_oof_f2_macro"])
    parts[1]["validacao_oof_acuracia_balanceada"] = float(best["validacao_oof_acuracia_balanceada"])
    save_criticality_evaluation(parts, "xgboost", args.horizon, CRITICALITY_RESULTS_DIR)

    CRITICALITY_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CRITICALITY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    final_model.save_model(CRITICALITY_MODELS_DIR / f"xgboost_criticidade_h{args.horizon}.json")
    joblib.dump(final_selector, CRITICALITY_MODELS_DIR / f"preprocessador_xgboost_criticidade_h{args.horizon}.joblib")
    optimization.to_csv(CRITICALITY_RESULTS_DIR / f"otimizacao_xgboost_h{args.horizon}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(candidate_fold_rows[best_id]).to_csv(
        CRITICALITY_RESULTS_DIR / f"validacao_progressiva_xgboost_h{args.horizon}.csv", index=False, encoding="utf-8-sig"
    )
    best_payload = {"candidato": best_id, "arvores": best_trees, **best_params}
    (CRITICALITY_RESULTS_DIR / f"melhores_hiperparametros_xgboost_h{args.horizon}.json").write_text(
        json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame({"feature": final_selector.get_feature_names_out(), "importancia": final_model.feature_importances_}).sort_values("importancia", ascending=False).to_csv(
        CRITICALITY_RESULTS_DIR / f"importancia_features_xgboost_h{args.horizon}.csv", index=False, encoding="utf-8-sig"
    )
    print(parts[1].to_string(index=False))


if __name__ == "__main__":
    main()
