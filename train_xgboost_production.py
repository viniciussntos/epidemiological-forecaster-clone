"""Retreina os quatro classificadores XGBoost com todos os dados consolidados."""

from __future__ import annotations

import argparse
from datetime import datetime
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CRITICALITY_MODELS_DIR, CRITICALITY_RESULTS_DIR, RANDOM_SEED, TEST_YEAR
from src.criticality_pipeline import CRITICALITY_NUMERIC_FEATURES
from src.training_data import build_training_panel, load_operational_base_from_database


FEATURES = CRITICALITY_NUMERIC_FEATURES + ["bairro_norm"]


def make_production_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), CRITICALITY_NUMERIC_FEATURES),
            ("bairro", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["bairro_norm"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def production_class_weights(labels: np.ndarray, power: float) -> np.ndarray:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=4).astype(float)
    if np.any(counts == 0):
        raise ValueError(f"Treino sem exemplos de alguma categoria: {counts.tolist()}")
    weights = (len(labels) / (4 * counts)) ** power
    return weights / weights.mean()


def production_model(parameters: dict, trees: int) -> xgb.XGBClassifier:
    model_parameters = {key: value for key, value in parameters.items() if key != "weight_power"}
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        n_estimators=int(trees),
        random_state=RANDOM_SEED,
        n_jobs=-1,
        **model_parameters,
    )


def load_selected_configuration(horizon: int) -> tuple[dict, int]:
    path = CRITICALITY_RESULTS_DIR / f"melhores_hiperparametros_xgboost_h{horizon}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    parameters = {
        key: payload[key]
        for key in [
            "learning_rate", "max_depth", "min_child_weight", "gamma", "subsample",
            "colsample_bytree", "reg_lambda", "reg_alpha", "weight_power",
        ]
    }
    parameters["max_depth"] = int(parameters["max_depth"])
    parameters["min_child_weight"] = int(parameters["min_child_weight"])
    return parameters, int(payload["arvores"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-through-year", type=int, default=TEST_YEAR)
    parser.add_argument("--version", type=str)
    args = parser.parse_args()
    np.random.seed(42)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    version = args.version or f"criticidade_producao_2015_{args.train_through_year}"
    destination = CRITICALITY_MODELS_DIR / "production"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "modelo": "XGBoost",
        "versao_modelo": version,
        "treino_inicio": 2015,
        "treino_fim": int(args.train_through_year),
        "gerado_em": generated_at,
        "horizontes": {},
    }
    base = load_operational_base_from_database()

    for horizon in range(1, 5):
        panel = build_training_panel(base, horizon)
        training = panel.loc[panel["target_epi_year"].le(args.train_through_year)].copy()
        parameters, trees = load_selected_configuration(horizon)
        selector = make_production_preprocessor()
        features = selector.fit_transform(training[FEATURES]).astype(np.float32)
        target = training["categoria_criticidade_id"].to_numpy(dtype=int)
        weights = production_class_weights(target, float(parameters["weight_power"]))
        model = production_model(parameters, trees)
        model.fit(features, target, sample_weight=weights[target], verbose=False)

        model_path = destination / f"xgboost_criticidade_h{horizon}.json"
        preprocessor_path = destination / f"preprocessador_xgboost_criticidade_h{horizon}.joblib"
        model.save_model(model_path)
        joblib.dump(selector, preprocessor_path)
        manifest["horizontes"][str(horizon)] = {
            "amostras_treino": int(len(training)),
            "arvores": trees,
            "parametros": parameters,
            "modelo": model_path.name,
            "preprocessador": preprocessor_path.name,
        }
        print(f"S+{horizon}: {len(training):,} amostras, {trees} árvores")

    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Modelos de produção gravados em {destination}")


if __name__ == "__main__":
    main()
