"""RNA multiclasse com selecao de epocas por validacao temporal progressiva."""

import argparse

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    CRITICALITY_MODELS_DIR, CRITICALITY_RESULTS_DIR, FORECAST_HORIZON_WEEKS,
    RANDOM_SEED, TEST_YEAR, TRAIN_END_YEAR,
)
from src.criticality_evaluation import evaluate_criticality, save_criticality_evaluation
from src.criticality_modeling import (
    CriticalityANN, FocalLoss, class_weights, fit_classifier_epochs,
    make_criticality_preprocessor, seeded_generator, set_seed, train_classifier_early_stopping,
)
from src.criticality_pipeline import CRITICALITY_NUMERIC_FEATURES
from src.temporal_validation import (
    fold_metric_row, pooled_validation_metrics, progressive_masks, robust_epoch_choice,
)
from src.training_data import load_criticality_training_panel


FEATURES = CRITICALITY_NUMERIC_FEATURES + ["bairro_norm"]
BATCH_SIZE = 512


def loader(x, y, shuffle):
    data = TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.int64)))
    return DataLoader(data, batch_size=BATCH_SIZE, shuffle=shuffle, generator=seeded_generator())


def forward(model, batch):
    x, target = batch
    return model(x), target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=FORECAST_HORIZON_WEEKS, choices=range(1, 5))
    args = parser.parse_args()
    panel = load_criticality_training_panel(args.horizon)
    train_full = panel.loc[panel["target_epi_year"].le(TRAIN_END_YEAR)].copy()
    test = panel.loc[panel["target_epi_year"].eq(TEST_YEAR)].copy()

    fold_rows, history_rows, actual_parts, probability_parts, best_epochs = [], [], [], [], []
    for year, train_mask, validation_mask in progressive_masks(train_full):
        set_seed(RANDOM_SEED + year)
        fold_train = train_full.loc[train_mask]
        fold_validation = train_full.loc[validation_mask]
        selector = make_criticality_preprocessor()
        x_train = selector.fit_transform(fold_train[FEATURES]).astype(np.float32)
        x_validation = selector.transform(fold_validation[FEATURES]).astype(np.float32)
        y_train = fold_train["categoria_criticidade_id"].to_numpy(dtype=int)
        y_validation = fold_validation["categoria_criticidade_id"].to_numpy(dtype=int)
        model = CriticalityANN(x_train.shape[1])
        model, best_epoch, history = train_classifier_early_stopping(
            model, loader(x_train, y_train, True), loader(x_validation, y_validation, False),
            forward, FocalLoss(class_weights(y_train, 0.5)), max_epochs=60, patience=10,
        )
        model.eval()
        with torch.no_grad():
            probabilities = torch.softmax(model(torch.from_numpy(x_validation)), dim=1).numpy()
        row = fold_metric_row(year, y_validation, probabilities)
        row["epoca_selecionada"] = best_epoch
        fold_rows.append(row)
        history_rows.extend({"ano_validacao": year, **item} for item in history)
        actual_parts.append(y_validation)
        probability_parts.append(probabilities)
        best_epochs.append(best_epoch)
        print(f"Validacao {year}: epoca={best_epoch}, F2 macro={row['f2_macro']:.4f}", flush=True)

    pooled = pooled_validation_metrics(actual_parts, probability_parts)
    final_epochs = robust_epoch_choice(best_epochs)
    set_seed(RANDOM_SEED)
    final_selector = make_criticality_preprocessor()
    x_train = final_selector.fit_transform(train_full[FEATURES]).astype(np.float32)
    x_test = final_selector.transform(test[FEATURES]).astype(np.float32)
    y_train = train_full["categoria_criticidade_id"].to_numpy(dtype=int)
    final_model = CriticalityANN(x_train.shape[1])
    final_model = fit_classifier_epochs(
        final_model, loader(x_train, y_train, True), forward,
        FocalLoss(class_weights(y_train, 0.5)), final_epochs,
    )
    final_model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(final_model(torch.from_numpy(x_test)), dim=1).numpy()
    parts = evaluate_criticality("RNA", test, probabilities, args.horizon)
    parts[1]["epocas_selecionadas"] = final_epochs
    parts[1]["potencia_peso_classes"] = 0.5
    parts[1]["validacao_oof_f2_macro"] = pooled["f2_macro"]
    parts[1]["validacao_oof_acuracia_balanceada"] = pooled["acuracia_balanceada"]
    save_criticality_evaluation(parts, "rna", args.horizon, CRITICALITY_RESULTS_DIR)

    CRITICALITY_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CRITICALITY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": final_model.state_dict(), "input_size": x_train.shape[1]}, CRITICALITY_MODELS_DIR / f"rna_criticidade_h{args.horizon}.pt")
    joblib.dump(final_selector, CRITICALITY_MODELS_DIR / f"preprocessador_rna_criticidade_h{args.horizon}.joblib")
    pd.DataFrame(fold_rows).to_csv(CRITICALITY_RESULTS_DIR / f"validacao_progressiva_rna_h{args.horizon}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history_rows).to_csv(CRITICALITY_RESULTS_DIR / f"historico_validacao_rna_h{args.horizon}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([pooled]).to_csv(CRITICALITY_RESULTS_DIR / f"metricas_validacao_oof_rna_h{args.horizon}.csv", index=False, encoding="utf-8-sig")
    print(parts[1].to_string(index=False))


if __name__ == "__main__":
    main()
