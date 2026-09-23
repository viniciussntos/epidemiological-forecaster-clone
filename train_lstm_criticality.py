"""LSTM multiclasse com selecao de epocas por validacao temporal progressiva."""

import argparse

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    CRITICALITY_LOOKBACK_WEEKS, CRITICALITY_MODELS_DIR, CRITICALITY_RESULTS_DIR,
    FORECAST_HORIZON_WEEKS, RANDOM_SEED, TEST_YEAR, TRAIN_END_YEAR,
)
from src.criticality_evaluation import evaluate_criticality, save_criticality_evaluation
from src.criticality_modeling import (
    CriticalityLSTM, FocalLoss, class_weights, fit_classifier_epochs,
    seeded_generator, set_seed, train_classifier_early_stopping,
)
from src.criticality_pipeline import CRITICALITY_STATIC_NUMERIC_FEATURES
from src.temporal_validation import (
    fold_metric_row, pooled_validation_metrics, progressive_masks, robust_epoch_choice,
)
from src.training_data import load_criticality_training_panel


BATCH_SIZE = 512
SEQUENCE_VARIABLES = ["casos", "chuva", "temp", "recife_casos"]


def build_sequences(panel):
    blocks = []
    for prefix in SEQUENCE_VARIABLES:
        columns = [f"{prefix}_obs{lag}" for lag in range(CRITICALITY_LOOKBACK_WEEKS, 0, -1)]
        blocks.append(panel[columns].to_numpy(dtype=np.float32)[..., None])
    return np.concatenate(blocks, axis=2)


def loader(sequence, static, y, shuffle):
    data = TensorDataset(
        torch.from_numpy(sequence.astype(np.float32)), torch.from_numpy(static.astype(np.float32)),
        torch.from_numpy(y.astype(np.int64)),
    )
    return DataLoader(data, batch_size=BATCH_SIZE, shuffle=shuffle, generator=seeded_generator())


def forward(model, batch):
    sequence, static, target = batch
    return model(sequence, static), target


def scaled_inputs(panel, raw_sequences, train_mask):
    sequence_scaler = StandardScaler().fit(raw_sequences[train_mask].reshape(-1, len(SEQUENCE_VARIABLES)))
    sequences = sequence_scaler.transform(raw_sequences.reshape(-1, len(SEQUENCE_VARIABLES))).reshape(raw_sequences.shape)
    static_scaler = StandardScaler().fit(panel.loc[train_mask, CRITICALITY_STATIC_NUMERIC_FEATURES])
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(panel.loc[train_mask, ["bairro_norm"]])
    static = np.hstack([
        static_scaler.transform(panel[CRITICALITY_STATIC_NUMERIC_FEATURES]),
        encoder.transform(panel[["bairro_norm"]]),
    ]).astype(np.float32)
    return sequences, static, sequence_scaler, static_scaler, encoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=FORECAST_HORIZON_WEEKS, choices=range(1, 5))
    args = parser.parse_args()
    panel = load_criticality_training_panel(args.horizon)
    train_full = panel.loc[panel["target_epi_year"].le(TRAIN_END_YEAR)].copy().reset_index(drop=True)
    raw_train_sequences = build_sequences(train_full)
    y_training_period = train_full["categoria_criticidade_id"].to_numpy(dtype=int)

    fold_rows, history_rows, actual_parts, probability_parts, best_epochs = [], [], [], [], []
    for year, train_mask, validation_mask in progressive_masks(train_full):
        set_seed(RANDOM_SEED + year)
        sequences, static, _, _, _ = scaled_inputs(train_full, raw_train_sequences, train_mask)
        model = CriticalityLSTM(len(SEQUENCE_VARIABLES), static.shape[1])
        model, best_epoch, history = train_classifier_early_stopping(
            model,
            loader(sequences[train_mask], static[train_mask], y_training_period[train_mask], True),
            loader(sequences[validation_mask], static[validation_mask], y_training_period[validation_mask], False),
            forward, FocalLoss(class_weights(y_training_period[train_mask], 0.5)),
            max_epochs=60, patience=10,
        )
        model.eval()
        with torch.no_grad():
            probabilities = torch.softmax(
                model(torch.from_numpy(sequences[validation_mask]), torch.from_numpy(static[validation_mask])), dim=1
            ).numpy()
        actual = y_training_period[validation_mask]
        row = fold_metric_row(year, actual, probabilities)
        row["epoca_selecionada"] = best_epoch
        fold_rows.append(row)
        history_rows.extend({"ano_validacao": year, **item} for item in history)
        actual_parts.append(actual)
        probability_parts.append(probabilities)
        best_epochs.append(best_epoch)
        print(f"Validacao {year}: epoca={best_epoch}, F2 macro={row['f2_macro']:.4f}", flush=True)

    pooled = pooled_validation_metrics(actual_parts, probability_parts)
    final_epochs = robust_epoch_choice(best_epochs)
    all_panel = panel.reset_index(drop=True)
    raw_all_sequences = build_sequences(all_panel)
    train_mask = all_panel["target_epi_year"].le(TRAIN_END_YEAR).to_numpy()
    test_mask = all_panel["target_epi_year"].eq(TEST_YEAR).to_numpy()
    y_all = all_panel["categoria_criticidade_id"].to_numpy(dtype=int)
    final_sequences, final_static, sequence_scaler, static_scaler, encoder = scaled_inputs(
        all_panel, raw_all_sequences, train_mask
    )
    set_seed(RANDOM_SEED)
    final_model = CriticalityLSTM(len(SEQUENCE_VARIABLES), final_static.shape[1])
    final_model = fit_classifier_epochs(
        final_model,
        loader(final_sequences[train_mask], final_static[train_mask], y_all[train_mask], True),
        forward, FocalLoss(class_weights(y_all[train_mask], 0.5)), final_epochs,
    )
    final_model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(
            final_model(torch.from_numpy(final_sequences[test_mask]), torch.from_numpy(final_static[test_mask])), dim=1
        ).numpy()
    test = all_panel.loc[test_mask].copy()
    parts = evaluate_criticality("LSTM", test, probabilities, args.horizon)
    parts[1]["epocas_selecionadas"] = final_epochs
    parts[1]["potencia_peso_classes"] = 0.5
    parts[1]["validacao_oof_f2_macro"] = pooled["f2_macro"]
    parts[1]["validacao_oof_acuracia_balanceada"] = pooled["acuracia_balanceada"]
    save_criticality_evaluation(parts, "lstm", args.horizon, CRITICALITY_RESULTS_DIR)

    CRITICALITY_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CRITICALITY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": final_model.state_dict(), "sequence_features": len(SEQUENCE_VARIABLES),
        "static_features": final_static.shape[1],
    }, CRITICALITY_MODELS_DIR / f"lstm_criticidade_h{args.horizon}.pt")
    joblib.dump(sequence_scaler, CRITICALITY_MODELS_DIR / f"escalonador_sequencia_lstm_criticidade_h{args.horizon}.joblib")
    joblib.dump(static_scaler, CRITICALITY_MODELS_DIR / f"escalonador_estatico_lstm_criticidade_h{args.horizon}.joblib")
    joblib.dump(encoder, CRITICALITY_MODELS_DIR / f"codificador_bairro_lstm_criticidade_h{args.horizon}.joblib")
    pd.DataFrame(fold_rows).to_csv(CRITICALITY_RESULTS_DIR / f"validacao_progressiva_lstm_h{args.horizon}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history_rows).to_csv(CRITICALITY_RESULTS_DIR / f"historico_validacao_lstm_h{args.horizon}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([pooled]).to_csv(CRITICALITY_RESULTS_DIR / f"metricas_validacao_oof_lstm_h{args.horizon}.csv", index=False, encoding="utf-8-sig")
    print(parts[1].to_string(index=False))


if __name__ == "__main__":
    main()
