"""Componentes compartilhados pelos classificadores de criticidade."""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import fbeta_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from src.config import RANDOM_SEED
from src.criticality_pipeline import CRITICALITY_NUMERIC_FEATURES


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_criticality_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), CRITICALITY_NUMERIC_FEATURES),
            ("bairro", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["bairro_norm"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def class_weights(labels: np.ndarray, power: float = 0.5, classes: int = 4) -> np.ndarray:
    """Pesos inversos suavizados, calculados somente no conjunto de treino."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=classes).astype(float)
    if np.any(counts == 0):
        raise ValueError(f"Treino sem exemplos de alguma categoria: {counts.tolist()}")
    weights = (len(labels) / (classes * counts)) ** power
    return weights / weights.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: np.ndarray, gamma: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = torch.log_softmax(logits, dim=1)
        row = torch.arange(target.shape[0], device=target.device)
        log_pt = log_prob[row, target]
        pt = log_pt.exp()
        return (-self.alpha[target] * (1.0 - pt).pow(self.gamma) * log_pt).mean()


class CriticalityANN(nn.Module):
    def __init__(self, input_size: int, classes: int = 4) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 192),
            nn.ReLU(),
            nn.BatchNorm1d(192),
            nn.Dropout(0.20),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(96, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CriticalityLSTM(nn.Module):
    def __init__(self, sequence_features: int, static_features: int, classes: int = 4) -> None:
        super().__init__()
        self.lstm = nn.LSTM(sequence_features, hidden_size=64, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(64 + static_features, 96),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(96, classes),
        )

    def forward(self, sequence: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(sequence)
        return self.head(torch.cat([hidden[-1], static], dim=1))


def train_classifier_early_stopping(
    model: nn.Module,
    train_loader,
    validation_loader,
    forward_fn,
    loss_fn: nn.Module,
    max_epochs: int = 100,
    patience: int = 15,
) -> tuple[nn.Module, int, list[dict[str, float]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    best_score = -1.0
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            logits, target = forward_fn(model, batch)
            loss = loss_fn(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))

        model.eval()
        validation_losses, actual, predicted = [], [], []
        with torch.no_grad():
            for batch in validation_loader:
                logits, target = forward_fn(model, batch)
                validation_losses.append(float(loss_fn(logits, target)))
                actual.extend(target.cpu().numpy().tolist())
                predicted.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        score = fbeta_score(actual, predicted, beta=2, labels=[0, 1, 2, 3], average="macro", zero_division=0)
        history.append(
            {
                "epoca": epoch,
                "loss_treino": float(np.mean(train_losses)),
                "loss_validacao": float(np.mean(validation_losses)),
                "f2_macro_validacao": float(score),
            }
        )
        if score > best_score + 1e-6:
            best_score = float(score)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return model, best_epoch, history


def fit_classifier_epochs(model, loader, forward_fn, loss_fn, epochs: int) -> nn.Module:
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    for _ in range(max(1, epochs)):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            logits, target = forward_fn(model, batch)
            loss = loss_fn(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


def seeded_generator() -> torch.Generator:
    return torch.Generator().manual_seed(RANDOM_SEED)
