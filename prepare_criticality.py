"""Valida no Supabase o painel de treino de criticidade."""

import argparse

from src.config import FORECAST_HORIZON_WEEKS
from src.training_data import load_criticality_training_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=FORECAST_HORIZON_WEEKS, choices=range(1, 5))
    args = parser.parse_args()
    panel = load_criticality_training_panel(args.horizon)
    distribution = panel["categoria_criticidade_alvo"].value_counts().sort_index()
    print(f"S+{args.horizon}: {len(panel):,} observações modeláveis carregadas do Supabase")
    print(distribution.to_string())


if __name__ == "__main__":
    main()
