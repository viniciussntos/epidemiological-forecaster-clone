"""Gera as bases processadas e todas as auditorias."""

import argparse
from pathlib import Path

from src.config import DATA_DIR, FORECAST_HORIZON_WEEKS
from src.data_pipeline import PipelinePaths, prepare_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=FORECAST_HORIZON_WEEKS, choices=range(1, 5))
    parser.add_argument("--dengue", type=Path, required=True, help="CSV bruto/consolidado de dengue.")
    parser.add_argument("--climate", type=Path, required=True, help="CSV semanal do INMET.")
    parser.add_argument("--population", type=Path, required=True, help="CSV populacional do IBGE.")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR, help="Diretório para os artefatos processados.")
    args = parser.parse_args()
    missing = [path for path in [args.dengue, args.climate, args.population] if not path.is_file()]
    if missing:
        parser.error("Arquivos inexistentes: " + ", ".join(str(path) for path in missing))
    outputs = prepare_all(
        PipelinePaths(
            dengue=args.dengue,
            climate=args.climate,
            population=args.population,
            output_dir=args.output_dir,
        ),
        horizon=args.horizon,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
