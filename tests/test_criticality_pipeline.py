import unittest

import numpy as np
import pandas as pd

from src.criticality_pipeline import CRITICALITY_NUMERIC_FEATURES, build_criticality_panel, category_from_incidence
from src.temporal_validation import progressive_masks


def sample_base(weeks: int = 16) -> pd.DataFrame:
    rows = []
    for neighborhood, population, multiplier in [("A", 10_000, 1), ("B", 20_000, 2)]:
        for index in range(weeks):
            rows.append(
                {
                    "bairro_norm": neighborhood,
                    "populacao": population,
                    "epi_year": 2021,
                    "epi_week": index + 1,
                    "time_index": index,
                    "target_epi_year": 2021,
                    "target_epi_week": index + 2,
                    "target_time_index": index + 1,
                    "casos_totais": index % 3 * multiplier,
                    "precipitacao_total": 10.0 + index,
                    "temp_max_media": 30.0 + index / 10,
                }
            )
    return pd.DataFrame(rows)


class CriticalityPipelineTests(unittest.TestCase):
    def test_four_week_thresholds(self):
        labels = category_from_incidence(np.array([0, 99.99, 100, 299.99, 300, 499.99, 500]))
        self.assertEqual(labels.tolist(), ["Baixo", "Baixo", "Medio", "Medio", "Alto", "Alto", "Critico"])

    def test_target_is_future_four_week_sum(self):
        base = sample_base()
        panel = build_criticality_panel(base, horizon=1)
        row = panel.loc[panel["bairro_norm"].eq("A")].iloc[10]
        expected = base.loc[base["bairro_norm"].eq("A")].iloc[8:12]["casos_totais"].sum()
        self.assertEqual(float(row["casos_acumulados_4s_alvo"]), float(expected))
        self.assertEqual(int(row["target_time_index"]), int(row["time_index"] + 1))

    def test_features_are_complete_after_lookback(self):
        panel = build_criticality_panel(sample_base(), horizon=1)
        modelable = panel.dropna(subset=CRITICALITY_NUMERIC_FEATURES + ["categoria_criticidade_id"])
        self.assertFalse(modelable.empty)
        self.assertFalse(modelable[CRITICALITY_NUMERIC_FEATURES].isna().any().any())

    def test_progressive_validation_never_uses_future_years(self):
        frame = pd.DataFrame({"target_epi_year": np.repeat(np.arange(2015, 2021), 2)})
        folds = list(progressive_masks(frame))
        self.assertEqual([year for year, _, _ in folds], [2016, 2017, 2018, 2019, 2020])
        years = frame["target_epi_year"].to_numpy()
        for validation_year, train_mask, validation_mask in folds:
            self.assertTrue((years[train_mask] < validation_year).all())
            self.assertTrue((years[validation_mask] == validation_year).all())


if __name__ == "__main__":
    unittest.main()
