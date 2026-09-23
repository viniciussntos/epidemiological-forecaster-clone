import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.dashboard_data import DashboardRepository, RISK_LABELS_DISPLAY, common_origins


def predictions() -> pd.DataFrame:
    rows = []
    for horizon in range(1, 5):
        for neighborhood, probabilities in [
            ("A", [0.7, 0.2, 0.08, 0.02]),
            ("B", [0.1, 0.2, 0.3, 0.4]),
        ]:
            rows.append(
                {
                    "modo_resultado": "producao",
                    "epi_year_origem": 2021,
                    "epi_week_origem": 48,
                    "horizonte_semanas": horizon,
                    "bairro_norm": neighborhood,
                    "prob_baixo": probabilities[0],
                    "prob_medio": probabilities[1],
                    "prob_alto": probabilities[2],
                    "prob_critico": probabilities[3],
                    "categoria_prevista": RISK_LABELS_DISPLAY[int(np.argmax(probabilities))],
                    "confianca_modelo": max(probabilities),
                }
            )
    return pd.DataFrame(rows)


class DashboardDataTests(unittest.TestCase):
    def test_database_url_is_mandatory(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "não utiliza arquivos CSV"):
                DashboardRepository()

    def test_probabilities_and_confidence_are_consistent(self):
        frame = predictions()
        probabilities = frame[["prob_baixo", "prob_medio", "prob_alto", "prob_critico"]].to_numpy()
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))
        self.assertTrue(np.allclose(probabilities.max(axis=1), frame["confianca_modelo"]))

    def test_common_origins_have_all_four_horizons(self):
        self.assertEqual(common_origins(predictions()), [(2021, 48)])

    def test_one_row_per_origin_horizon_and_neighborhood(self):
        frame = predictions()
        duplicated = frame.duplicated(
            ["modo_resultado", "epi_year_origem", "epi_week_origem", "horizonte_semanas", "bairro_norm"]
        )
        self.assertFalse(duplicated.any())


if __name__ == "__main__":
    unittest.main()
