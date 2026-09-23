import unittest

import numpy as np
import pandas as pd

from src.data_pipeline import _epi_week_from_date, normalize_text
from src.criticality_pipeline import category_from_incidence


class PipelineUnitTests(unittest.TestCase):
    def test_normalize_text(self):
        self.assertEqual(normalize_text("  Sítio  dos Pintos "), "sitio dos pintos")

    def test_epidemiological_week_is_sunday_to_saturday(self):
        self.assertEqual(_epi_week_from_date(pd.Timestamp("2021-01-03")), (2021, 1))
        self.assertEqual(_epi_week_from_date(pd.Timestamp("2021-01-09")), (2021, 1))
        self.assertEqual(_epi_week_from_date(pd.Timestamp("2021-01-10")), (2021, 2))

    def test_risk_thresholds(self):
        result = category_from_incidence(np.array([0, 99.99, 100, 299.99, 300, 499.99, 500]))
        self.assertEqual(
            result.tolist(),
            ["Baixo", "Baixo", "Medio", "Medio", "Alto", "Alto", "Critico"],
        )


if __name__ == "__main__":
    unittest.main()
