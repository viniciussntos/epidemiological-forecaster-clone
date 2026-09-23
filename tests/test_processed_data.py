import inspect
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR, PROJECT_ROOT
from src.data_pipeline import PipelinePaths, _build_weekly_panel


class DataPipelineTests(unittest.TestCase):
    def test_raw_sources_are_explicit(self):
        signature = inspect.signature(PipelinePaths)
        for name in ["dengue", "climate", "population"]:
            self.assertIs(signature.parameters[name].default, inspect.Parameter.empty)
        with self.assertRaises(TypeError):
            PipelinePaths()

    def test_weekly_panel_fills_absent_cases_with_zero(self):
        population = pd.DataFrame({"bairro_norm": ["A", "B"], "populacao": [10_000, 20_000]})
        climate = pd.DataFrame(
            {
                "epi_year": [2021, 2021, 2021],
                "epi_week": [1, 2, 3],
                "time_index": [0, 1, 2],
                "precipitacao_total": [10.0, 20.0, 30.0],
                "temp_max_media": [30.0, 31.0, 32.0],
                "week_sin": np.sin(2 * np.pi * np.array([1, 2, 3]) / 53.0),
                "week_cos": np.cos(2 * np.pi * np.array([1, 2, 3]) / 53.0),
            }
        )
        for lag in range(1, 5):
            climate[f"chuva_lag{lag}"] = climate["precipitacao_total"].shift(lag)
            climate[f"temp_max_lag{lag}"] = climate["temp_max_media"].shift(lag)
        dengue = pd.DataFrame({"bairro_norm": ["A"], "epi_year": [2021], "epi_week": [1]})
        panel = _build_weekly_panel(dengue, climate, population, horizon=1)
        self.assertEqual(len(panel), 6)
        self.assertEqual(int(panel["casos_totais"].sum()), 1)
        self.assertTrue(panel.loc[panel["bairro_norm"].eq("B"), "casos_totais"].eq(0).all())

    def test_output_directory_is_scoped_to_project_by_default(self):
        paths = PipelinePaths(Path("dengue.csv"), Path("climate.csv"), Path("population.csv"))
        self.assertEqual(paths.output_dir, DATA_DIR)
        self.assertTrue(paths.output_dir.is_relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    unittest.main()
