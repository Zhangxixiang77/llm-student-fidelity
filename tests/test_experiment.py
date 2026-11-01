import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from student_fidelity.data import make_synthetic_cdm_dataset
from student_fidelity.experiment import _load_or_compute_split_half_noise, execute_pilot


class ExperimentTests(unittest.TestCase):
    def test_minimal_end_to_end_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = make_synthetic_cdm_dataset(
                n_students=24, n_items=5, n_attributes=2, seed=12
            )
            summary = execute_pilot(
                dataset,
                directory,
                baseline_names=["independent", "dina"],
                n_students=6,
                noise_repeats=2,
                seed=4,
            )
            self.assertEqual(summary["design"]["n_virtual_students"], 6)
            self.assertIn("simulator", summary["results"])
            output_dir = Path(directory)
            self.assertTrue((output_dir / "raw_simulation.jsonl").exists())
            loaded = json.loads((output_dir / "summary.json").read_text())
            self.assertTrue(loaded["warning"].startswith("Pilot output only"))
            self.assertTrue((output_dir / "run_manifest.json").exists())
            with self.assertRaisesRegex(ValueError, "resume configuration differs"):
                execute_pilot(
                    dataset,
                    directory,
                    baseline_names=["independent", "dina"],
                    n_students=5,
                    noise_repeats=2,
                    seed=4,
                    resume=True,
                )

    @patch("student_fidelity.experiment.split_half_noise")
    def test_noise_cache_adopts_completed_run_and_reuses_it(self, compute_noise):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = make_synthetic_cdm_dataset(
                n_students=12, n_items=5, n_attributes=2, seed=12
            )
            first_path = root / "first" / "split_half_noise.csv"
            first_path.parent.mkdir()
            expected = np.array([0.1, 0.2, 0.3])
            pd.DataFrame({"split_half_dependency_rmse": expected}).to_csv(
                first_path, index=False
            )
            cache = root / "shared.npz"
            first = _load_or_compute_split_half_noise(
                dataset, 3, 142, cache, first_path
            )
            second = _load_or_compute_split_half_noise(
                dataset, 3, 142, cache, root / "second" / "split_half_noise.csv"
            )
            np.testing.assert_array_equal(first, expected)
            np.testing.assert_array_equal(second, expected)
            compute_noise.assert_not_called()

    @patch("student_fidelity.experiment.split_half_noise")
    def test_noise_cache_rejects_changed_design(self, compute_noise):
        compute_noise.return_value = np.array([0.1, 0.2])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = make_synthetic_cdm_dataset(
                n_students=12, n_items=5, n_attributes=2, seed=12
            )
            cache = root / "shared.npz"
            _load_or_compute_split_half_noise(
                dataset, 2, 142, cache, root / "split_half_noise.csv"
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                _load_or_compute_split_half_noise(
                    dataset, 3, 142, cache, root / "split_half_noise.csv"
                )
            self.assertEqual(compute_noise.call_count, 1)


if __name__ == "__main__":
    unittest.main()
