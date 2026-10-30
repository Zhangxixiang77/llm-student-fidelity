import json
import tempfile
import unittest
from pathlib import Path

from student_fidelity.data import make_synthetic_cdm_dataset
from student_fidelity.experiment import execute_pilot


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


if __name__ == "__main__":
    unittest.main()
