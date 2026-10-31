import json
import tempfile
import unittest
from pathlib import Path

from student_fidelity.audit import write_audit
from student_fidelity.data import make_synthetic_cdm_dataset
from student_fidelity.experiment import execute_pilot


class AuditTests(unittest.TestCase):
    def test_offline_audit_includes_fit_and_length_sensitivity(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = make_synthetic_cdm_dataset(
                n_students=24, n_items=5, n_attributes=2, seed=12
            )
            execute_pilot(
                dataset,
                directory,
                baseline_names=["independent", "dina"],
                n_students=6,
                noise_repeats=2,
                seed=4,
            )
            run_dir = Path(directory)
            raw_path = run_dir / "raw_simulation.jsonl"
            records = [json.loads(line) for line in raw_path.read_text().splitlines()]
            records[0]["response_metadata"] = {"finish_reason": "length"}
            raw_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            output = write_audit(
                dataset, run_dir, ceiling_repeats=2, comparison_bootstraps=20
            )
            audit = json.loads(output.read_text())
            self.assertEqual(audit["response_audit"]["scored_records"], 30)
            self.assertEqual(audit["length_sensitivity"]["n_retained_students"], 5)
            self.assertEqual(audit["readback_ceiling"]["n_repeats"], 2)
            self.assertIn("holdout_srmsr", audit["dina_fit_diagnostics"])
            self.assertEqual(audit["cdm_fit_comparison"]["n_bootstrap"], 20)
            self.assertEqual(audit["gdina_readback_ceiling"]["n_repeats"], 2)
            self.assertIn("gdina_attribute_readback_agreement", audit["primary_all_scored"])
            self.assertIn("gdina_mean_nonmaster_excess", audit["primary_all_scored"])
            self.assertIn("descriptive_gate_references", audit)


if __name__ == "__main__":
    unittest.main()
