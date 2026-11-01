import json
import tempfile
import unittest
from pathlib import Path

import yaml

from student_fidelity.audit import (
    FRCSUB_ELIGIBLE_READBACK_ATTRIBUTES,
    write_audit,
)
from student_fidelity.data import make_synthetic_cdm_dataset
from student_fidelity.experiment import execute_pilot


class AuditTests(unittest.TestCase):
    def test_offline_audit_includes_fit_and_length_sensitivity(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = make_synthetic_cdm_dataset(
                n_students=24, n_items=5, n_attributes=2, seed=12
            )
            dataset.attribute_ids = [
                "separate_whole_from_fraction",
                "borrow_from_whole_part",
            ]
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
            records.append(
                {
                    "student_id": "virtual_0002",
                    "error": "unparsed truncated response",
                    "response_metadata": {"finish_reason": "length"},
                }
            )
            raw_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            output = write_audit(
                dataset,
                run_dir,
                ceiling_repeats=2,
                comparison_bootstraps=20,
                itemfit_simulations=1000,
                dimension_b_model="dina",
            )
            audit = json.loads(output.read_text())
            self.assertEqual(audit["response_audit"]["scored_records"], 30)
            self.assertEqual(audit["response_audit"]["attempt_records"], 31)
            self.assertEqual(audit["length_sensitivity"]["n_retained_students"], 4)
            self.assertEqual(audit["readback_ceiling"]["n_repeats"], 2)
            self.assertIn("holdout_srmsr", audit["dina_fit_diagnostics"])
            self.assertEqual(audit["cdm_fit_comparison"]["n_bootstrap"], 20)
            self.assertEqual(
                audit["cdm_fit_comparison"]["dina"]["convergence"]["n_starts"], 10
            )
            self.assertEqual(audit["gdina_readback_ceiling"]["n_repeats"], 2)
            self.assertIn("gdina_attribute_readback_agreement", audit["primary_all_scored"])
            self.assertIn("gdina_mean_nonmaster_excess", audit["primary_all_scored"])
            self.assertIn("dina_mean_nonmaster_excess", audit["primary_all_scored"])
            self.assertEqual(len(audit["item_pair_fit"]["gdina"]["item_pairs"]), 10)
            self.assertEqual(audit["item_pair_fit"]["gdina"]["n_simulated_students"], 1000)
            self.assertIsNone(audit["q_matrix_sensitivity"])
            self.assertIn(
                "gdina_eligible_attribute_readback_agreement",
                audit["primary_all_scored"],
            )
            self.assertIn(
                "dina_eligible_attribute_readback_agreement",
                audit["primary_all_scored"],
            )
            gates = audit["descriptive_gate_references"]
            self.assertEqual(gates["dimension_b_model"], "dina")
            self.assertEqual(gates["dimension_b_model_source"], "preregistered_argument")
            self.assertEqual(gates["gate_3_metric"], "dina_mean_nonmaster_excess")
            self.assertEqual(gates["gate_3_threshold_max_0_30"], 0.30)
            self.assertFalse(gates["gate_2_pattern_readback_enabled"])

    def test_locked_gate_configs_match_audit_constants(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        historical = yaml.safe_load(
            (config_dir / "pilot_gates.frcsub.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(historical["dimension_b"]["model"], "gdina")
        self.assertEqual(historical["gate_3"]["metric"], "gdina_mean_nonmaster_excess")

        config = yaml.safe_load(
            (config_dir / "pilot_gates.frcsub.dina.v2.yaml").read_text(encoding="utf-8")
        )
        dimension_b = config["dimension_b"]
        self.assertEqual(config["primary_q_matrix"], "original_8")
        self.assertEqual(config["supersedes"], "configs/pilot_gates.frcsub.yaml")
        self.assertEqual(dimension_b["model"], "dina")
        self.assertEqual(dimension_b["estimator"]["dina_n_starts"], 30)
        self.assertEqual(dimension_b["estimator"]["dina_max_iter"], 300)
        self.assertFalse(dimension_b["pattern_readback_enabled"])
        self.assertEqual(
            tuple(dimension_b["attribute_readback"]["eligible_attributes"]),
            FRCSUB_ELIGIBLE_READBACK_ATTRIBUTES,
        )
        self.assertEqual(config["gate_3"]["metric"], "dina_mean_nonmaster_excess")
        self.assertEqual(config["gate_3"]["threshold_max"], 0.30)
        self.assertEqual(config["gate_3"]["item_scope"], "all_20_items")
        self.assertEqual(config["gate_3"]["current_status"], "locked_not_run")


if __name__ == "__main__":
    unittest.main()
