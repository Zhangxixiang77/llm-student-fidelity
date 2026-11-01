import json
import unittest

import numpy as np

from student_fidelity.baselines import DINABaseline, GDINABaseline
from student_fidelity.data import make_synthetic_cdm_dataset
from student_fidelity.parameter_recovery import (
    dina_item_recovery,
    recovery_replication,
    simulate_dataset,
    summarize_recovery,
)


class ParameterRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.dataset = make_synthetic_cdm_dataset(
            n_students=60, n_items=6, n_attributes=3, seed=91
        )

    def test_simulation_retains_q_and_is_reproducible(self):
        generating = DINABaseline(max_iter=30, n_starts=2, seed=4).fit(self.dataset)
        first = simulate_dataset(generating, self.dataset, 30, seed=18)
        second = simulate_dataset(generating, self.dataset, 30, seed=18)
        np.testing.assert_array_equal(first.responses, second.responses)
        np.testing.assert_array_equal(first.q_matrix, self.dataset.q_matrix)
        self.assertEqual(first.responses.shape, (30, 6))

    def test_dina_recovery_and_summary_are_serializable(self):
        generating = DINABaseline(max_iter=40, n_starts=2, seed=4).fit(self.dataset)
        record = recovery_replication(
            generating,
            self.dataset,
            "dina",
            40,
            1,
            seed=81,
            dina_starts=2,
            dina_max_iter=40,
            itemfit_simulations=2000,
        )
        summary = summarize_recovery([record])
        rows = dina_item_recovery([record], generating, self.dataset.item_ids)
        self.assertEqual(len(rows), 6)
        self.assertIn("guess_rmse", record)
        rejection = summary["rejection_rates"]["itemfit_cor_holm_test_any_rejection"]
        self.assertIn("monte_carlo_ci95_wilson", rejection)
        if not record["itemfit_cor_holm_test_any_rejection"]:
            self.assertGreater(rejection["monte_carlo_ci95_wilson"][1], 0)
        json.dumps(summary, allow_nan=False)

    def test_gdina_recovers_pattern_probabilities_without_comparing_betas(self):
        generating = GDINABaseline(max_iter=40).fit(self.dataset)
        record = recovery_replication(
            generating,
            self.dataset,
            "gdina",
            40,
            1,
            seed=82,
            gdina_max_iter=40,
        )
        self.assertIn("p_correct_rmse", record)
        self.assertIn("p_correct_prior_weighted_rmse", record)
        self.assertNotIn("betas", record)
        json.dumps(summarize_recovery([record]), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
