import unittest

import numpy as np

from student_fidelity.baselines import DINABaseline
from student_fidelity.data import ResponseDataset, make_synthetic_cdm_dataset
from student_fidelity.item_fit import holm_adjust, item_pair_fit


class ItemFitTests(unittest.TestCase):
    def test_holm_adjustment_matches_step_down_example(self):
        adjusted = holm_adjust(np.array([0.01, 0.04, 0.03]))
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

    def test_item_pair_output_is_reproducible_and_complete(self):
        dataset = make_synthetic_cdm_dataset(
            n_students=80, n_items=4, n_attributes=2, seed=8
        )
        model = DINABaseline().fit(dataset)
        first = item_pair_fit(model, dataset.responses, n_simulated_students=2000, seed=9)
        second = item_pair_fit(model, dataset.responses, n_simulated_students=2000, seed=9)
        self.assertEqual(len(first["item_pairs"]), 6)
        self.assertEqual(len(first["item_level"]), 4)
        self.assertEqual(first["test_level_holm_family_size"], 6)
        self.assertEqual(first["item_level_holm_family_size"], 3)
        self.assertEqual(first["item_pairs"], second["item_pairs"])
        for pair in first["item_pairs"]:
            self.assertGreaterEqual(pair["correlation_holm_test_p"], pair["correlation_p"])
            self.assertGreaterEqual(
                pair["log_odds_ratio_holm_test_p"], pair["log_odds_ratio_p"]
            )

    def test_zero_cells_are_flagged_and_corrected_only_for_log_or(self):
        responses = np.array(
            [
                [1, 1, 1],
                [1, 1, 0],
                [1, 1, 1],
                [1, 1, 0],
                [0, 1, 1],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0],
            ],
            dtype=np.int8,
        )
        dataset = ResponseDataset(
            responses=responses,
            q_matrix=np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8),
        )
        model = DINABaseline().fit(dataset)
        result = item_pair_fit(model, responses, n_simulated_students=2000, seed=4)
        pair = result["item_pairs"][0]
        self.assertEqual(pair["observed_cells_n11_n10_n01_n00"], [4, 0, 2, 2])
        self.assertEqual(pair["observed_zero_cells_corrected"], [1])
        expected_log_or = np.log(4 * 2 / (0.5 * 2))
        self.assertAlmostEqual(pair["observed_log_odds_ratio"], expected_log_or)
        self.assertGreaterEqual(result["n_pairs_with_zero_cell_correction"], 1)


if __name__ == "__main__":
    unittest.main()
