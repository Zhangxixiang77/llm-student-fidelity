import unittest

from student_fidelity.data import make_synthetic_cdm_dataset
from student_fidelity.repeated_splits import repeated_split_diagnostics


class RepeatedSplitTests(unittest.TestCase):
    def test_repeated_splits_summarize_model_and_focus_item_stability(self):
        dataset = make_synthetic_cdm_dataset(
            n_students=32, n_items=4, n_attributes=2, seed=19
        )
        result = repeated_split_diagnostics(
            dataset,
            n_splits=2,
            seed=7,
            comparison_bootstraps=10,
            itemfit_simulations=1000,
            focus_item_id="item_002",
            focus_pair=("item_001", "item_002"),
        )
        self.assertEqual(result["n_splits"], 2)
        self.assertEqual(result["gdina_max_iter"], 300)
        self.assertEqual(len(result["splits"]), 2)
        self.assertEqual(
            result["model_comparison_summary"]["n_gdina_converged"], 2
        )
        for statistic in ("correlation", "log_odds_ratio"):
            summary = result["focus_item_stability"][statistic]
            self.assertIn("n_splits_with_focus_item_significant", summary)
            self.assertIn("recurrent_significant_pairs", summary)
        self.assertIn("no item is deleted", result["interpretation"])


if __name__ == "__main__":
    unittest.main()
