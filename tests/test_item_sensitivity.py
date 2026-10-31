import json
import unittest

import numpy as np

from student_fidelity.data import make_synthetic_cdm_dataset
from student_fidelity.item_sensitivity import exclude_items, item_exclusion_sensitivity


class ItemSensitivityTests(unittest.TestCase):
    def test_exclusion_preserves_student_alignment(self):
        dataset = make_synthetic_cdm_dataset(
            n_students=24, n_items=4, n_attributes=2, seed=13
        )
        reduced = exclude_items(dataset, ["item_002"])
        self.assertEqual(reduced.student_ids, dataset.student_ids)
        self.assertEqual(reduced.item_ids, ["item_001", "item_003", "item_004"])
        np.testing.assert_array_equal(
            reduced.responses, dataset.responses[:, [0, 2, 3]]
        )

    def test_sensitivity_retains_full_item_set_as_primary(self):
        dataset = make_synthetic_cdm_dataset(
            n_students=24, n_items=4, n_attributes=2, seed=13
        )
        result = item_exclusion_sensitivity(
            dataset,
            excluded_item_ids=["item_002"],
            n_splits=1,
            itemfit_simulations=1000,
        )
        self.assertEqual(result["primary_analysis"], "all_items_retained")
        self.assertEqual(result["n_full_items"], 4)
        self.assertEqual(result["n_reduced_items"], 3)
        self.assertEqual(result["summary"]["n_full_converged"], 1)
        self.assertEqual(result["summary"]["n_reduced_converged"], 1)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
