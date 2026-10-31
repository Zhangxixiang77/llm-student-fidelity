import unittest

import numpy as np

from student_fidelity.data import make_synthetic_cdm_dataset, train_test_split
from student_fidelity.q_sensitivity import merge_attribute_groups, q_matrix_sensitivity


class QSensitivityTests(unittest.TestCase):
    def test_merge_preserves_alignment_and_or_combines_attributes(self):
        dataset = make_synthetic_cdm_dataset(
            n_students=30, n_items=5, n_attributes=2, seed=3
        )
        dataset.attribute_ids = ["first", "second"]
        merged = merge_attribute_groups(
            dataset,
            [{"attributes": ["first", "second"], "new_attribute_id": "combined"}],
        )
        self.assertEqual(merged.attribute_ids, ["combined"])
        self.assertEqual(merged.item_ids, dataset.item_ids)
        self.assertEqual(merged.student_ids, dataset.student_ids)
        np.testing.assert_array_equal(merged.responses, dataset.responses)
        np.testing.assert_array_equal(merged.q_matrix[:, 0], dataset.q_matrix.max(axis=1))

    def test_same_split_sensitivity_does_not_select_a_q_matrix(self):
        dataset = make_synthetic_cdm_dataset(
            n_students=40, n_items=5, n_attributes=2, seed=5
        )
        dataset.attribute_ids = ["first", "second"]
        fitting, holdout = train_test_split(dataset, seed=11)
        specs = [
            {"name": "original_2", "merges": []},
            {
                "name": "combined_1",
                "merges": [
                    {"attributes": ["first", "second"], "new_attribute_id": "combined"}
                ],
            },
        ]
        result = q_matrix_sensitivity(fitting, holdout, n_boot=20, seed=7, specs=specs)
        self.assertEqual(result["primary_q_matrix"], "original_2")
        self.assertEqual(result["status"], "pre_specified_sensitivity_not_model_selection")
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(result["candidates"][0]["n_attributes"], 2)
        self.assertEqual(
            result["candidates"][0]["paired_student_bootstrap_95_ci"], [0.0, 0.0]
        )
        self.assertEqual(result["candidates"][1]["n_attributes"], 1)
        self.assertEqual(result["candidates"][1]["n_bootstrap"], 20)
        self.assertNotIn("selected_q_matrix", result)


if __name__ == "__main__":
    unittest.main()
