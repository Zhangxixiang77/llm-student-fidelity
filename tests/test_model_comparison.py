import unittest

from student_fidelity.baselines import DINABaseline, GDINABaseline
from student_fidelity.data import make_synthetic_cdm_dataset, train_test_split
from student_fidelity.model_comparison import compare_cdm_fits


class ModelComparisonTests(unittest.TestCase):
    def test_same_split_comparison_and_bootstrap(self):
        dataset = make_synthetic_cdm_dataset(
            n_students=40, n_items=5, n_attributes=2, seed=12
        )
        fitting, holdout = train_test_split(dataset, seed=4)
        dina = DINABaseline(max_iter=10).fit(fitting)
        gdina = GDINABaseline(max_iter=8).fit(fitting)
        result = compare_cdm_fits(dina, gdina, fitting, holdout, n_boot=20, seed=7)
        self.assertEqual(result["n_bootstrap"], 20)
        self.assertEqual(len(result["paired_student_bootstrap_95_ci"]), 2)
        self.assertIn(result["provisional_dimension_b_model"], {"dina", "gdina"})
        self.assertIn("holdout_srmsr", result["dina"])
        self.assertIn("convergence", result["gdina"])
        self.assertEqual(result["dina"]["nominal_n_parameters"], 13)
        self.assertGreaterEqual(result["gdina"]["nominal_n_parameters"], 13)


if __name__ == "__main__":
    unittest.main()
