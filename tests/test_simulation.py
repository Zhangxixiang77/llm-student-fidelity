import json
import tempfile
import unittest
from pathlib import Path

from student_fidelity.baselines import DINABaseline
from student_fidelity.data import make_synthetic_cdm_dataset, train_test_split
from student_fidelity.simulation import (
    DINAMockSimulator,
    attribute_personas,
    run_simulation,
)
from student_fidelity.simulation.simulators import parse_structured_answer


class SimulationTests(unittest.TestCase):
    def test_offline_simulation_writes_every_item(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = make_synthetic_cdm_dataset(
                n_students=20, n_items=6, n_attributes=3, seed=10
            )
            fitting, holdout = train_test_split(dataset, seed=2)
            model = DINABaseline(max_iter=5).fit(fitting)
            alpha = model.sample_patterns(4, seed=3)
            personas = attribute_personas(alpha, dataset.attribute_ids)
            output = Path(directory) / "raw.jsonl"
            responses = run_simulation(
                holdout,
                DINAMockSimulator(model, seed=4),
                personas,
                output,
                state="stateful",
            )
            self.assertEqual(responses.shape, (4, 6))
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(records), 24)
            self.assertEqual({record["item_id"] for record in records}, set(dataset.item_ids))

    def test_answer_parser_accepts_code_fence(self):
        final, work = parse_structured_answer(
            '```json\n{"final_answer":"3/8","work":"brief"}\n```'
        )
        self.assertEqual(final, "3/8")
        self.assertEqual(work, "brief")


if __name__ == "__main__":
    unittest.main()
