import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from student_fidelity.baselines import DINABaseline
from student_fidelity.data import make_synthetic_cdm_dataset, train_test_split
from student_fidelity.simulation import (
    DINAMockSimulator,
    attribute_personas,
    run_simulation,
)
from student_fidelity.simulation.personas import Persona
from student_fidelity.simulation.runner import (
    answers_equivalent,
    latin_square_item_order,
    parse_rational_answer,
)
from student_fidelity.simulation.simulators import (
    Item,
    OpenAICompatibleSimulator,
    SimulationAnswer,
    parse_structured_answer,
)


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

            class FailIfCalled:
                def answer(self, *args):
                    raise AssertionError("completed responses should not be requested again")

            resumed = run_simulation(
                holdout, FailIfCalled(), personas, output, state="stateful", resume=True
            )
            np.testing.assert_array_equal(resumed, responses)
            self.assertEqual(len(output.read_text().splitlines()), 24)

    def test_call_budget_is_cumulative_across_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = make_synthetic_cdm_dataset(
                n_students=20, n_items=3, n_attributes=2, seed=10
            )
            fitting, holdout = train_test_split(dataset, seed=2)
            model = DINABaseline(max_iter=5).fit(fitting)
            personas = attribute_personas(model.sample_patterns(2, seed=3), dataset.attribute_ids)
            output = Path(directory) / "raw.jsonl"
            with self.assertRaisesRegex(RuntimeError, "call budget exhausted"):
                run_simulation(
                    holdout, DINAMockSimulator(model), personas, output, max_calls=1
                )
            self.assertEqual(len(output.read_text().splitlines()), 1)
            with self.assertRaisesRegex(RuntimeError, "call budget exhausted"):
                run_simulation(
                    holdout,
                    DINAMockSimulator(model),
                    personas,
                    output,
                    resume=True,
                    max_calls=1,
                )
            self.assertEqual(len(output.read_text().splitlines()), 1)

    def test_transport_failure_is_logged_and_counts_toward_resume_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = make_synthetic_cdm_dataset(
                n_students=20, n_items=3, n_attributes=2, seed=10
            )
            fitting, holdout = train_test_split(dataset, seed=2)
            model = DINABaseline(max_iter=5).fit(fitting)
            personas = attribute_personas(model.sample_patterns(2, seed=3), dataset.attribute_ids)
            output = Path(directory) / "raw.jsonl"
            failing = Mock()
            failing.answer.side_effect = RuntimeError("transport failed")
            with self.assertRaisesRegex(RuntimeError, "transport failed"):
                run_simulation(holdout, failing, personas, output, max_calls=1)
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["error_type"], "RuntimeError")
            with self.assertRaisesRegex(RuntimeError, "call budget exhausted"):
                run_simulation(
                    holdout,
                    DINAMockSimulator(model),
                    personas,
                    output,
                    resume=True,
                    max_calls=1,
                )
            self.assertEqual(len(output.read_text().splitlines()), 1)

    def test_parallel_students_preserve_item_order_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = make_synthetic_cdm_dataset(
                n_students=20, n_items=6, n_attributes=3, seed=10
            )
            _, holdout = train_test_split(dataset, seed=2)
            personas = [Persona(kind="none") for _ in range(4)]
            output = Path(directory) / "raw.jsonl"

            class RecordingSimulator:
                def __init__(self):
                    self.calls = []
                    self.lock = threading.Lock()

                def answer(self, persona, item, history, temperature):
                    with self.lock:
                        self.calls.append((id(persona), item.item_id))
                    return SimulationAnswer(item.answer_key, "", "parallel-mock")

            simulator = RecordingSimulator()
            responses = run_simulation(
                holdout, simulator, personas, output, max_workers=3, seed=42
            )
            self.assertEqual(responses.shape, (4, 6))
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(records), 24)
            self.assertEqual(
                len({(r["student_id"], r["item_id"]) for r in records}), 24
            )
            for index, persona in enumerate(personas):
                observed = [
                    item_id for persona_id, item_id in simulator.calls if persona_id == id(persona)
                ]
                expected = [
                    holdout.item_ids[item]
                    for item in latin_square_item_order(holdout.n_items, index, 42)
                ]
                self.assertEqual(observed, expected)

            class FailIfCalled:
                def answer(self, *args):
                    raise AssertionError("completed responses should not be requested again")

            resumed = run_simulation(
                holdout, FailIfCalled(), personas, output, resume=True, max_workers=3
            )
            np.testing.assert_array_equal(resumed, responses)
            self.assertEqual(len(output.read_text().splitlines()), 24)

    def test_parallel_budget_never_exceeds_cumulative_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = make_synthetic_cdm_dataset(
                n_students=20, n_items=4, n_attributes=2, seed=10
            )
            _, holdout = train_test_split(dataset, seed=2)
            personas = [Persona(kind="none") for _ in range(3)]
            output = Path(directory) / "raw.jsonl"

            class ImmediateSimulator:
                def answer(self, persona, item, history, temperature):
                    return SimulationAnswer(item.answer_key, "", "parallel-mock")

            with self.assertRaisesRegex(RuntimeError, "call budget exhausted"):
                run_simulation(
                    holdout,
                    ImmediateSimulator(),
                    personas,
                    output,
                    max_calls=5,
                    max_workers=3,
                )
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(records), 5)
            self.assertEqual(
                len({(r["student_id"], r["item_id"]) for r in records}), 5
            )
            with self.assertRaisesRegex(RuntimeError, "call budget exhausted"):
                run_simulation(
                    holdout,
                    ImmediateSimulator(),
                    personas,
                    output,
                    resume=True,
                    max_calls=5,
                    max_workers=3,
                )
            self.assertEqual(len(output.read_text().splitlines()), 5)

    def test_answer_parser_accepts_code_fence(self):
        final, work = parse_structured_answer(
            '```json\n{"final_answer":"3/8","work":"brief"}\n```'
        )
        self.assertEqual(final, "3/8")
        self.assertEqual(work, "brief")

    def test_answer_parser_accepts_unquoted_keys_and_ignores_think_braces(self):
        final, work = parse_structured_answer(
            '<think>{not json}</think>{final_answer: "3/8", work: "brief"}'
        )
        self.assertEqual(final, "3/8")
        self.assertEqual(work, "brief")
        self.assertEqual(parse_structured_answer('{final_answer: 0, work: "none"}')[0], "0")

    def test_rational_answer_equivalence(self):
        for answer in ["6/5", "1 1/5", "1.2", r"\frac{6}{5}", r"1\frac{1}{5}", r"$\dfrac{12}{10}$"]:
            self.assertTrue(answers_equivalent(answer, "6/5"), answer)
        self.assertEqual(str(parse_rational_answer("-1 1/2")), "-3/2")
        self.assertFalse(answers_equivalent("11/5", "6/5"))
        self.assertFalse(answers_equivalent("not a number", "6/5"))

    def test_error_example_prompt_and_temperature_are_explicit(self):
        simulator = OpenAICompatibleSimulator(
            model="controlled-model",
            api_key_env="TEST_API_KEY",
            prompt_strategy="error_examples",
            extra_body={"response_format": {"type": "json_object"}},
        )
        response = Mock()
        response.json.return_value = {
            "model": "controlled-model",
            "choices": [
                {"message": {"content": '{"final_answer":"3/4","work":""}'}}
            ],
        }
        item = Item("i1", "1/2 + 1/4 = ?", "3/4", np.array([1]), 0)
        with patch.dict("os.environ", {"TEST_API_KEY": "not-a-real-secret"}):
            with patch(
                "student_fidelity.simulation.simulators.httpx.post", return_value=response
            ) as post:
                simulator.answer(Persona(kind="none"), item, [], 0.0)
        request = post.call_args.kwargs["json"]
        self.assertEqual(request["temperature"], 0.0)
        self.assertIn("failing to regroup", request["messages"][0]["content"])
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(simulator.metadata()["prompt_strategy"], "error_examples")

    def test_fixed_temperature_provider_omits_temperature(self):
        simulator = OpenAICompatibleSimulator(
            model="fixed-model",
            api_key_env="TEST_API_KEY",
            temperature_supported=False,
            fixed_temperature=1.0,
            extra_body={"thinking": {"type": "disabled"}},
        )
        response = Mock()
        response.json.return_value = {
            "model": "fixed-model-actual",
            "choices": [
                {
                    "message": {"content": '{"final_answer":"6/5","work":""}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"total_tokens": 25},
        }
        item = Item("i1", "1/5 + 1 = ?", "6/5", np.array([1]), 0)
        with patch.dict("os.environ", {"TEST_API_KEY": "not-a-real-secret"}):
            with patch(
                "student_fidelity.simulation.simulators.httpx.post", return_value=response
            ) as post:
                answer = simulator.answer(Persona(kind="none"), item, [], 1.0)
        request = post.call_args.kwargs["json"]
        self.assertNotIn("temperature", request)
        self.assertEqual(request["thinking"], {"type": "disabled"})
        self.assertEqual(simulator.metadata()["fixed_temperature"], 1.0)
        self.assertEqual(answer.response_metadata["provider_model"], "fixed-model-actual")
        self.assertEqual(answer.response_metadata["usage"]["total_tokens"], 25)
        self.assertEqual(answer.response_metadata["finish_reason"], "stop")

    def test_fixed_temperature_provider_rejects_false_record(self):
        simulator = OpenAICompatibleSimulator(
            model="fixed-model",
            api_key_env="TEST_API_KEY",
            temperature_supported=False,
            fixed_temperature=1.0,
        )
        item = Item("i1", "1/5 + 1 = ?", "6/5", np.array([1]), 0)
        with patch.dict("os.environ", {"TEST_API_KEY": "not-a-real-secret"}):
            with self.assertRaisesRegex(ValueError, "refusing to record"):
                simulator.answer(Persona(kind="none"), item, [], 0.0)


if __name__ == "__main__":
    unittest.main()
