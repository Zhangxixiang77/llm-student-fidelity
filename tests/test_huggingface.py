import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from student_fidelity.huggingface_runner import (
    load_huggingface_design,
    run_huggingface_preflight,
)
from student_fidelity.simulation.huggingface import (
    HuggingFaceLocalSimulator,
    validate_huggingface_revision,
)
from student_fidelity.simulation.personas import Persona
from student_fidelity.simulation.simulators import Item, SimulationAnswer

REVISION = "a" * 40


class HuggingFaceTests(unittest.TestCase):
    def _candidate_config(self, root: Path) -> dict:
        repository = Path(__file__).resolve().parents[1]
        config = yaml.safe_load(
            (repository / "configs/huggingface_candidates_v1.yaml").read_text(
                encoding="utf-8"
            )
        )
        config["output_dir"] = str(root / "outputs")
        config["cache_dir"] = str(root / "cache")
        return config

    def _write_candidate(self, root: Path, resolved: bool = False) -> Path:
        config = self._candidate_config(root)
        if resolved:
            for model in config["models"]:
                model["revision"] = REVISION
        path = root / "candidate.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path

    def test_dry_run_uses_no_model_dataset_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_candidate(root)
            plan = run_huggingface_preflight(path)
            self.assertFalse(plan["executed"])
            self.assertFalse(plan["student_generation_executed"])
            self.assertFalse(plan["dataset_read"])
            self.assertEqual(plan["planned_preflight_calls"], 6)
            self.assertFalse((root / "outputs").exists())

    def test_mutable_or_missing_revisions_cannot_execute(self):
        with self.assertRaisesRegex(ValueError, "40-character"):
            validate_huggingface_revision("main")
        with self.assertRaisesRegex(ValueError, "40-character"):
            HuggingFaceLocalSimulator("candidate", REVISION[:-1])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_candidate(root, resolved=False)
            with self.assertRaisesRegex(ValueError, "40-character"):
                run_huggingface_preflight(path, execute=True)
            self.assertFalse((root / "outputs").exists())

    def test_local_answer_reuses_error_prompt_and_records_revision(self):
        simulator = HuggingFaceLocalSimulator(
            "candidate/model", REVISION, prompt_strategy="error_examples"
        )
        metadata = {
            "provider_model": "candidate/model",
            "provider_revision": REVISION,
            "finish_reason": "stop",
        }
        item = Item("i1", "1/2 + 1/4", "3/4", np.array([1]), 0)
        with patch.object(
            simulator, "_generate", return_value=('{"final_answer":"3/4","work":""}', metadata)
        ) as generate:
            answer = simulator.answer(Persona(kind="none"), item, [], 0.0)
        self.assertEqual(answer.final_answer, "3/4")
        self.assertIn("failing to regroup", generate.call_args.args[0])
        self.assertEqual(answer.response_metadata["provider_revision"], REVISION)

    def test_greedy_generation_does_not_claim_sampling(self):
        class FakeArray:
            def __init__(self, values):
                self.values = np.asarray(values)

            @property
            def shape(self):
                return self.values.shape

            def to(self, device):
                return self

            def __getitem__(self, key):
                return FakeArray(self.values[key])

            def item(self):
                return self.values.item()

        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        class FakeTorch:
            cuda = FakeCuda()

            @staticmethod
            def manual_seed(seed):
                return None

            @staticmethod
            def inference_mode():
                return nullcontext()

        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"]

            def __call__(self, rendered, return_tensors):
                return {"input_ids": FakeArray([[8, 8, 8, 8]])}

            def decode(self, output_ids, skip_special_tokens):
                return '{"final_answer":"3/4","work":""}'

        class FakeModel:
            device = "cpu"

            def __init__(self):
                self.kwargs = None

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return FakeArray([[8, 8, 8, 8, 5, 2]])

        simulator = HuggingFaceLocalSimulator("candidate/model", REVISION, max_new_tokens=2)
        simulator._torch = FakeTorch()
        simulator._tokenizer = FakeTokenizer()
        simulator._model = FakeModel()
        content, metadata = simulator._generate("prompt", 0.0)
        self.assertIn("final_answer", content)
        self.assertFalse(simulator._model.kwargs["do_sample"])
        self.assertNotIn("temperature", simulator._model.kwargs)
        self.assertNotIn("top_p", simulator._model.kwargs)
        self.assertEqual(metadata["finish_reason"], "stop")
        self.assertEqual(metadata["usage"]["output_tokens"], 2)

    def test_preflight_covers_three_models_and_resumes_without_new_calls(self):
        class FakeSimulator:
            calls = 0
            closes = 0

            def __init__(self, model, revision, **kwargs):
                self.model = model
                self.revision = revision
                self.prompt_strategy = "answer_first"

            def answer(self, persona, item, history, temperature):
                type(self).calls += 1
                return SimulationAnswer(
                    "3/4",
                    "",
                    '{"final_answer":"3/4","work":""}',
                    response_metadata={
                        "provider_model": self.model,
                        "provider_revision": self.revision,
                        "finish_reason": "stop",
                    },
                )

            def close(self):
                type(self).closes += 1

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_candidate(root, resolved=True)
            with patch(
                "student_fidelity.huggingface_runner.HuggingFaceLocalSimulator",
                FakeSimulator,
            ):
                plan = run_huggingface_preflight(path, execute=True)
                self.assertEqual(FakeSimulator.calls, 6)
                self.assertEqual(plan["successful_preflight_calls"], 6)
                self.assertTrue(plan["all_successful_answers_correct"])
                resumed = run_huggingface_preflight(path, execute=True)
                self.assertEqual(FakeSimulator.calls, 6)
                self.assertEqual(resumed["attempts_after_run"], 6)
            self.assertEqual(FakeSimulator.closes, 6)
            for model in load_huggingface_design(path)["models"]:
                ledger = root / "outputs" / model["id"] / "preflight.jsonl"
                records = [json.loads(line) for line in ledger.read_text().splitlines()]
                self.assertEqual(
                    {record["prompt_strategy"] for record in records},
                    {"answer_first", "error_examples"},
                )


if __name__ == "__main__":
    unittest.main()
