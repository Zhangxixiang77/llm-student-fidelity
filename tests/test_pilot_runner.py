import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from student_fidelity.pilot_runner import (
    _cell_is_complete,
    _run_preflight,
    attempt_count,
    load_locked_design,
    run_locked_pilot,
    total_attempts,
    validate_locked_design,
)


class LockedPilotTests(unittest.TestCase):
    def test_repository_design_dry_run_uses_no_credentials_or_output(self):
        root = Path(__file__).resolve().parents[1]
        config_path = root / "configs/pilot_frcsub_minimax_m3_v1.yaml"
        config = load_locked_design(config_path)
        self.assertEqual(config["simulator"]["model"], "MiniMax-M3")
        self.assertEqual(config["global_max_calls"], 8800)
        self.assertEqual(config["planned_student_calls"], 8000)
        self.assertEqual(config["simulator"]["fixed_temperature"], 1.0)
        self.assertEqual(config["temperature"], 1.0)
        plan = run_locked_pilot(config_path, execute=False)
        self.assertFalse(plan["executed"])
        self.assertEqual(len(plan["cells"]), 4)

    def test_design_rejects_unapplied_temperature_or_changed_budget(self):
        root = Path(__file__).resolve().parents[1]
        config = load_locked_design(root / "configs/pilot_frcsub_minimax_m3_v1.yaml")
        config["temperature"] = 0.0
        with self.assertRaisesRegex(ValueError, "fixed temperature"):
            validate_locked_design(config)
        config["temperature"] = 1.0
        config["global_max_calls"] = 9000
        with self.assertRaisesRegex(ValueError, "8800"):
            validate_locked_design(config)

    def test_superseded_openai_design_cannot_be_executed(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "configs/pilot_frcsub_gpt4omini_v1.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "locked"):
            validate_locked_design(config)
        config["status"] = "locked_ready_for_preflight"
        validate_locked_design(config)

    def test_attempt_ledger_counts_success_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempts.jsonl"
            path.write_text(
                '{"is_correct": true}\n{"error_type": "HTTPStatusError"}\n'
            )
            self.assertEqual(attempt_count(path), 2)

    def test_completed_cell_requires_matching_summary_and_unique_successes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = {
                "student_id": "student_000",
                "item_id": "item_001",
                "final_answer": "1/2",
            }
            (root / "raw_simulation.jsonl").write_text(
                json.dumps(record) + "\n" + json.dumps({"error": "transport"}) + "\n"
            )
            (root / "run_manifest.json").write_text(
                json.dumps({"configuration_sha256": "abc"})
            )
            (root / "summary.json").write_text(
                json.dumps({"configuration_sha256": "abc"})
            )
            self.assertTrue(_cell_is_complete(root, 1))
            self.assertFalse(_cell_is_complete(root, 2))
            (root / "summary.json").write_text(
                json.dumps({"configuration_sha256": "changed"})
            )
            self.assertFalse(_cell_is_complete(root, 1))

    def test_preflight_covers_both_prompts_and_resumes_without_new_calls(self):
        root = Path(__file__).resolve().parents[1]
        config = load_locked_design(root / "configs/pilot_frcsub_minimax_m3_v1.yaml")
        response = Mock()
        response.json.return_value = {
            "model": "MiniMax-M3",
            "choices": [
                {
                    "message": {"content": '{"final_answer":"3/4","work":""}'},
                    "finish_reason": "stop",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config["output_dir"] = str(output)
            with patch.dict("os.environ", {"MINIMAX_API_KEY": "not-a-real-secret"}):
                with patch(
                    "student_fidelity.simulation.simulators.httpx.post",
                    return_value=response,
                ) as post:
                    _run_preflight(output, config)
                    self.assertEqual(post.call_count, 2)
                    for call in post.call_args_list:
                        self.assertNotIn("temperature", call.kwargs["json"])
                    prompts = [
                        call.kwargs["json"]["messages"][0]["content"]
                        for call in post.call_args_list
                    ]
                    self.assertNotIn("examples of mistakes", prompts[0])
                    self.assertIn("examples of mistakes", prompts[1])
                    _run_preflight(output, config)
                    self.assertEqual(post.call_count, 2)
            self.assertEqual(total_attempts(output, config), 2)

    def test_preflight_rejects_provider_model_mismatch(self):
        root = Path(__file__).resolve().parents[1]
        config = load_locked_design(root / "configs/pilot_frcsub_minimax_m3_v1.yaml")
        response = Mock()
        response.json.return_value = {
            "model": "unexpected-model",
            "choices": [
                {
                    "message": {"content": '{"final_answer":"3/4","work":""}'},
                    "finish_reason": "stop",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config["output_dir"] = str(output)
            with patch.dict("os.environ", {"MINIMAX_API_KEY": "not-a-real-secret"}):
                with patch(
                    "student_fidelity.simulation.simulators.httpx.post",
                    return_value=response,
                ):
                    with self.assertRaisesRegex(RuntimeError, "unexpected-model"):
                        _run_preflight(output, config)
            self.assertEqual(total_attempts(output, config), 1)

    def test_dry_run_does_not_create_output_directory(self):
        root = Path(__file__).resolve().parents[1]
        config = load_locked_design(root / "configs/pilot_frcsub_minimax_m3_v1.yaml")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            config["output_dir"] = str(output)
            path = Path(directory) / "design.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            run_locked_pilot(path, execute=False)
            self.assertFalse(output.exists())
            self.assertEqual(total_attempts(output, config), 0)


if __name__ == "__main__":
    unittest.main()
