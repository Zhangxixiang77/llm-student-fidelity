"""Execute a locked multi-cell pilot with one cumulative call ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from student_fidelity.data import ResponseDataset
from student_fidelity.experiment import execute_pilot
from student_fidelity.simulation.personas import Persona
from student_fidelity.simulation.simulators import Item, OpenAICompatibleSimulator


REQUEST_WORKERS = 8


def load_locked_design(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    validate_locked_design(config)
    return config


def validate_locked_design(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("locked pilot schema_version must be 1")
    if config.get("status") != "locked_ready_for_preflight":
        raise ValueError("pilot must be locked before execution")
    if config.get("n_students") != 100 or config.get("n_items") != 20:
        raise ValueError("locked pilot requires 100 students and 20 items")
    if config.get("state") != "stateless":
        raise ValueError("locked pilot requires stateless responses")
    if config.get("global_max_calls") != 8800:
        raise ValueError("locked cumulative call budget must be 8800")
    if config.get("dina_n_starts") != 30 or config.get("dina_max_iter") != 300:
        raise ValueError("locked DINA estimator requires 30 starts and 300 iterations")
    simulator = config.get("simulator", {})
    if simulator.get("type") != "openai_compatible":
        raise ValueError("formal pilot requires an explicit real model")
    temperature = config.get("temperature")
    fixed_temperature = simulator.get("fixed_temperature")
    if simulator.get("temperature_supported", False):
        if fixed_temperature is not None:
            raise ValueError("fixed-temperature models are ineligible for explicit control")
        if temperature != 0.0:
            raise ValueError("explicit-temperature pilot requires temperature 0")
    else:
        if fixed_temperature is None or temperature != fixed_temperature:
            raise ValueError("locked temperature must equal the model's fixed temperature")
    checks = _preflight_checks(config)
    if len(checks) != 2 or len({check.get("id") for check in checks}) != 2:
        raise ValueError("locked pilot requires two uniquely identified preflight checks")
    for check in checks:
        if check.get("prompt_strategy") not in {"answer_first", "error_examples"}:
            raise ValueError("invalid preflight prompt strategy")
        if not simulator.get("temperature_supported", False):
            if check.get("temperature") != fixed_temperature:
                raise ValueError("fixed-temperature preflight uses an unapplied value")
    cells = config.get("cells")
    if not isinstance(cells, list) or len(cells) != 4:
        raise ValueError("locked pilot requires exactly four cells")
    expected = {
        ("none", "answer_first"),
        ("none", "error_examples"),
        ("attribute", "answer_first"),
        ("attribute", "error_examples"),
    }
    observed = {(cell.get("persona"), cell.get("prompt_strategy")) for cell in cells}
    if observed != expected or len({cell.get("id") for cell in cells}) != 4:
        raise ValueError("pilot cells do not match the locked two-by-two design")
    if any(cell.get("max_calls") != 2200 for cell in cells):
        raise ValueError("each locked cell has a 2200-call cumulative ceiling")
    planned = config["n_students"] * config["n_items"] * len(cells)
    if planned != 8000 or config.get("planned_student_calls") != planned:
        raise ValueError("planned student call count must be 8000")


def configuration_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def attempt_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        json.loads(line)
        count += 1
    return count


def _ledger_paths(root: Path, config: dict[str, Any]) -> list[Path]:
    return [root / "preflight.jsonl"] + [
        root / cell["id"] / "raw_simulation.jsonl" for cell in config["cells"]
    ]


def total_attempts(root: Path, config: dict[str, Any]) -> int:
    return sum(attempt_count(path) for path in _ledger_paths(root, config))


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")


def _preflight_checks(config: dict[str, Any]) -> list[dict[str, Any]]:
    return config.get(
        "preflight_checks",
        [
            {"id": "temperature_0", "temperature": 0.0, "prompt_strategy": "answer_first"},
            {"id": "temperature_1", "temperature": 1.0, "prompt_strategy": "answer_first"},
        ],
    )


def _run_preflight(root: Path, config: dict[str, Any]) -> None:
    path = root / "preflight.jsonl"
    existing = []
    if path.exists():
        existing = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    simulator_config = {
        key: value for key, value in config["simulator"].items() if key != "type"
    }
    item = Item(
        item_id="preflight",
        text="Compute 1/2 + 1/4.",
        answer_key="3/4",
        q_vector=np.zeros(8, dtype=np.int8),
        index=0,
    )
    for check in _preflight_checks(config):
        temperature = check["temperature"]
        prompt_strategy = check["prompt_strategy"]
        simulator = OpenAICompatibleSimulator(
            **simulator_config, prompt_strategy=prompt_strategy
        )
        if any(
            record.get("check_id") == check["id"] and "final_answer" in record
            for record in existing
        ):
            continue
        if total_attempts(root, config) >= config["global_max_calls"]:
            raise RuntimeError("global call budget exhausted before preflight")
        try:
            answer = simulator.answer(Persona(kind="none"), item, [], temperature)
        except Exception as error:
            _append_jsonl(
                path,
                {
                    "check_id": check["id"],
                    "prompt_strategy": prompt_strategy,
                    "temperature": temperature,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "raw_response": getattr(error, "raw_response", None),
                    "response_metadata": getattr(error, "response_metadata", {}),
                },
            )
            raise
        record = {
            "check_id": check["id"],
            "prompt_strategy": prompt_strategy,
            "temperature": temperature,
            "final_answer": answer.final_answer,
            "work": answer.work,
            "raw_response": answer.raw_response,
            "response_metadata": answer.response_metadata,
        }
        provider_model = answer.response_metadata.get("provider_model")
        finish_reason = answer.response_metadata.get("finish_reason")
        if provider_model != config["simulator"]["model"] or finish_reason != "stop":
            failure = {
                **record,
                "error_type": "PreflightValidationError",
                "error": (
                    f"provider_model={provider_model!r}, finish_reason={finish_reason!r}; "
                    "student generation was not started"
                ),
            }
            failure.pop("final_answer")
            _append_jsonl(path, failure)
            raise RuntimeError(failure["error"])
        _append_jsonl(path, record)
        existing.append(record)


def run_locked_pilot(config_path: str | Path, execute: bool = False) -> dict[str, Any]:
    """Validate a design always; perform paid requests only with execute=True."""

    config = load_locked_design(config_path)
    root = Path(config["output_dir"])
    digest = configuration_sha256(config)
    plan = {
        "configuration_sha256": digest,
        "planned_student_calls": config["planned_student_calls"],
        "planned_preflight_calls": len(_preflight_checks(config)),
        "global_max_calls": config["global_max_calls"],
        "attempts_before_run": total_attempts(root, config),
        "cells": [cell["id"] for cell in config["cells"]],
        "request_workers": REQUEST_WORKERS,
        "executed": execute,
    }
    if not execute:
        return plan

    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "pilot_plan.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        if previous.get("configuration_sha256") != digest:
            raise ValueError("locked pilot configuration changed after execution began")
    else:
        plan_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    _run_preflight(root, config)
    dataset = ResponseDataset.from_directory(Path(config["dataset"]["path"]))
    if dataset.n_items != config["n_items"]:
        raise ValueError("dataset item count differs from the locked design")

    for cell in config["cells"]:
        cell_root = root / cell["id"]
        raw_path = cell_root / "raw_simulation.jsonl"
        cell_attempts = attempt_count(raw_path)
        remaining = config["global_max_calls"] - total_attempts(root, config)
        if remaining < 0:
            raise RuntimeError("global call ledger exceeds the locked budget")
        allowed = min(cell["max_calls"], cell_attempts + remaining)
        if allowed < cell_attempts:
            raise RuntimeError("global call budget exhausted before cell resume")
        execute_pilot(
            dataset=dataset,
            output_dir=cell_root,
            simulator_config=config["simulator"],
            baseline_names=config["baselines"],
            persona_kind=cell["persona"],
            n_students=config["n_students"],
            state=config["state"],
            temperature=config["temperature"],
            noise_repeats=config["noise_repeats"],
            seed=config["seed"],
            resume=True,
            max_calls=allowed,
            prompt_strategy=cell["prompt_strategy"],
            dina_n_starts=config["dina_n_starts"],
            dina_max_iter=config["dina_max_iter"],
            max_workers=REQUEST_WORKERS,
        )

    plan["attempts_after_run"] = total_attempts(root, config)
    plan["status"] = "completed"
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return plan
