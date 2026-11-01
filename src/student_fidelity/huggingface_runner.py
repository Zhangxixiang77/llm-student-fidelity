"""Resolve, download, and preflight candidate Hugging Face models explicitly."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from student_fidelity.pilot_runner import attempt_count
from student_fidelity.simulation.huggingface import (
    HuggingFaceLocalSimulator,
    validate_huggingface_revision,
)
from student_fidelity.simulation.personas import Persona
from student_fidelity.simulation.runner import answers_equivalent
from student_fidelity.simulation.simulators import Item


def _configuration_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_huggingface_design(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    validate_huggingface_design(config)
    return config


def validate_huggingface_design(
    config: dict[str, Any], require_revisions: bool = False
) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Hugging Face candidate schema_version must be 1")
    if config.get("status") != "candidates_preflight_only":
        raise ValueError("candidate configuration permits preflight only")
    if config.get("n_students") != 100 or config.get("n_items") != 20:
        raise ValueError("candidate design must retain 100 students and 20 items")
    if config.get("state") != "stateless" or config.get("request_workers") != 1:
        raise ValueError("local candidate design requires stateless, sequential inference")
    if config.get("planned_student_calls_per_model") != 8000:
        raise ValueError("candidate design must retain 8000 student calls per model")
    if config.get("global_max_calls_per_model") != 8800:
        raise ValueError("candidate per-model cumulative ceiling must be 8800")
    if config.get("preflight_max_calls") != 12:
        raise ValueError("three-model preflight has a 12-call cumulative ceiling")
    if not 0 <= float(config.get("temperature", -1)) <= 2:
        raise ValueError("candidate temperature must be between zero and two")
    if int(config.get("max_new_tokens", 0)) < 1:
        raise ValueError("max_new_tokens must be positive")
    models = config.get("models")
    if not isinstance(models, list) or len(models) != 3:
        raise ValueError("exactly three Hugging Face candidates are required")
    if len({model.get("id") for model in models}) != 3 or len(
        {model.get("repo_id") for model in models}
    ) != 3:
        raise ValueError("candidate ids and repositories must be unique")
    for model in models:
        if model.get("license") not in {"apache-2.0", "mit"}:
            raise ValueError("candidate license must be recorded as apache-2.0 or mit")
        revision = model.get("revision")
        if require_revisions:
            validate_huggingface_revision(revision or "")
        elif revision is not None:
            validate_huggingface_revision(revision)
        if model.get("local_files_only") is not True:
            raise ValueError("generation must use local files only after explicit download")
        if model.get("trust_remote_code") is not False:
            raise ValueError("candidate models must not require trust_remote_code")
    checks = config.get("preflight_checks")
    if not isinstance(checks, list) or len(checks) != 2:
        raise ValueError("two prompt-specific preflight checks are required")
    if {check.get("prompt_strategy") for check in checks} != {
        "answer_first",
        "error_examples",
    }:
        raise ValueError("preflight must cover both locked prompt strategies")
    cells = config.get("cells")
    expected = {
        ("none", "answer_first"),
        ("none", "error_examples"),
        ("attribute", "answer_first"),
        ("attribute", "error_examples"),
    }
    if not isinstance(cells, list) or len(cells) != 4:
        raise ValueError("candidate design requires four cells")
    if {(cell.get("persona"), cell.get("prompt_strategy")) for cell in cells} != expected:
        raise ValueError("candidate cells do not match the two-by-two design")
    if any(cell.get("max_calls") != 2200 for cell in cells):
        raise ValueError("each candidate cell has a 2200-call cumulative ceiling")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_huggingface_metadata(
    config_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Resolve public model metadata only; this does not download model weights."""

    config = load_huggingface_design(config_path)
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError("metadata resolution requires the optional 'hf' dependencies") from error
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    resolved = []
    for model in config["models"]:
        info = api.model_info(model["repo_id"])
        revision = info.sha or ""
        validate_huggingface_revision(revision)
        card = info.cardData or {}
        license_value = card.get("license")
        if isinstance(license_value, list):
            license_value = license_value[0] if license_value else None
        normalized_license = str(license_value).lower() if license_value else None
        if normalized_license != model["license"]:
            raise ValueError(
                f"license mismatch for {model['repo_id']}: "
                f"expected {model['license']}, found {normalized_license}"
            )
        resolved.append(
            {
                "id": model["id"],
                "repo_id": info.id,
                "revision": revision,
                "license": normalized_license,
                "gated": info.gated,
                "last_modified": info.lastModified,
            }
        )
    result = {
        "resolved_at_utc": datetime.now(UTC).isoformat(),
        "candidate_configuration_sha256": _configuration_sha256(config),
        "weights_downloaded": False,
        "models": resolved,
    }
    _write_json(Path(output_path), result)
    return result


def _merge_metadata(
    config: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    if metadata.get("candidate_configuration_sha256") != _configuration_sha256(config):
        raise ValueError("resolved model metadata does not match the candidate configuration")
    by_id = {model["id"]: model for model in metadata.get("models", [])}
    if set(by_id) != {model["id"] for model in config["models"]}:
        raise ValueError("resolved metadata must contain every candidate exactly once")
    merged = copy.deepcopy(config)
    for model in merged["models"]:
        resolved = by_id[model["id"]]
        if resolved.get("repo_id") != model["repo_id"]:
            raise ValueError("resolved repository differs from the candidate repository")
        if resolved.get("license") != model["license"]:
            raise ValueError("resolved license differs from the candidate license")
        model["revision"] = validate_huggingface_revision(resolved.get("revision", ""))
    validate_huggingface_design(merged, require_revisions=True)
    return merged


def _load_with_optional_metadata(
    config_path: str | Path, metadata_path: str | Path | None
) -> dict[str, Any]:
    config = load_huggingface_design(config_path)
    if metadata_path is None:
        return config
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return _merge_metadata(config, metadata)


def download_huggingface_models(
    config_path: str | Path, metadata_path: str | Path, manifest_path: str | Path
) -> dict[str, Any]:
    """Download explicitly resolved snapshots; never invoked by dry run or preflight."""

    config = _load_with_optional_metadata(config_path, metadata_path)
    validate_huggingface_design(config, require_revisions=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("model download requires the optional 'hf' dependencies") from error
    token = os.environ.get("HF_TOKEN") or None
    downloaded = []
    for model in config["models"]:
        local_path = snapshot_download(
            repo_id=model["repo_id"],
            revision=model["revision"],
            cache_dir=config.get("cache_dir"),
            allow_patterns=config.get("download_allow_patterns"),
            token=token,
        )
        downloaded.append(
            {
                "id": model["id"],
                "repo_id": model["repo_id"],
                "revision": model["revision"],
                "local_path": local_path,
            }
        )
    result = {
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "resolved_configuration_sha256": _configuration_sha256(config),
        "models": downloaded,
    }
    _write_json(Path(manifest_path), result)
    return result


def _preflight_paths(root: Path, config: dict[str, Any]) -> list[Path]:
    return [root / model["id"] / "preflight.jsonl" for model in config["models"]]


def _preflight_attempts(root: Path, config: dict[str, Any]) -> int:
    return sum(attempt_count(path) for path in _preflight_paths(root, config))


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_huggingface_preflight(
    config_path: str | Path,
    execute: bool = False,
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    """Preflight local snapshots; this function never reads FrcSub or generates students."""

    config = _load_with_optional_metadata(config_path, metadata_path)
    validate_huggingface_design(config, require_revisions=execute)
    root = Path(config["output_dir"])
    resolved_sha = _configuration_sha256(config)
    plan = {
        "resolved_configuration_sha256": resolved_sha,
        "candidate_ids": [model["id"] for model in config["models"]],
        "candidate_revisions": {
            model["id"]: model["revision"] for model in config["models"]
        },
        "planned_preflight_calls": len(config["models"]) * len(config["preflight_checks"]),
        "preflight_max_calls": config["preflight_max_calls"],
        "attempts_before_run": _preflight_attempts(root, config),
        "student_generation_executed": False,
        "dataset_read": False,
        "executed": execute,
    }
    if not execute:
        return plan

    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "hf_preflight_plan.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        if previous.get("resolved_configuration_sha256") != resolved_sha:
            raise ValueError("resolved candidate configuration changed after preflight began")
    _write_json(plan_path, plan)
    item = Item(
        item_id="preflight",
        text="Compute 1/2 + 1/4.",
        answer_key="3/4",
        q_vector=np.zeros(8, dtype=np.int8),
        index=0,
    )
    for model in config["models"]:
        path = root / model["id"] / "preflight.jsonl"
        existing = []
        if path.exists():
            existing = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        simulator = HuggingFaceLocalSimulator(
            model=model["repo_id"],
            revision=model["revision"],
            cache_dir=config.get("cache_dir"),
            device=model.get("device", "auto"),
            dtype=model.get("dtype", "auto"),
            max_new_tokens=int(config["max_new_tokens"]),
            local_files_only=True,
            trust_remote_code=False,
            chat_template_kwargs=model.get("chat_template_kwargs"),
            top_p=float(config.get("top_p", 1.0)),
            seed=int(config["seed"]),
        )
        try:
            for check in config["preflight_checks"]:
                prompt_strategy = check["prompt_strategy"]
                simulator.prompt_strategy = prompt_strategy
                if any(
                    record.get("check_id") == check["id"] and "final_answer" in record
                    for record in existing
                ):
                    continue
                if _preflight_attempts(root, config) >= config["preflight_max_calls"]:
                    raise RuntimeError("Hugging Face preflight call budget exhausted")
                try:
                    answer = simulator.answer(
                        Persona(kind="none"), item, [], float(config["temperature"])
                    )
                except Exception as error:
                    _append_jsonl(
                        path,
                        {
                            "check_id": check["id"],
                            "prompt_strategy": prompt_strategy,
                            "temperature": config["temperature"],
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "raw_response": getattr(error, "raw_response", None),
                            "response_metadata": getattr(error, "response_metadata", {}),
                        },
                    )
                    raise
                correct = answers_equivalent(answer.final_answer, item.answer_key)
                record = {
                    "check_id": check["id"],
                    "prompt_strategy": prompt_strategy,
                    "temperature": config["temperature"],
                    "final_answer": answer.final_answer,
                    "work": answer.work,
                    "is_correct": correct,
                    "raw_response": answer.raw_response,
                    "response_metadata": answer.response_metadata,
                }
                metadata = answer.response_metadata
                valid = (
                    metadata.get("provider_model") == model["repo_id"]
                    and metadata.get("provider_revision") == model["revision"]
                    and metadata.get("finish_reason") == "stop"
                )
                if not valid:
                    failure = {
                        **record,
                        "error_type": "HuggingFacePreflightValidationError",
                        "error": "local model identity, revision, or finish reason failed validation",
                    }
                    failure.pop("final_answer")
                    _append_jsonl(path, failure)
                    raise RuntimeError(failure["error"])
                _append_jsonl(path, record)
                existing.append(record)
        finally:
            simulator.close()

    successful = [
        json.loads(line)
        for path in _preflight_paths(root, config)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and "final_answer" in json.loads(line)
    ]
    plan["attempts_after_run"] = _preflight_attempts(root, config)
    plan["successful_preflight_calls"] = len(successful)
    plan["all_successful_answers_correct"] = all(
        record["is_correct"] for record in successful
    )
    plan["status"] = "preflight_completed"
    _write_json(plan_path, plan)
    return plan
