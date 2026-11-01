"""End-to-end pilot orchestration with strict train/holdout separation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from student_fidelity.baselines import DINABaseline, IRT2PLBaseline, make_baseline
from student_fidelity.data import ResponseDataset, train_test_split
from student_fidelity.metrics import (
    evaluate_structure,
    implicit_slip_guess,
    readback_agreement,
    split_half_noise,
)
from student_fidelity.simulation import (
    DINAMockSimulator,
    OpenAICompatibleSimulator,
    Persona,
    attribute_personas,
    run_simulation,
)


def _noise_cache_metadata(
    dataset: ResponseDataset, n_repeats: int, seed: int
) -> dict[str, Any]:
    return {
        "responses_sha256": hashlib.sha256(dataset.responses.tobytes()).hexdigest(),
        "n_repeats": n_repeats,
        "seed": seed,
    }


def _validated_noise(values: Any, n_repeats: int) -> NDArray[np.float64]:
    noise = np.asarray(values, dtype=np.float64)
    if noise.shape != (n_repeats,) or not np.isfinite(noise).all():
        raise ValueError("cached split-half noise has an invalid shape or values")
    return noise


def _write_noise_cache(
    path: Path, noise: NDArray[np.float64], metadata: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez(temporary, values=noise, metadata=json.dumps(metadata, sort_keys=True))
    temporary.replace(path)


def _load_or_compute_split_half_noise(
    dataset: ResponseDataset,
    n_repeats: int,
    seed: int,
    cache_path: str | Path | None,
    run_noise_path: str | Path,
) -> NDArray[np.float64]:
    """Reuse only identical real-data noise estimates; never cache LLM responses."""

    metadata = _noise_cache_metadata(dataset, n_repeats, seed)
    cache = Path(cache_path) if cache_path is not None else None
    if cache is not None and cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            saved_metadata = json.loads(str(saved["metadata"].item()))
            if saved_metadata != metadata:
                raise ValueError("split-half noise cache does not match this dataset and design")
            return _validated_noise(saved["values"], n_repeats)

    run_path = Path(run_noise_path)
    noise = None
    if run_path.exists():
        frame = pd.read_csv(run_path)
        if list(frame.columns) == ["split_half_dependency_rmse"]:
            noise = _validated_noise(
                frame["split_half_dependency_rmse"].to_numpy(), n_repeats
            )
    if noise is None:
        noise = _validated_noise(split_half_noise(dataset, n_repeats, seed), n_repeats)
    if cache is not None:
        _write_noise_cache(cache, noise, metadata)
    return noise


def execute_pilot(
    dataset: ResponseDataset,
    output_dir: str | Path,
    simulator_config: dict[str, Any] | None = None,
    baseline_names: Sequence[str] = ("independent", "copula", "irt", "dina"),
    persona_kind: str = "attribute",
    n_students: int | None = None,
    state: str = "stateless",
    temperature: float = 0.0,
    noise_repeats: int = 200,
    seed: int = 42,
    resume: bool = False,
    max_calls: int | None = None,
    prompt_strategy: str = "answer_first",
    dina_n_starts: int = 10,
    dina_max_iter: int = 100,
    max_workers: int = 1,
    noise_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one experimental cell; callers must repeat cells/seeds for the full design."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fitting, holdout = train_test_split(dataset, test_size=0.5, seed=seed)
    n_students = n_students or holdout.n_students
    if n_students < 1:
        raise ValueError("n_students must be positive")

    dina = DINABaseline(
        max_iter=dina_max_iter, n_starts=dina_n_starts, seed=seed + 9
    ).fit(fitting)
    target_alpha = dina.sample_patterns(n_students, seed + 1)
    if persona_kind == "attribute":
        personas = attribute_personas(target_alpha, dataset.attribute_ids)
    elif persona_kind == "ability":
        irt_for_theta = IRT2PLBaseline().fit(fitting)
        theta = irt_for_theta.sample_theta(n_students, seed + 2)
        personas = [Persona(kind="ability", theta=float(value)) for value in theta]
    elif persona_kind == "none":
        personas = [Persona(kind="none") for _ in range(n_students)]
    else:
        raise ValueError("persona_kind must be none, ability, or attribute")

    simulator_config = simulator_config or {"type": "dina_mock"}
    simulator_type = simulator_config.get("type", "dina_mock")
    if simulator_type == "dina_mock":
        simulator = DINAMockSimulator(dina, seed=seed + 3)
    elif simulator_type == "openai_compatible":
        simulator = OpenAICompatibleSimulator(
            model=simulator_config["model"],
            base_url=simulator_config.get("base_url", "https://api.openai.com/v1"),
            api_key_env=simulator_config.get("api_key_env", "OPENAI_API_KEY"),
            timeout=float(simulator_config.get("timeout", 120)),
            extra_body=simulator_config.get("extra_body"),
            temperature_supported=bool(simulator_config.get("temperature_supported", True)),
            fixed_temperature=simulator_config.get("fixed_temperature"),
            prompt_strategy=prompt_strategy,
        )
    elif simulator_type == "huggingface_local":
        from student_fidelity.simulation.huggingface import HuggingFaceLocalSimulator

        simulator = HuggingFaceLocalSimulator(
            model=simulator_config["model"],
            revision=simulator_config["revision"],
            cache_dir=simulator_config.get("cache_dir"),
            device=simulator_config.get("device", "auto"),
            dtype=simulator_config.get("dtype", "auto"),
            max_new_tokens=int(simulator_config.get("max_new_tokens", 1024)),
            local_files_only=bool(simulator_config.get("local_files_only", True)),
            trust_remote_code=bool(simulator_config.get("trust_remote_code", False)),
            chat_template_kwargs=simulator_config.get("chat_template_kwargs"),
            prompt_strategy=prompt_strategy,
            top_p=float(simulator_config.get("top_p", 1.0)),
            seed=seed + 20,
        )
    else:
        raise ValueError(f"unsupported simulator type: {simulator_type}")

    configuration = {
        "dataset": {
            "responses_sha256": hashlib.sha256(dataset.responses.tobytes()).hexdigest(),
            "q_matrix_sha256": hashlib.sha256(dataset.q_matrix.tobytes()).hexdigest(),
            "item_ids": list(dataset.item_ids),
            "student_ids": list(dataset.student_ids),
            "attribute_ids": list(dataset.attribute_ids),
            "item_texts": list(dataset.item_texts) if dataset.item_texts is not None else None,
            "answer_keys": list(dataset.answer_keys) if dataset.answer_keys is not None else None,
        },
        "design": {
            "persona": persona_kind,
            "state": state,
            "temperature": temperature,
            "prompt_strategy": prompt_strategy,
            "n_virtual_students": n_students,
            "dina_n_starts": dina_n_starts,
            "dina_max_iter": dina_max_iter,
            "seed": seed,
        },
        "simulator": simulator_config,
    }
    configuration_sha256 = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    raw_path = output_dir / "raw_simulation.jsonl"
    manifest_path = output_dir / "run_manifest.json"
    if resume and raw_path.exists():
        if not manifest_path.exists():
            raise RuntimeError("cannot safely resume without a configuration manifest")
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("configuration_sha256") != configuration_sha256:
            raise ValueError("resume configuration differs from the previous run")
    manifest_path.write_text(
        json.dumps(
            {"configuration_sha256": configuration_sha256, "configuration": configuration},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    simulated = run_simulation(
        holdout,
        simulator,
        personas,
        raw_path,
        state=state,
        temperature=temperature,
        seed=seed,
        resume=resume,
        max_calls=max_calls,
        max_workers=max_workers,
    )
    np.savetxt(output_dir / "assigned_alpha.csv", target_alpha, fmt="%d", delimiter=",")

    results: dict[str, dict[str, Any]] = {
        "simulator": evaluate_structure(holdout.responses, simulated)
    }
    if persona_kind == "attribute":
        estimated_alpha = dina.map_patterns(simulated)
        results["simulator"].update(readback_agreement(target_alpha, estimated_alpha))
        results["simulator"].update(
            implicit_slip_guess(simulated, target_alpha, dataset.q_matrix)
        )
        results["simulator"]["readback_ceiling"] = dina.readback_ceiling(
            target_alpha, seed=seed + 4
        )

    generated_dir = output_dir / "generated"
    generated_dir.mkdir(exist_ok=True)
    _write_responses(simulated, generated_dir / "simulator.csv")
    for index, name in enumerate(baseline_names):
        if name == "dina":
            baseline = dina
        else:
            baseline = make_baseline(name).fit(fitting)
        generated = baseline.sample(n_students, seed=seed + 10 + index)
        results[name] = evaluate_structure(holdout.responses, generated)
        _write_responses(generated, generated_dir / f"{name}.csv")

    noise_path = output_dir / "split_half_noise.csv"
    noise = _load_or_compute_split_half_noise(
        dataset,
        n_repeats=noise_repeats,
        seed=seed + 100,
        cache_path=noise_cache_path,
        run_noise_path=noise_path,
    )
    delta = float(np.quantile(noise, 0.95))
    pd.DataFrame({"split_half_dependency_rmse": noise}).to_csv(noise_path, index=False)
    rows = []
    for condition, metrics in results.items():
        rows.append(
            {
                "condition": condition,
                **{key: value for key, value in metrics.items() if isinstance(value, (int, float))},
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)

    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "warning": (
            "Pilot output only; small cells are engineering checks, not population-level evidence."
        ),
        "data": {
            "n_fitting_students": fitting.n_students,
            "n_holdout_students": holdout.n_students,
            "n_items": dataset.n_items,
            "n_attributes": dataset.n_attributes,
        },
        "design": {
            "persona": persona_kind,
            "state": state,
            "temperature": temperature,
            "prompt_strategy": prompt_strategy,
            "n_virtual_students": n_students,
            "dina_n_starts": dina_n_starts,
            "dina_max_iter": dina_max_iter,
            "baselines": list(baseline_names),
            "noise_repeats": noise_repeats,
            "seed": seed,
            "max_calls": max_calls,
            "request_workers": max_workers,
        },
        "configuration_sha256": configuration_sha256,
        "simulator": simulator.metadata(),
        "dina_fit": {
            "training_log_likelihood": dina.training_log_likelihood_,
            "best_start": dina.best_start_,
            "n_iter": dina.n_iter_,
            "converged": dina.converged_,
            "fit_history": dina.fit_history_,
        },
        "noise_delta_95": delta,
        "results": results,
        "inference_note": (
            "Delta is a descriptive reference here. Formal three-way decisions additionally "
            "require the preregistered bootstrap CI and repeated splits."
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as output:
        json.dump(_jsonable(summary), output, ensure_ascii=False, indent=2, allow_nan=False)
    return summary


def _write_responses(responses: np.ndarray, path: Path) -> None:
    frame = pd.DataFrame(responses.astype(int))
    frame.index.name = "virtual_student"
    frame.to_csv(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value
