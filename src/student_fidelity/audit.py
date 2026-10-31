"""Offline pilot audit: truncation sensitivity and CDM gate diagnostics."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from student_fidelity.baselines import DINABaseline, GDINABaseline
from student_fidelity.data import ResponseDataset, train_test_split
from student_fidelity.experiment import _jsonable
from student_fidelity.item_fit import item_pair_fit
from student_fidelity.metrics import evaluate_structure, implicit_slip_guess, readback_agreement
from student_fidelity.model_comparison import compare_cdm_fits
from student_fidelity.q_sensitivity import (
    FRCSUB_Q_VARIANT_SPECS,
    q_matrix_sensitivity,
)


def audit_run(
    dataset: ResponseDataset,
    run_dir: str | Path,
    ceiling_repeats: int = 100,
    comparison_bootstraps: int = 500,
    itemfit_simulations: int | None = None,
) -> dict[str, Any]:
    """Analyze stored responses without making model calls or treating small pilots as inference."""

    if ceiling_repeats < 1:
        raise ValueError("ceiling_repeats must be positive")
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (run_dir / "raw_simulation.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    scored = [record for record in records if isinstance(record.get("is_correct"), bool)]
    n_students = int(summary["design"]["n_virtual_students"])
    responses, target_alpha = _load_scored_responses(dataset, run_dir, scored, n_students)
    seed = int(summary["design"]["seed"])
    fitting, holdout = train_test_split(dataset, test_size=0.5, seed=seed)
    dina = DINABaseline().fit(fitting)
    gdina = GDINABaseline().fit(fitting)
    fit_comparison = compare_cdm_fits(
        dina,
        gdina,
        fitting,
        holdout,
        n_boot=comparison_bootstraps,
        seed=seed + 5,
    )
    pair_fit = {
        "dina": item_pair_fit(
            dina,
            holdout.responses,
            item_ids=dataset.item_ids,
            n_simulated_students=itemfit_simulations,
            seed=seed + 6,
        ),
        "gdina": item_pair_fit(
            gdina,
            holdout.responses,
            item_ids=dataset.item_ids,
            n_simulated_students=itemfit_simulations,
            seed=seed + 7,
        ),
    }
    required_attributes = {
        attribute
        for spec in FRCSUB_Q_VARIANT_SPECS
        for merge in spec["merges"]
        for attribute in merge["attributes"]
    }
    q_sensitivity = None
    if required_attributes.issubset(dataset.attribute_ids):
        q_sensitivity = q_matrix_sensitivity(
            fitting,
            holdout,
            n_boot=comparison_bootstraps,
            seed=seed + 8,
            original_model=gdina,
        )

    def evaluate_rows(rows: list[int]) -> dict[str, Any]:
        selected = responses[rows]
        result: dict[str, Any] = evaluate_structure(holdout.responses, selected)
        if summary["design"]["persona"] == "attribute":
            alpha = target_alpha[rows]
            estimated = dina.map_patterns(selected)
            result.update(readback_agreement(alpha, estimated))
            result.update(implicit_slip_guess(selected, alpha, dataset.q_matrix))
            gdina_estimated = gdina.map_patterns(selected)
            gdina_readback = readback_agreement(alpha, gdina_estimated)
            result.update(
                {
                    f"gdina_{key}": value
                    for key, value in gdina_readback.items()
                }
            )
            result.update(gdina.response_residuals(selected, alpha))
        return result

    all_rows = list(range(n_students))
    truncated_ids = sorted(
        {
            record["student_id"]
            for record in scored
            if record.get("response_metadata", {}).get("finish_reason") == "length"
        }
    )
    complete_rows = [
        index for index in all_rows if f"virtual_{index + 1:04d}" not in truncated_ids
    ]
    sensitivity = None
    if truncated_ids:
        sensitivity = {
            "rule": (
                "Exclude every student with a finish_reason=length record. This is a conservative "
                "complete-student sensitivity analysis, not an item-level imputation."
            ),
            "excluded_student_ids": truncated_ids,
            "n_retained_students": len(complete_rows),
            "metrics": evaluate_rows(complete_rows) if complete_rows else None,
        }

    usage: Counter[str] = Counter()
    for record in scored:
        for key, value in record.get("response_metadata", {}).get("usage", {}).items():
            if isinstance(value, (int, float)):
                usage[key] += value
    finish_counts = Counter(
        str(record.get("response_metadata", {}).get("finish_reason", "not_recorded"))
        for record in scored
    )
    ceiling = None
    gdina_ceiling = None
    gate_references = None
    if summary["design"]["persona"] == "attribute":
        ceiling = dina.readback_ceiling(
            target_alpha, seed=seed + 4, repeats=ceiling_repeats
        )
        gdina_ceiling = gdina.readback_ceiling(
            target_alpha, seed=seed + 4, repeats=ceiling_repeats
        )
        primary = evaluate_rows(all_rows)
        human_guess = float(dina.guess_.mean())
        gate_references = {
            "warning": "Descriptive pilot references only; small cells cannot certify a gate.",
            "gate_2_attribute_min_0_80": min(ceiling["attribute_readback_by_attribute"]) >= 0.80,
            "gate_2_pattern_min_0_50": ceiling["pattern_readback_agreement"] >= 0.50,
            "gate_3_human_mean_guess": human_guess,
            "gate_3_threshold_human_plus_0_30": human_guess + 0.30,
            "gate_3_observed_ability_leakage": primary["ability_leakage_rate"],
            "gate_3_within_descriptive_bound": (
                primary["ability_leakage_rate"] <= human_guess + 0.30
            ),
        }

    result = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "configuration_sha256": summary.get("configuration_sha256"),
        "warning": (
            "Offline pilot audit only. No bootstrap inferential decision is made; the generated "
            "student count is too small for stable population claims."
        ),
        "response_audit": {
            "attempt_records": len(records),
            "scored_records": len(scored),
            "error_records": len(records) - len(scored),
            "n_virtual_students": n_students,
            "finish_reason_counts": dict(finish_counts),
            "usage_totals": dict(usage),
        },
        "primary_all_scored": evaluate_rows(all_rows),
        "length_sensitivity": sensitivity,
        "dina_fit_diagnostics": {
            "holdout_srmsr": dina.srmsr(holdout.responses),
            "srmsr_reference_0_05": 0.05,
            **dina.classification_confidence(holdout.responses),
            "fitted_mean_slip": float(dina.slip_.mean()),
            "fitted_mean_guess": float(dina.guess_.mean()),
            "note": (
                "Classification confidence is marginal-posterior self-consistency, "
                "not accuracy against an external criterion. External classification "
                "validation and repeated splits remain separate pilot requirements."
            ),
        },
        "cdm_fit_comparison": fit_comparison,
        "item_pair_fit": pair_fit,
        "q_matrix_sensitivity": q_sensitivity,
        "readback_ceiling": ceiling,
        "gdina_readback_ceiling": gdina_ceiling,
        "descriptive_gate_references": gate_references,
    }
    return _jsonable(result)


def write_audit(
    dataset: ResponseDataset,
    run_dir: str | Path,
    ceiling_repeats: int = 100,
    comparison_bootstraps: int = 500,
    itemfit_simulations: int | None = None,
) -> Path:
    run_dir = Path(run_dir)
    result = audit_run(
        dataset,
        run_dir,
        ceiling_repeats=ceiling_repeats,
        comparison_bootstraps=comparison_bootstraps,
        itemfit_simulations=itemfit_simulations,
    )
    output_path = run_dir / "audit.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _load_scored_responses(
    dataset: ResponseDataset,
    run_dir: Path,
    records: list[dict[str, Any]],
    n_students: int,
) -> tuple[np.ndarray, np.ndarray]:
    target_alpha = np.loadtxt(run_dir / "assigned_alpha.csv", delimiter=",", dtype=np.int8, ndmin=2)
    if target_alpha.shape != (n_students, dataset.n_attributes):
        raise ValueError("assigned_alpha.csv is incompatible with the run and dataset")
    responses = np.full((n_students, dataset.n_items), -1, dtype=np.int8)
    item_indices = {item_id: index for index, item_id in enumerate(dataset.item_ids)}
    for record in records:
        student_index = int(record["student_id"].rsplit("_", 1)[1]) - 1
        item_index = item_indices[record["item_id"]]
        if not 0 <= student_index < n_students:
            raise ValueError("run contains an out-of-range student")
        if responses[student_index, item_index] >= 0:
            raise ValueError("run contains duplicate scored responses")
        responses[student_index, item_index] = int(record["is_correct"])
    if (responses < 0).any():
        raise ValueError("audit requires a fully scored run")
    return responses, target_alpha
