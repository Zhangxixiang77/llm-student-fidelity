"""Parametric-bootstrap recovery checks for the Python DINA and G-DINA estimators.

The generating parameters are estimated from real responses once. Simulated data
are then refit with the same implementation, under the correctly specified model.
These checks validate implementation and sampling behavior, not real-data model fit.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import pearsonr

from student_fidelity.baselines import DINABaseline, GDINABaseline
from student_fidelity.data import ResponseDataset
from student_fidelity.item_fit import item_pair_fit

ModelName = Literal["dina", "gdina"]


def fit_generating_model(
    dataset: ResponseDataset,
    model_name: ModelName,
    *,
    dina_starts: int = 30,
    dina_max_iter: int = 300,
    gdina_max_iter: int = 300,
    seed: int = 20261031,
) -> DINABaseline | GDINABaseline:
    """Estimate the parameters used only as the simulation's generating truth."""

    if model_name == "dina":
        return DINABaseline(
            max_iter=dina_max_iter, n_starts=dina_starts, seed=seed
        ).fit(dataset)
    if model_name == "gdina":
        return GDINABaseline(max_iter=gdina_max_iter).fit(dataset)
    raise ValueError("model_name must be 'dina' or 'gdina'")


def simulate_dataset(
    generating_model: DINABaseline | GDINABaseline,
    template: ResponseDataset,
    n_students: int,
    seed: int,
) -> ResponseDataset:
    """Generate responses while retaining the template's Q matrix and identifiers."""

    if n_students < 4:
        raise ValueError("recovery simulations require at least four students")
    responses = generating_model.sample(n_students, seed=seed)
    return ResponseDataset(
        responses=responses,
        q_matrix=template.q_matrix.copy(),
        item_ids=list(template.item_ids),
        student_ids=[f"simulated_{index + 1:04d}" for index in range(n_students)],
        attribute_ids=list(template.attribute_ids),
        item_texts=[""] * template.n_items,
        answer_keys=["correct"] * template.n_items,
    )


def _correlation(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(pearsonr(x, y).statistic)


def _error_metrics(
    estimate: NDArray[np.float64], truth: NDArray[np.float64]
) -> dict[str, float]:
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    error = estimate - truth
    return {
        "bias": float(error.mean()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "correlation": _correlation(estimate, truth),
    }


def _prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _item_fit_recovery(
    fitted: DINABaseline | GDINABaseline,
    simulated: ResponseDataset,
    *,
    seed: int,
    simulations: int,
) -> dict[str, Any]:
    fit = item_pair_fit(
        fitted,
        simulated.responses,
        item_ids=simulated.item_ids,
        n_simulated_students=simulations,
        seed=seed,
    )
    result: dict[str, Any] = {
        "itemfit_n_pairs_with_small_expected_cells": fit[
            "n_pairs_with_small_expected_cells"
        ],
        "itemfit_n_pairs_with_zero_cell_correction": fit[
            "n_pairs_with_zero_cell_correction"
        ],
    }
    n_pairs = len(fit["item_pairs"])
    n_items = len(fit["item_level"])
    for statistic in ("correlation", "log_odds_ratio"):
        short = "cor" if statistic == "correlation" else "lor"
        raw = np.array(
            [pair[f"{statistic}_p"] < 0.05 for pair in fit["item_pairs"]], dtype=float
        )
        test = np.array(
            [pair[f"{statistic}_holm_test_p"] < 0.05 for pair in fit["item_pairs"]],
            dtype=float,
        )
        item = np.array(
            [row[f"{statistic}_holm_item_p"] < 0.05 for row in fit["item_level"]],
            dtype=float,
        )
        result[f"itemfit_{short}_raw_any_rejection"] = bool(raw.any())
        result[f"itemfit_{short}_raw_pair_rejection_rate"] = float(raw.mean())
        result[f"itemfit_{short}_holm_test_any_rejection"] = bool(test.any())
        result[f"itemfit_{short}_holm_test_pair_rejection_rate"] = float(test.mean())
        result[f"itemfit_{short}_holm_item_rejection_rate"] = float(item.mean())
        result[f"itemfit_{short}_n_pairs"] = n_pairs
        result[f"itemfit_{short}_n_items"] = n_items
    return result


def recovery_replication(
    generating_model: DINABaseline | GDINABaseline,
    template: ResponseDataset,
    model_name: ModelName,
    n_students: int,
    replication: int,
    *,
    seed: int,
    dina_starts: int = 10,
    dina_max_iter: int = 300,
    gdina_max_iter: int = 300,
    itemfit_simulations: int | None = None,
) -> dict[str, Any]:
    """Simulate and refit one correctly specified data set."""

    simulated = simulate_dataset(generating_model, template, n_students, seed)
    if model_name == "dina":
        if not isinstance(generating_model, DINABaseline):
            raise TypeError("DINA recovery requires a DINA generating model")
        fitted: DINABaseline | GDINABaseline = DINABaseline(
            max_iter=dina_max_iter, n_starts=dina_starts, seed=seed + 100
        ).fit(simulated)
        record: dict[str, Any] = {
            **_prefixed("guess", _error_metrics(fitted.guess_, generating_model.guess_)),
            **_prefixed("slip", _error_metrics(fitted.slip_, generating_model.slip_)),
            "best_start": fitted.best_start_,
            "n_converged_starts": sum(
                start["converged"] for start in fitted.fit_history_
            ),
            "n_starts": fitted.n_starts,
            "estimated_guess": fitted.guess_.tolist(),
            "estimated_slip": fitted.slip_.tolist(),
        }
    elif model_name == "gdina":
        if not isinstance(generating_model, GDINABaseline):
            raise TypeError("G-DINA recovery requires a G-DINA generating model")
        fitted = GDINABaseline(max_iter=gdina_max_iter).fit(simulated)
        p_error = _error_metrics(fitted.p_correct_, generating_model.p_correct_)
        weights = generating_model.pattern_prior_[:, None]
        weighted_error = (fitted.p_correct_ - generating_model.p_correct_) * np.sqrt(
            weights
        )
        record = {
            **_prefixed("p_correct", p_error),
            "p_correct_prior_weighted_rmse": float(
                np.sqrt(np.sum(weighted_error**2) / template.n_items)
            ),
            "n_starts": 1,
            "n_converged_starts": int(fitted.converged_),
        }
    else:
        raise ValueError("model_name must be 'dina' or 'gdina'")

    true_prior = generating_model.pattern_prior_
    estimated_prior = fitted.pattern_prior_
    record.update(
        {
            "model": model_name,
            "n_students": n_students,
            "replication": replication,
            "seed": seed,
            "converged": fitted.converged_,
            "n_iter": fitted.n_iter_,
            "training_log_likelihood": fitted.training_log_likelihood_,
            "srmsr": fitted.srmsr(simulated.responses),
            "pattern_prior_total_variation": float(
                0.5 * np.abs(estimated_prior - true_prior).sum()
            ),
            "pattern_prior_rmse": float(np.sqrt(np.mean((estimated_prior - true_prior) ** 2))),
        }
    )
    if itemfit_simulations is not None:
        if itemfit_simulations < 1:
            raise ValueError("itemfit_simulations must be positive")
        record.update(
            _item_fit_recovery(
                fitted, simulated, seed=seed + 200, simulations=itemfit_simulations
            )
        )
    return record


def _mean_sd(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = np.array([record[key] for record in records], dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "n_finite": int(finite.size),
        "mean": float(finite.mean()),
        "sd": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "median": float(np.median(finite)),
        "q05": float(np.quantile(finite, 0.05)),
        "q95": float(np.quantile(finite, 0.95)),
        "monte_carlo_se_mean": (
            float(finite.std(ddof=1) / np.sqrt(finite.size)) if finite.size > 1 else 0.0
        ),
    }


def _proportion(
    records: list[dict[str, Any]], key: str, *, binomial: bool = False
) -> dict[str, Any]:
    values = np.array([record[key] for record in records], dtype=float)
    estimate = float(values.mean())
    n = values.size
    if binomial:
        # Familywise any-rejection indicators are independent across simulated data sets.
        se = float(np.sqrt(estimate * (1 - estimate) / n))
        z = 1.959963984540054
        denominator = 1 + z**2 / n
        center = (estimate + z**2 / (2 * n)) / denominator
        half_width = (
            z * np.sqrt(estimate * (1 - estimate) / n + z**2 / (4 * n**2)) / denominator
        )
        return {
            "estimate": estimate,
            "monte_carlo_se": se,
            "monte_carlo_ci95_wilson": [max(0.0, center - half_width), min(1.0, center + half_width)],
            "n_replications": int(n),
        }
    # Pair/item rejection rates are first averaged within each replicate; the
    # replicate-level interval does not pretend pairs within a data set are independent.
    se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {
        "estimate": estimate,
        "monte_carlo_se": se,
        "monte_carlo_ci95_normal": [max(0.0, estimate - 1.96 * se), min(1.0, estimate + 1.96 * se)],
        "n_replications": int(n),
    }


def summarize_recovery(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one model-by-sample-size cell, including Monte Carlo uncertainty."""

    if not records:
        raise ValueError("at least one recovery replication is required")
    keys = {
        "model", "n_students", "replication", "seed", "estimated_guess", "estimated_slip"
    }
    metric_keys = [
        key
        for key, value in records[0].items()
        if key not in keys
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not key.startswith("itemfit_")
        and key not in {"best_start", "n_starts", "n_converged_starts", "n_iter"}
    ]
    summary: dict[str, Any] = {
        "model": records[0]["model"],
        "n_students": records[0]["n_students"],
        "n_replications": len(records),
        "convergence_rate": float(np.mean([record["converged"] for record in records])),
        "metrics": {key: _mean_sd(records, key) for key in metric_keys},
        "rejection_rates": {},
    }
    summary["itemfit_diagnostics"] = {
        key: _mean_sd(records, key)
        for key in records[0]
        if key.startswith("itemfit_")
        and (key.endswith("small_expected_cells") or key.endswith("zero_cell_correction"))
    }
    for key in records[0]:
        if key.startswith("itemfit_") and key.endswith("any_rejection"):
            summary["rejection_rates"][key] = _proportion(records, key, binomial=True)
        elif key.startswith("itemfit_") and key.endswith("rejection_rate"):
            summary["rejection_rates"][key] = _proportion(records, key)
    if records[0]["model"] == "dina":
        summary["multistart"] = {
            "all_starts_converged_rate": float(
                np.mean(
                    [r["n_converged_starts"] == r["n_starts"] for r in records]
                )
            ),
            "mean_converged_starts": float(
                np.mean([r["n_converged_starts"] for r in records])
            ),
        }
    return summary


def dina_item_recovery(
    records: list[dict[str, Any]],
    generating_model: DINABaseline,
    item_ids: list[str],
) -> list[dict[str, Any]]:
    """Item-level DINA bias and RMSE across replications for appendix tables."""

    if not records or any(record["model"] != "dina" for record in records):
        raise ValueError("DINA recovery records are required")
    estimated_guess = np.array([record["estimated_guess"] for record in records])
    estimated_slip = np.array([record["estimated_slip"] for record in records])
    rows = []
    for index, item_id in enumerate(item_ids):
        row: dict[str, Any] = {"item_id": item_id}
        for name, estimates, truth in (
            ("guess", estimated_guess[:, index], generating_model.guess_[index]),
            ("slip", estimated_slip[:, index], generating_model.slip_[index]),
        ):
            row[f"{name}_true"] = float(truth)
            row[f"{name}_bias"] = float(np.mean(estimates - truth))
            row[f"{name}_mae"] = float(np.mean(np.abs(estimates - truth)))
            row[f"{name}_rmse"] = float(np.sqrt(np.mean((estimates - truth) ** 2)))
        rows.append(row)
    return rows
