"""Repeated human-data split diagnostics for CDM choice and localized misfit."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from student_fidelity.baselines import DINABaseline, GDINABaseline
from student_fidelity.data import ResponseDataset, train_test_split
from student_fidelity.item_fit import item_pair_fit
from student_fidelity.model_comparison import compare_cdm_fits

STATISTICS = ("correlation", "log_odds_ratio")


def repeated_split_diagnostics(
    dataset: ResponseDataset,
    n_splits: int = 20,
    seed: int = 42,
    comparison_bootstraps: int = 200,
    itemfit_simulations: int | None = None,
    gdina_max_iter: int = 300,
    dina_max_iter: int = 300,
    dina_n_starts: int = 10,
    focus_item_id: str = "item_008",
    focus_pair: tuple[str, str] = ("item_004", "item_008"),
) -> dict[str, Any]:
    """Repeat student-level splits without refitting on each holdout half.

    DINA and G-DINA are compared on every split. Item-pair tests are run for
    G-DINA, the provisional dimension-B model. The output summarizes whether
    misfit around a pre-specified item and pair recurs across splits; it does
    not turn repeated diagnostics into a new confirmatory significance test.
    """

    if (
        n_splits < 1
        or comparison_bootstraps < 1
        or gdina_max_iter < 1
        or dina_max_iter < 1
        or dina_n_starts < 1
    ):
        raise ValueError("split, bootstrap, iteration, and start counts must be positive")
    if focus_item_id not in dataset.item_ids:
        raise ValueError("focus_item_id is absent from the dataset")
    if len(set(focus_pair)) != 2 or not set(focus_pair).issubset(dataset.item_ids):
        raise ValueError("focus_pair must contain two distinct dataset items")

    split_records = []
    recurrent: dict[str, Counter[tuple[str, str]]] = {
        statistic: Counter() for statistic in STATISTICS
    }
    for split_index in range(n_splits):
        split_seed = seed + split_index
        fitting, holdout = train_test_split(dataset, test_size=0.5, seed=split_seed)
        dina = DINABaseline(
            max_iter=dina_max_iter, n_starts=dina_n_starts, seed=split_seed + 500
        ).fit(fitting)
        gdina = GDINABaseline(max_iter=gdina_max_iter).fit(fitting)
        comparison = compare_cdm_fits(
            dina,
            gdina,
            fitting,
            holdout,
            n_boot=comparison_bootstraps,
            seed=split_seed + 10_000,
        )
        pair_fit = item_pair_fit(
            gdina,
            holdout.responses,
            item_ids=dataset.item_ids,
            n_simulated_students=itemfit_simulations,
            seed=split_seed + 20_000,
        )
        record: dict[str, Any] = {
            "split_index": split_index + 1,
            "split_seed": split_seed,
            "n_fitting_students": fitting.n_students,
            "n_holdout_students": holdout.n_students,
            "dina_mean_holdout_log_likelihood": comparison["dina"][
                "mean_holdout_log_likelihood"
            ],
            "gdina_mean_holdout_log_likelihood": comparison["gdina"][
                "mean_holdout_log_likelihood"
            ],
            "gdina_minus_dina_mean_holdout_log_likelihood": comparison[
                "gdina_minus_dina_mean_holdout_log_likelihood"
            ],
            "paired_student_bootstrap_95_ci": comparison[
                "paired_student_bootstrap_95_ci"
            ],
            "provisional_dimension_b_model": comparison[
                "provisional_dimension_b_model"
            ],
            "dina_holdout_srmsr": comparison["dina"]["holdout_srmsr"],
            "gdina_holdout_srmsr": comparison["gdina"]["holdout_srmsr"],
            "gdina_converged": comparison["gdina"]["convergence"]["converged"],
            "gdina_n_iter": comparison["gdina"]["convergence"]["n_iter"],
            "dina_converged": dina.converged_,
            "dina_n_starts": dina.n_starts,
            "dina_n_converged_starts": sum(
                start["converged"] for start in dina.fit_history_
            ),
            "dina_best_start": dina.best_start_,
            "dina_multistart_training_log_likelihood_gain": float(
                dina.training_log_likelihood_
                - dina.fit_history_[0]["training_log_likelihood"]
            ),
        }
        for statistic in STATISTICS:
            significant = [
                pair
                for pair in pair_fit["item_pairs"]
                if pair[f"{statistic}_holm_test_p"] <= 0.05
            ]
            focus_significant = [
                pair
                for pair in significant
                if focus_item_id in (pair["item_1"], pair["item_2"])
            ]
            target_significant = any(
                {pair["item_1"], pair["item_2"]} == set(focus_pair)
                for pair in significant
            )
            record[f"{statistic}_n_significant_pairs"] = len(significant)
            record[f"{statistic}_n_focus_item_significant_pairs"] = len(
                focus_significant
            )
            record[f"{statistic}_focus_item_share"] = (
                len(focus_significant) / len(significant) if significant else None
            )
            record[f"{statistic}_focus_pair_significant"] = target_significant
            for pair in significant:
                recurrent[statistic][(pair["item_1"], pair["item_2"])] += 1
        split_records.append(record)

    model_summary = {
        "n_gdina_holdout_better": sum(
            record["gdina_minus_dina_mean_holdout_log_likelihood"] > 0
            for record in split_records
        ),
        "n_gdina_provisional_selected": sum(
            record["provisional_dimension_b_model"] == "gdina"
            for record in split_records
        ),
        "n_gdina_converged": sum(record["gdina_converged"] for record in split_records),
        "n_dina_best_start_converged": sum(
            record["dina_converged"] for record in split_records
        ),
        "n_dina_multistart_improved_training_likelihood": sum(
            record["dina_multistart_training_log_likelihood_gain"] > 1e-8
            for record in split_records
        ),
        "n_dina_best_start_not_deterministic": sum(
            record["dina_best_start"] != 0 for record in split_records
        ),
        "dina_multistart_training_log_likelihood_gain": _distribution_summary(
            [
                record["dina_multistart_training_log_likelihood_gain"]
                for record in split_records
            ]
        ),
        "n_gdina_srmsr_at_most_0_05": sum(
            record["gdina_holdout_srmsr"] <= 0.05 for record in split_records
        ),
        "gdina_minus_dina_mean_holdout_log_likelihood": _distribution_summary(
            [
                record["gdina_minus_dina_mean_holdout_log_likelihood"]
                for record in split_records
            ]
        ),
        "gdina_holdout_srmsr": _distribution_summary(
            [record["gdina_holdout_srmsr"] for record in split_records]
        ),
    }
    focus_summary = {}
    for statistic in STATISTICS:
        shares = [
            record[f"{statistic}_focus_item_share"]
            for record in split_records
            if record[f"{statistic}_focus_item_share"] is not None
        ]
        focus_summary[statistic] = {
            "n_significant_pairs": _distribution_summary(
                [record[f"{statistic}_n_significant_pairs"] for record in split_records]
            ),
            "n_splits_with_any_significant_pairs": sum(
                record[f"{statistic}_n_significant_pairs"] > 0
                for record in split_records
            ),
            "n_splits_with_focus_item_significant": sum(
                record[f"{statistic}_n_focus_item_significant_pairs"] > 0
                for record in split_records
            ),
            "n_splits_with_focus_pair_significant": sum(
                record[f"{statistic}_focus_pair_significant"]
                for record in split_records
            ),
            "focus_item_share_when_any_significant": _distribution_summary(shares),
            "recurrent_significant_pairs": [
                {
                    "item_1": pair[0],
                    "item_2": pair[1],
                    "n_splits": count,
                    "proportion_splits": count / n_splits,
                }
                for pair, count in sorted(
                    recurrent[statistic].items(),
                    key=lambda entry: (-entry[1], entry[0]),
                )
            ],
        }

    return {
        "method": "repeated_student_level_split_diagnostics",
        "n_splits": n_splits,
        "split_seed_start": seed,
        "comparison_bootstraps_per_split": comparison_bootstraps,
        "itemfit_simulations_per_split": itemfit_simulations,
        "gdina_max_iter": gdina_max_iter,
        "dina_max_iter": dina_max_iter,
        "dina_n_starts": dina_n_starts,
        "focus_item_id": focus_item_id,
        "focus_pair": list(focus_pair),
        "model_comparison_summary": model_summary,
        "focus_item_stability": focus_summary,
        "splits": split_records,
        "interpretation": (
            "Every split refits on one human half and evaluates on the other. "
            "Recurrence describes robustness to splitting, not a new confirmatory "
            "p-value. The focus item and pair remain in the primary analysis; "
            "no item is deleted or downweighted by this diagnostic."
        ),
    }


def _distribution_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "max": None, "mean": None}
    array = np.asarray(values, dtype=float)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }
