"""Literature-motivated item-exclusion sensitivity without changing the primary test."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from student_fidelity.baselines import GDINABaseline
from student_fidelity.data import ResponseDataset, train_test_split
from student_fidelity.item_fit import item_pair_fit
from student_fidelity.repeated_splits import STATISTICS, _distribution_summary


def exclude_items(
    dataset: ResponseDataset, excluded_item_ids: Sequence[str]
) -> ResponseDataset:
    """Return a dataset with complete item columns removed, preserving all students."""

    excluded = set(excluded_item_ids)
    missing = excluded - set(dataset.item_ids)
    if missing:
        raise ValueError(f"unknown items requested for exclusion: {sorted(missing)}")
    keep = [index for index, item in enumerate(dataset.item_ids) if item not in excluded]
    if len(keep) < 2:
        raise ValueError("item-pair sensitivity requires at least two retained items")
    return ResponseDataset(
        responses=dataset.responses[:, keep],
        q_matrix=dataset.q_matrix[keep],
        item_ids=[dataset.item_ids[index] for index in keep],
        student_ids=dataset.student_ids,
        attribute_ids=dataset.attribute_ids,
        item_texts=[dataset.item_texts[index] for index in keep]
        if dataset.item_texts is not None
        else None,
        answer_keys=[dataset.answer_keys[index] for index in keep]
        if dataset.answer_keys is not None
        else None,
    )


def item_exclusion_sensitivity(
    dataset: ResponseDataset,
    excluded_item_ids: Sequence[str] = ("item_008",),
    n_splits: int = 20,
    seed: int = 42,
    itemfit_simulations: int | None = 100_000,
    gdina_max_iter: int = 300,
) -> dict[str, Any]:
    """Compare full and reduced item sets on identical repeated student splits.

    Held-out log likelihood is reported per item because the item sets differ.
    SRMSR and item-pair counts are descriptive sensitivity diagnostics, not a
    rule for selecting whichever item set fits better after seeing the results.
    """

    if n_splits < 1 or gdina_max_iter < 1:
        raise ValueError("n_splits and gdina_max_iter must be positive")
    reduced_dataset = exclude_items(dataset, excluded_item_ids)
    records = []
    for split_index in range(n_splits):
        split_seed = seed + split_index
        fitting, holdout = train_test_split(dataset, test_size=0.5, seed=split_seed)
        reduced_fitting = exclude_items(fitting, excluded_item_ids)
        reduced_holdout = exclude_items(holdout, excluded_item_ids)
        full_model = GDINABaseline(max_iter=gdina_max_iter).fit(fitting)
        reduced_model = GDINABaseline(max_iter=gdina_max_iter).fit(reduced_fitting)
        full_pair_fit = item_pair_fit(
            full_model,
            holdout.responses,
            item_ids=dataset.item_ids,
            n_simulated_students=itemfit_simulations,
            seed=split_seed + 30_000,
        )
        reduced_pair_fit = item_pair_fit(
            reduced_model,
            reduced_holdout.responses,
            item_ids=reduced_dataset.item_ids,
            n_simulated_students=itemfit_simulations,
            seed=split_seed + 40_000,
        )
        record: dict[str, Any] = {
            "split_index": split_index + 1,
            "split_seed": split_seed,
            "full_converged": full_model.converged_,
            "reduced_converged": reduced_model.converged_,
            "full_n_iter": full_model.n_iter_,
            "reduced_n_iter": reduced_model.n_iter_,
            "full_holdout_log_likelihood_per_item": float(
                full_model.log_likelihood_by_student(holdout.responses).mean()
                / dataset.n_items
            ),
            "reduced_holdout_log_likelihood_per_item": float(
                reduced_model.log_likelihood_by_student(reduced_holdout.responses).mean()
                / reduced_dataset.n_items
            ),
            "full_holdout_srmsr": full_model.srmsr(holdout.responses),
            "reduced_holdout_srmsr": reduced_model.srmsr(reduced_holdout.responses),
        }
        for statistic in STATISTICS:
            record[f"full_{statistic}_n_significant_pairs"] = full_pair_fit[
                "n_test_significant_holm_0_05"
            ][statistic]
            record[f"reduced_{statistic}_n_significant_pairs"] = reduced_pair_fit[
                "n_test_significant_holm_0_05"
            ][statistic]
        records.append(record)

    summary: dict[str, Any] = {
        "n_full_converged": sum(record["full_converged"] for record in records),
        "n_reduced_converged": sum(record["reduced_converged"] for record in records),
        "n_splits_reduced_srmsr_lower": sum(
            record["reduced_holdout_srmsr"] < record["full_holdout_srmsr"]
            for record in records
        ),
        "full_holdout_srmsr": _distribution_summary(
            [record["full_holdout_srmsr"] for record in records]
        ),
        "reduced_holdout_srmsr": _distribution_summary(
            [record["reduced_holdout_srmsr"] for record in records]
        ),
    }
    for statistic in STATISTICS:
        summary[statistic] = {
            "full_n_significant_pairs": _distribution_summary(
                [record[f"full_{statistic}_n_significant_pairs"] for record in records]
            ),
            "reduced_n_significant_pairs": _distribution_summary(
                [
                    record[f"reduced_{statistic}_n_significant_pairs"]
                    for record in records
                ]
            ),
            "n_splits_reduced_has_fewer_significant_pairs": sum(
                record[f"reduced_{statistic}_n_significant_pairs"]
                < record[f"full_{statistic}_n_significant_pairs"]
                for record in records
            ),
        }

    return {
        "method": "repeated_split_item_exclusion_sensitivity",
        "primary_analysis": "all_items_retained",
        "excluded_item_ids": list(excluded_item_ids),
        "n_full_items": dataset.n_items,
        "n_reduced_items": reduced_dataset.n_items,
        "n_splits": n_splits,
        "split_seed_start": seed,
        "itemfit_simulations_per_model_and_split": itemfit_simulations,
        "gdina_max_iter": gdina_max_iter,
        "summary": summary,
        "splits": records,
        "interpretation": (
            "The full item set remains primary. The reduced analysis is reported only "
            "as a literature-motivated sensitivity. Per-item likelihood, SRMSR, and "
            "pair counts across different item sets do not constitute an automatic "
            "item-selection rule or substantive validation of the Q matrix."
        ),
    }
