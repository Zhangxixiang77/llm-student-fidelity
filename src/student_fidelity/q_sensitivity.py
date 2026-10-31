"""Pre-specified Q-matrix aggregation sensitivity analyses.

These candidates are fixed content-level transformations, not a search over
item entries. The original matrix remains primary regardless of held-out fit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from student_fidelity.baselines import GDINABaseline
from student_fidelity.data import ResponseDataset
from student_fidelity.model_comparison import _paired_bootstrap_ci

QVariantSpec = Mapping[str, Any]

FRCSUB_Q_VARIANT_SPECS: tuple[QVariantSpec, ...] = (
    {"name": "original_8", "merges": ()},
    {
        "name": "mixed_number_7",
        "merges": (
            {
                "attributes": (
                    "convert_whole_to_fraction",
                    "separate_whole_from_fraction",
                ),
                "new_attribute_id": "mixed_number_handling",
            },
        ),
    },
    {
        "name": "simplification_7",
        "merges": (
            {
                "attributes": (
                    "simplify_before_subtracting",
                    "reduce_to_simplest_form",
                ),
                "new_attribute_id": "simplification",
            },
        ),
    },
    {
        "name": "borrowing_7",
        "merges": (
            {
                "attributes": ("borrow_from_whole_part", "column_borrow"),
                "new_attribute_id": "borrowing",
            },
        ),
    },
    {
        "name": "combined_5",
        "merges": (
            {
                "attributes": (
                    "convert_whole_to_fraction",
                    "separate_whole_from_fraction",
                ),
                "new_attribute_id": "mixed_number_handling",
            },
            {
                "attributes": (
                    "simplify_before_subtracting",
                    "reduce_to_simplest_form",
                ),
                "new_attribute_id": "simplification",
            },
            {
                "attributes": ("borrow_from_whole_part", "column_borrow"),
                "new_attribute_id": "borrowing",
            },
        ),
    },
)


def merge_attribute_groups(
    dataset: ResponseDataset,
    merges: Sequence[Mapping[str, Any]],
) -> ResponseDataset:
    """OR-combine disjoint named attributes while preserving item/student order."""

    attribute_to_group: dict[str, int] = {}
    group_attributes: list[tuple[str, ...]] = []
    group_ids: list[str] = []
    for group_index, merge in enumerate(merges):
        attributes = tuple(str(value) for value in merge["attributes"])
        new_attribute_id = str(merge["new_attribute_id"])
        if len(attributes) < 2 or len(set(attributes)) != len(attributes):
            raise ValueError("each merge must contain at least two distinct attributes")
        missing = set(attributes) - set(dataset.attribute_ids)
        if missing:
            raise ValueError(f"unknown attributes in Q-matrix merge: {sorted(missing)}")
        for attribute in attributes:
            if attribute in attribute_to_group:
                raise ValueError("an attribute cannot occur in multiple merge groups")
            attribute_to_group[attribute] = group_index
        group_attributes.append(attributes)
        group_ids.append(new_attribute_id)
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("new attribute ids must be unique")

    columns = []
    attribute_ids = []
    emitted_groups: set[int] = set()
    for column, attribute in enumerate(dataset.attribute_ids):
        group_index = attribute_to_group.get(attribute)
        if group_index is None:
            columns.append(dataset.q_matrix[:, column])
            attribute_ids.append(attribute)
        elif group_index not in emitted_groups:
            locations = [
                dataset.attribute_ids.index(value)
                for value in group_attributes[group_index]
            ]
            columns.append(dataset.q_matrix[:, locations].max(axis=1))
            attribute_ids.append(group_ids[group_index])
            emitted_groups.add(group_index)
    q_matrix = np.column_stack(columns).astype(np.int8)
    if len(set(attribute_ids)) != len(attribute_ids):
        raise ValueError("merged attribute ids collide with existing ids")
    return ResponseDataset(
        responses=dataset.responses.copy(),
        q_matrix=q_matrix,
        item_ids=dataset.item_ids,
        student_ids=dataset.student_ids,
        attribute_ids=attribute_ids,
        item_texts=dataset.item_texts,
        answer_keys=dataset.answer_keys,
    )


def q_matrix_sensitivity(
    fitting: ResponseDataset,
    holdout: ResponseDataset,
    n_boot: int = 500,
    seed: int = 42,
    specs: Sequence[QVariantSpec] | None = None,
    original_model: GDINABaseline | None = None,
) -> dict[str, Any]:
    """Evaluate fixed Q aggregations on the same fitting and holdout students.

    Paired intervals describe each variant's mean held-out log-likelihood
    difference from the original matrix. No automatic model/Q selection is made.
    """

    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    if fitting.attribute_ids != holdout.attribute_ids:
        raise ValueError("fitting and holdout attribute ids must match")
    if fitting.item_ids != holdout.item_ids:
        raise ValueError("fitting and holdout item ids must match")
    specs = FRCSUB_Q_VARIANT_SPECS if specs is None else specs
    if not specs or specs[0].get("merges"):
        raise ValueError("the first Q-variant spec must be the unmerged original matrix")

    original_fitting = merge_attribute_groups(fitting, specs[0]["merges"])
    original_holdout = merge_attribute_groups(holdout, specs[0]["merges"])
    if original_model is None:
        original_model = GDINABaseline().fit(original_fitting)
    elif not np.array_equal(original_model.q_matrix_, original_fitting.q_matrix):
        raise ValueError("original_model does not match the original Q matrix")
    original_log_likelihood = original_model.log_likelihood_by_student(
        original_holdout.responses
    )

    candidates = []
    for index, spec in enumerate(specs):
        if index == 0:
            variant_fitting = original_fitting
            variant_holdout = original_holdout
            model = original_model
            holdout_log_likelihood = original_log_likelihood
        else:
            variant_fitting = merge_attribute_groups(fitting, spec["merges"])
            variant_holdout = merge_attribute_groups(holdout, spec["merges"])
            model = GDINABaseline().fit(variant_fitting)
            holdout_log_likelihood = model.log_likelihood_by_student(
                variant_holdout.responses
            )
        difference = holdout_log_likelihood - original_log_likelihood
        if index == 0:
            interval = [0.0, 0.0]
        else:
            lower, upper = _paired_bootstrap_ci(
                difference, n_boot=n_boot, seed=seed + index
            )
            interval = [lower, upper]
        candidates.append(
            {
                "name": str(spec["name"]),
                "merges": _jsonable_merges(spec["merges"]),
                "n_attributes": variant_fitting.n_attributes,
                "attribute_ids": list(variant_fitting.attribute_ids),
                "training_log_likelihood": float(
                    model.log_likelihood_by_student(variant_fitting.responses).sum()
                ),
                "mean_holdout_log_likelihood": float(holdout_log_likelihood.mean()),
                "minus_original_mean_holdout_log_likelihood": float(difference.mean()),
                "paired_student_bootstrap_95_ci": interval,
                "n_bootstrap": 0 if index == 0 else n_boot,
                "holdout_srmsr": model.srmsr(variant_holdout.responses),
                "nominal_n_parameters": model.nominal_n_parameters_,
                "convergence": {
                    "converged": model.converged_,
                    "n_iter": model.n_iter_,
                    "max_iter": model.max_iter,
                    "mean_student_log_likelihood_tolerance": model.tolerance,
                    "ridge": model.ridge,
                },
            }
        )

    return {
        "status": "pre_specified_sensitivity_not_model_selection",
        "primary_q_matrix": str(specs[0]["name"]),
        "n_fitting_students": fitting.n_students,
        "n_holdout_students": holdout.n_students,
        "bootstrap_seed": seed,
        "candidates": candidates,
        "interpretation": (
            "Attribute groups are OR-combined before fitting; no item Q entry is edited. "
            "All candidates use the same students. Positive log-likelihood differences "
            "favor the aggregated variant, but results do not replace the primary matrix "
            "or certify substantive Q validity. No multiplicity-adjusted confirmatory "
            "Q selection is performed."
        ),
    }


def _jsonable_merges(merges: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "attributes": list(merge["attributes"]),
            "new_attribute_id": str(merge["new_attribute_id"]),
        }
        for merge in merges
    ]
