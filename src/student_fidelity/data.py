"""Validated response/Q-matrix data structures and a synthetic smoke-test dataset."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class ResponseDataset:
    """Complete binary student responses aligned with an item-by-attribute Q matrix."""

    responses: NDArray[np.int8]
    q_matrix: NDArray[np.int8]
    item_ids: Sequence[str] | None = None
    student_ids: Sequence[str] | None = None
    attribute_ids: Sequence[str] | None = None
    item_texts: Sequence[str] | None = None
    answer_keys: Sequence[str] | None = None

    def __post_init__(self) -> None:
        self.responses = np.asarray(self.responses, dtype=np.int8)
        self.q_matrix = np.asarray(self.q_matrix, dtype=np.int8)
        if self.responses.ndim != 2 or self.q_matrix.ndim != 2:
            raise ValueError("responses and q_matrix must both be two-dimensional")
        if self.responses.shape[1] != self.q_matrix.shape[0]:
            raise ValueError("response columns must equal Q-matrix rows")
        if self.responses.shape[0] == 0 or self.responses.shape[1] == 0:
            raise ValueError("dataset must contain students and items")
        if not np.isin(self.responses, [0, 1]).all():
            raise ValueError("only complete binary (0/1) responses are supported")
        if not np.isin(self.q_matrix, [0, 1]).all():
            raise ValueError("Q matrix must be binary")
        if (self.q_matrix.sum(axis=1) == 0).any():
            raise ValueError("every item must require at least one attribute")

        n_students, n_items = self.responses.shape
        n_attributes = self.q_matrix.shape[1]
        self.item_ids = _ids(self.item_ids, n_items, "item")
        self.student_ids = _ids(self.student_ids, n_students, "student")
        self.attribute_ids = _ids(self.attribute_ids, n_attributes, "attribute")
        if self.item_texts is not None and len(self.item_texts) != n_items:
            raise ValueError("item_texts must align with response columns")
        if self.answer_keys is not None and len(self.answer_keys) != n_items:
            raise ValueError("answer_keys must align with response columns")

    @property
    def n_students(self) -> int:
        return self.responses.shape[0]

    @property
    def n_items(self) -> int:
        return self.responses.shape[1]

    @property
    def n_attributes(self) -> int:
        return self.q_matrix.shape[1]

    @classmethod
    def from_directory(cls, path: str | Path) -> ResponseDataset:
        """Load responses.csv, q_matrix.csv, and optional items.csv from a directory."""

        root = Path(path)
        response_frame = pd.read_csv(root / "responses.csv").set_index("student_id")
        q_frame = pd.read_csv(root / "q_matrix.csv").set_index("item_id")
        missing = set(response_frame.columns) - set(q_frame.index)
        extra = set(q_frame.index) - set(response_frame.columns)
        if missing or extra:
            raise ValueError(
                f"item mismatch: missing Q rows={sorted(missing)}, extra={sorted(extra)}"
            )
        q_frame = q_frame.loc[response_frame.columns]

        item_texts = None
        answer_keys = None
        items_path = root / "items.csv"
        if items_path.exists():
            items = pd.read_csv(items_path).set_index("item_id").loc[response_frame.columns]
            if "text" in items:
                item_texts = items["text"].fillna("").astype(str).tolist()
            if "answer_key" in items:
                answer_keys = items["answer_key"].fillna("").astype(str).tolist()

        return cls(
            responses=response_frame.to_numpy(dtype=np.int8),
            q_matrix=q_frame.to_numpy(dtype=np.int8),
            item_ids=response_frame.columns.astype(str).tolist(),
            student_ids=response_frame.index.astype(str).tolist(),
            attribute_ids=q_frame.columns.astype(str).tolist(),
            item_texts=item_texts,
            answer_keys=answer_keys,
        )

    def to_directory(self, path: str | Path) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            self.responses,
            index=pd.Index(self.student_ids, name="student_id"),
            columns=self.item_ids,
        ).to_csv(root / "responses.csv")
        pd.DataFrame(
            self.q_matrix, index=pd.Index(self.item_ids, name="item_id"), columns=self.attribute_ids
        ).to_csv(root / "q_matrix.csv")
        if self.item_texts is not None or self.answer_keys is not None:
            items = pd.DataFrame(index=pd.Index(self.item_ids, name="item_id"))
            items["text"] = self.item_texts or [""] * self.n_items
            items["answer_key"] = self.answer_keys or [""] * self.n_items
            items.to_csv(root / "items.csv")

    def subset_students(self, indices: Sequence[int] | NDArray[np.int_]) -> ResponseDataset:
        indices = np.asarray(indices, dtype=int)
        return ResponseDataset(
            responses=self.responses[indices],
            q_matrix=self.q_matrix.copy(),
            item_ids=self.item_ids,
            student_ids=[self.student_ids[i] for i in indices],
            attribute_ids=self.attribute_ids,
            item_texts=self.item_texts,
            answer_keys=self.answer_keys,
        )


def _ids(values: Sequence[str] | None, size: int, prefix: str) -> list[str]:
    if values is None:
        return [f"{prefix}_{i + 1:03d}" for i in range(size)]
    result = [str(value) for value in values]
    if len(result) != size or len(set(result)) != size:
        raise ValueError(f"{prefix} ids must be unique and match the matrix dimensions")
    return result


def train_test_split(
    dataset: ResponseDataset, test_size: float = 0.5, seed: int = 42
) -> tuple[ResponseDataset, ResponseDataset]:
    """Split students, never items; both returned sets retain the original Q matrix."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one")
    rng = np.random.default_rng(seed)
    order = rng.permutation(dataset.n_students)
    n_test = int(round(dataset.n_students * test_size))
    n_test = min(max(n_test, 1), dataset.n_students - 1)
    return dataset.subset_students(order[n_test:]), dataset.subset_students(order[:n_test])


def make_synthetic_cdm_dataset(
    n_students: int = 120, n_items: int = 20, n_attributes: int = 8, seed: int = 42
) -> ResponseDataset:
    """Create a DINA-like dataset for smoke tests only; this is not FrcSub data."""

    if n_students < 4 or n_items < 2 or n_attributes < 1:
        raise ValueError("invalid synthetic dataset dimensions")
    rng = np.random.default_rng(seed)
    alpha = rng.binomial(1, 0.55, size=(n_students, n_attributes)).astype(np.int8)
    q_matrix = np.zeros((n_items, n_attributes), dtype=np.int8)
    for item in range(n_items):
        q_matrix[item, item % n_attributes] = 1
        extras = rng.random(n_attributes) < 0.2
        q_matrix[item] = np.maximum(q_matrix[item], extras.astype(np.int8))
    eta = np.all(alpha[:, None, :] >= q_matrix[None, :, :], axis=2)
    slip = rng.uniform(0.05, 0.2, size=n_items)
    guess = rng.uniform(0.05, 0.25, size=n_items)
    p_correct = np.where(eta, 1 - slip[None, :], guess[None, :])
    responses = rng.binomial(1, p_correct).astype(np.int8)
    return ResponseDataset(
        responses=responses,
        q_matrix=q_matrix,
        item_texts=[f"Synthetic diagnostic item {i + 1}" for i in range(n_items)],
        answer_keys=["correct"] * n_items,
    )
