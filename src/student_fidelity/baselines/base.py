"""Common interface for fit-on-training-half generative baselines."""

from __future__ import annotations

from typing import Protocol, Self

import numpy as np
from numpy.typing import NDArray

from student_fidelity.data import ResponseDataset


class GenerativeBaseline(Protocol):
    def fit(self, dataset: ResponseDataset) -> Self: ...

    def sample(self, n_students: int, seed: int) -> NDArray[np.int8]: ...
