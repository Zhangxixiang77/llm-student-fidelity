"""Structural-fidelity evaluation for LLM-simulated students."""

from student_fidelity.data import ResponseDataset, make_synthetic_cdm_dataset, train_test_split

__all__ = ["ResponseDataset", "make_synthetic_cdm_dataset", "train_test_split"]
__version__ = "0.1.0"
