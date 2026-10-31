"""Generative baselines used by the pilot pipeline."""

from student_fidelity.baselines.base import GenerativeBaseline
from student_fidelity.baselines.copula import GaussianCopulaBaseline
from student_fidelity.baselines.dina import DINABaseline
from student_fidelity.baselines.gdina import GDINABaseline
from student_fidelity.baselines.independent import IndependentBernoulliBaseline
from student_fidelity.baselines.irt import IRT2PLBaseline

BASELINES = {
    "independent": IndependentBernoulliBaseline,
    "copula": GaussianCopulaBaseline,
    "irt": IRT2PLBaseline,
    "dina": DINABaseline,
    "gdina": GDINABaseline,
}


def make_baseline(name: str) -> GenerativeBaseline:
    try:
        return BASELINES[name]()
    except KeyError as error:
        raise ValueError(f"unknown baseline {name!r}; choose from {sorted(BASELINES)}") from error


__all__ = [
    "BASELINES",
    "DINABaseline",
    "GDINABaseline",
    "GaussianCopulaBaseline",
    "GenerativeBaseline",
    "IRT2PLBaseline",
    "IndependentBernoulliBaseline",
    "make_baseline",
]
