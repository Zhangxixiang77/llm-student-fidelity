"""Persona definitions kept separate from fidelity evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Persona:
    kind: str
    alpha: tuple[int, ...] | None = None
    theta: float | None = None
    attribute_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"none", "ability", "attribute"}:
            raise ValueError("persona kind must be none, ability, or attribute")
        if self.kind == "attribute" and self.alpha is None:
            raise ValueError("attribute personas require alpha")
        if self.kind == "ability" and self.theta is None:
            raise ValueError("ability personas require theta")

    def render(self) -> str:
        if self.kind == "none":
            return "Simulate one student. Do not assume the student is perfect."
        if self.kind == "ability":
            ability = f"{self.theta:.3f}"
            return (
                f"Simulate a student whose estimated overall mathematics ability is {ability} "
                "on a standardized scale. Stay in character, including plausible mistakes."
            )
        names = self.attribute_ids or tuple(
            f"attribute_{i + 1}" for i in range(len(self.alpha or ()))
        )
        lines = ["Simulate a student with exactly this skill-mastery profile:"]
        for name, mastered in zip(names, self.alpha or (), strict=True):
            status = "mastered" if mastered else "NOT mastered"
            lines.append(f"- {name}: {status}")
        lines.append("Stay in character, including mistakes implied by skills not mastered.")
        return "\n".join(lines)


def attribute_personas(
    alpha: NDArray[np.int_], attribute_ids: Sequence[str]
) -> list[Persona]:
    alpha = np.asarray(alpha, dtype=int)
    return [
        Persona(
            kind="attribute",
            alpha=tuple(int(value) for value in row),
            attribute_ids=tuple(attribute_ids),
        )
        for row in alpha
    ]
