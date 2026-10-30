"""Longitudinal item-by-item simulation with raw-output retention."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from student_fidelity.data import ResponseDataset
from student_fidelity.simulation.personas import Persona
from student_fidelity.simulation.simulators import Item, StudentSimulator


def normalize_answer(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    return "".join(normalized.split())


def latin_square_item_order(n_items: int, student_index: int, seed: int) -> NDArray[np.int_]:
    """Cyclic Latin-square rows balance each item's position across simulated students."""

    return np.roll(np.arange(n_items), (student_index + seed) % n_items)


def run_simulation(
    dataset: ResponseDataset,
    simulator: StudentSimulator,
    personas: list[Persona],
    output_path: str | Path,
    state: str = "stateless",
    temperature: float = 0.0,
    seed: int = 42,
) -> NDArray[np.int8]:
    if state not in {"stateless", "stateful"}:
        raise ValueError("state must be stateless or stateful")
    if not personas:
        raise ValueError("at least one persona is required")
    if not 0 <= temperature <= 2:
        raise ValueError("temperature must be between zero and two")

    n_students = len(personas)
    responses = np.full((n_students, dataset.n_items), -1, dtype=np.int8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for student_index, persona in enumerate(personas):
            history: list[dict[str, Any]] = []
            order = latin_square_item_order(dataset.n_items, student_index, seed)
            for position, item_index in enumerate(order):
                text = (
                    dataset.item_texts[item_index]
                    if dataset.item_texts
                    else dataset.item_ids[item_index]
                )
                answer_key = dataset.answer_keys[item_index] if dataset.answer_keys else ""
                item = Item(
                    item_id=dataset.item_ids[item_index],
                    text=text,
                    answer_key=answer_key,
                    q_vector=dataset.q_matrix[item_index],
                    index=int(item_index),
                )
                answer = simulator.answer(persona, item, history, temperature)
                correct = answer.is_correct
                if correct is None:
                    correct = normalize_answer(answer.final_answer) == normalize_answer(answer_key)
                responses[student_index, item_index] = int(correct)
                record = {
                    "student_id": f"virtual_{student_index + 1:04d}",
                    "item_id": item.item_id,
                    "position": position,
                    "persona_kind": persona.kind,
                    "target_alpha": list(persona.alpha) if persona.alpha is not None else None,
                    "target_theta": persona.theta,
                    "state": state,
                    "temperature": temperature,
                    "final_answer": answer.final_answer,
                    "work": answer.work,
                    "is_correct": bool(correct),
                    "raw_response": answer.raw_response,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                if state == "stateful":
                    history.append(
                        {
                            "item_id": item.item_id,
                            "question": item.text,
                            "final_answer": answer.final_answer,
                            "work": answer.work,
                        }
                    )
    if (responses < 0).any():
        raise RuntimeError("simulation ended with unscored responses")
    return responses
