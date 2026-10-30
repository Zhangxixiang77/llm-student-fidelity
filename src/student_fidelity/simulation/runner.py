"""Longitudinal item-by-item simulation with raw-output retention."""

from __future__ import annotations

import json
import re
import unicodedata
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from student_fidelity.data import ResponseDataset
from student_fidelity.simulation.personas import Persona
from student_fidelity.simulation.simulators import Item, StructuredAnswerError, StudentSimulator


def normalize_answer(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    return "".join(normalized.split())


def parse_rational_answer(value: str) -> Fraction | None:
    """Parse exact integers, fractions, mixed numbers, finite decimals, and LaTeX fractions."""

    text = unicodedata.normalize("NFKC", value).strip().replace("−", "-")
    text = text.replace("$", "").replace("\\left", "").replace("\\right", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    mixed = re.fullmatch(r"([+-]?\d+)\s+(\d+)\s*/\s*(\d+)", text)
    if mixed:
        whole, numerator, denominator = (int(part) for part in mixed.groups())
        if denominator == 0:
            return None
        magnitude = abs(whole) + Fraction(numerator, denominator)
        return -magnitude if whole < 0 else magnitude
    compact = "".join(text.split())
    latex_mixed = re.fullmatch(r"([+-]?\d+)\\frac\{(\d+)\}\{(\d+)\}", compact)
    if latex_mixed:
        whole, numerator, denominator = (int(part) for part in latex_mixed.groups())
        if denominator == 0:
            return None
        magnitude = abs(whole) + Fraction(numerator, denominator)
        return -magnitude if whole < 0 else magnitude
    latex = re.fullmatch(r"([+-]?)\\frac\{(\d+)\}\{(\d+)\}", compact)
    if latex:
        sign, numerator, denominator = latex.groups()
        if int(denominator) == 0:
            return None
        return Fraction(int(numerator), int(denominator)) * (-1 if sign else 1)
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?", compact):
        return None
    try:
        return Fraction(compact)
    except (ValueError, ZeroDivisionError):
        return None


def answers_equivalent(answer: str, answer_key: str) -> bool:
    parsed_answer = parse_rational_answer(answer)
    parsed_key = parse_rational_answer(answer_key)
    if parsed_answer is not None and parsed_key is not None:
        return parsed_answer == parsed_key
    return normalize_answer(answer) == normalize_answer(answer_key)


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
    resume: bool = False,
    max_calls: int | None = None,
) -> NDArray[np.int8]:
    if state not in {"stateless", "stateful"}:
        raise ValueError("state must be stateless or stateful")
    if not personas:
        raise ValueError("at least one persona is required")
    if not 0 <= temperature <= 2:
        raise ValueError("temperature must be between zero and two")
    if max_calls is not None and max_calls < 1:
        raise ValueError("max_calls must be positive")

    n_students = len(personas)
    responses = np.full((n_students, dataset.n_items), -1, dtype=np.int8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_records: list[dict[str, Any]] = []
    completed: set[tuple[int, int]] = set()
    item_indices = {item_id: index for index, item_id in enumerate(dataset.item_ids)}
    if resume and output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            existing_records.append(record)
            if not isinstance(record.get("is_correct"), bool):
                continue
            student_index = int(record["student_id"].rsplit("_", 1)[1]) - 1
            item_index = item_indices[record["item_id"]]
            if not 0 <= student_index < n_students:
                raise ValueError("resume file contains an out-of-range student")
            key = (student_index, item_index)
            if key in completed:
                raise ValueError("resume file contains duplicate scored responses")
            completed.add(key)
            responses[student_index, item_index] = int(record["is_correct"])
    attempts = len(existing_records)
    with output_path.open("w", encoding="utf-8") as output:
        for record in existing_records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
        for student_index, persona in enumerate(personas):
            history: list[dict[str, Any]] = []
            if state == "stateful":
                previous = [
                    record
                    for record in existing_records
                    if record.get("student_id") == f"virtual_{student_index + 1:04d}"
                    and isinstance(record.get("is_correct"), bool)
                ]
                for record in sorted(previous, key=lambda value: value["position"]):
                    history.append(
                        {
                            "item_id": record["item_id"],
                            "final_answer": record["final_answer"],
                            "work": record.get("work", ""),
                        }
                    )
            order = latin_square_item_order(dataset.n_items, student_index, seed)
            for position, item_index in enumerate(order):
                if (student_index, int(item_index)) in completed:
                    continue
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
                if max_calls is not None and attempts >= max_calls:
                    raise RuntimeError(
                        f"call budget exhausted ({max_calls}); resume only after explicit review"
                    )
                attempts += 1
                try:
                    answer = simulator.answer(persona, item, history, temperature)
                except StructuredAnswerError as error:
                    failure = {
                        "student_id": f"virtual_{student_index + 1:04d}",
                        "item_id": item.item_id,
                        "position": position,
                        "persona_kind": persona.kind,
                        "state": state,
                        "temperature": temperature,
                        "error": str(error),
                        "raw_response": error.raw_response,
                        "response_metadata": error.response_metadata,
                    }
                    output.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    output.flush()
                    raise
                correct = answer.is_correct
                if correct is None:
                    correct = answers_equivalent(answer.final_answer, answer_key)
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
                    "response_metadata": answer.response_metadata,
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
