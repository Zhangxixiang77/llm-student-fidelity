"""Offline mock and optional OpenAI-compatible student simulators."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from student_fidelity.baselines.dina import DINABaseline
from student_fidelity.metrics import ideal_dina_responses
from student_fidelity.simulation.personas import Persona


@dataclass
class Item:
    item_id: str
    text: str
    answer_key: str
    q_vector: np.ndarray
    index: int


@dataclass
class SimulationAnswer:
    final_answer: str
    work: str
    raw_response: str
    is_correct: bool | None = None


class StudentSimulator(ABC):
    @abstractmethod
    def answer(
        self,
        persona: Persona,
        item: Item,
        history: list[dict[str, Any]],
        temperature: float,
    ) -> SimulationAnswer:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError


class DINAMockSimulator(StudentSimulator):
    """Deterministic pipeline stand-in that samples from assigned alpha and fitted slip/guess."""

    def __init__(self, model: DINABaseline, seed: int = 42):
        model._check_fitted()
        self.slip = model.slip_
        self.guess = model.guess_
        self.q_matrix = model.q_matrix_
        self.rng = np.random.default_rng(seed)

    def answer(
        self,
        persona: Persona,
        item: Item,
        history: list[dict[str, Any]],
        temperature: float,
    ) -> SimulationAnswer:
        if persona.alpha is None:
            raise ValueError("the DINA mock requires an attribute persona")
        alpha = np.asarray(persona.alpha, dtype=np.int8)[None, :]
        eta = int(ideal_dina_responses(alpha, item.q_vector[None, :])[0, 0])
        p_correct = 1 - self.slip[item.index] if eta else self.guess[item.index]
        correct = bool(self.rng.random() < p_correct)
        final_answer = item.answer_key if correct else "incorrect"
        work = "Offline DINA mock; no language-model reasoning was generated."
        return SimulationAnswer(final_answer, work, "offline-dina-mock", is_correct=correct)

    def metadata(self) -> dict[str, Any]:
        return {"type": "dina_mock", "network_access": False}


class OpenAICompatibleSimulator(StudentSimulator):
    """Call a chat-completions endpoint without adding a provider-specific SDK."""

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = timeout

    def answer(
        self,
        persona: Persona,
        item: Item,
        history: list[dict[str, Any]],
        temperature: float,
    ) -> SimulationAnswer:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"set {self.api_key_env} before running a real model")
        history_text = (
            json.dumps(history, ensure_ascii=False) if history else "No previous answers."
        )
        prompt = f"""{persona.render()}

Answer as this student, not as an assistant trying to maximize accuracy.
Give the final answer first, then optionally show brief work. Return only JSON:
{{"final_answer": "...", "work": "..."}}

Previous answers by this student:
{history_text}

Current question ({item.item_id}):
{item.text}
"""
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        final_answer, work = parse_structured_answer(content)
        return SimulationAnswer(final_answer, work, content)

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "openai_compatible",
            "model": self.model,
            "base_url": self.base_url,
            "network_access": True,
        }


def parse_structured_answer(content: str) -> tuple[str, str]:
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        raise ValueError("model output did not contain a JSON object")
    try:
        payload = json.loads(match.group(0))
        return str(payload["final_answer"]), str(payload.get("work", ""))
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("model output must include string field final_answer") from error
