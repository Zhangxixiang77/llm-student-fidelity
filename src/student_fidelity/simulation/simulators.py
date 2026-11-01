"""Offline mock and optional OpenAI-compatible student simulators."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

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


class StructuredAnswerError(ValueError):
    def __init__(self, raw_response: str):
        super().__init__("model output must include a parseable final_answer field")
        self.raw_response = raw_response
        self.response_metadata: dict[str, Any] = {}


@dataclass
class SimulationAnswer:
    final_answer: str
    work: str
    raw_response: str
    is_correct: bool | None = None
    response_metadata: dict[str, Any] = field(default_factory=dict)


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


def build_student_prompt(
    persona: Persona,
    item: Item,
    history: list[dict[str, Any]],
    prompt_strategy: Literal["answer_first", "error_examples"],
) -> str:
    """Build the same student prompt for hosted and local model backends."""

    history_text = json.dumps(history, ensure_ascii=False) if history else "No previous answers."
    error_instruction = ""
    if prompt_strategy == "error_examples":
        error_instruction = """
Use these only as examples of mistakes a student can make, not as a demand to err:
adding numerators and denominators directly, or failing to regroup when subtracting
a fraction from a whole number. If such a mistake follows from this student's profile,
reproduce it; otherwise respond as this student would.
"""
    return f"""{persona.render()}

Answer as this student, not as an assistant trying to maximize accuracy.
{error_instruction}
Give the final answer first, then optionally show brief work. Return only JSON:
{{"final_answer": "...", "work": "..."}}

Previous answers by this student:
{history_text}

Current question ({item.item_id}):
{item.text}
"""


class OpenAICompatibleSimulator(StudentSimulator):
    """Call a chat-completions endpoint without adding a provider-specific SDK."""

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 120.0,
        extra_body: dict[str, Any] | None = None,
        temperature_supported: bool = True,
        fixed_temperature: float | None = None,
        prompt_strategy: Literal["answer_first", "error_examples"] = "answer_first",
    ):
        if fixed_temperature is not None and not 0 <= fixed_temperature <= 2:
            raise ValueError("fixed_temperature must be between zero and two")
        if prompt_strategy not in {"answer_first", "error_examples"}:
            raise ValueError("prompt_strategy must be answer_first or error_examples")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.extra_body = dict(extra_body or {})
        self.temperature_supported = temperature_supported
        self.fixed_temperature = fixed_temperature
        self.prompt_strategy = prompt_strategy

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
        prompt = build_student_prompt(persona, item, history, self.prompt_strategy)
        if not self.temperature_supported:
            if self.fixed_temperature is None:
                raise ValueError("fixed_temperature is required when temperature is unsupported")
            if temperature != self.fixed_temperature:
                raise ValueError(
                    f"{self.model} fixes temperature at {self.fixed_temperature}; "
                    f"refusing to record an unapplied value of {temperature}"
                )
        request_body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            **self.extra_body,
        }
        if self.temperature_supported:
            request_body["temperature"] = temperature
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=request_body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        choice = payload["choices"][0]
        content = choice["message"]["content"]
        response_metadata = {
            "provider_model": payload.get("model", self.model),
            "finish_reason": choice.get("finish_reason"),
            "usage": payload.get("usage", {}),
        }
        try:
            final_answer, work = parse_structured_answer(content)
        except StructuredAnswerError as error:
            error.response_metadata = response_metadata
            raise
        return SimulationAnswer(
            final_answer, work, content, response_metadata=response_metadata
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "openai_compatible",
            "model": self.model,
            "base_url": self.base_url,
            "network_access": True,
            "temperature_supported": self.temperature_supported,
            "fixed_temperature": self.fixed_temperature,
            "prompt_strategy": self.prompt_strategy,
            "extra_body": self.extra_body,
        }


def parse_structured_answer(content: str) -> tuple[str, str]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", content):
        try:
            payload, _ = decoder.raw_decode(content[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "final_answer" in payload:
            return str(payload["final_answer"]), str(payload.get("work", ""))

    # Some compatible endpoints emit JSON-like flow mappings with unquoted keys.
    final = re.search(
        r"[\"']?final_answer[\"']?\s*:\s*([\"'])(.*?)\1", content, flags=re.DOTALL
    )
    if final:
        work = re.search(r"[\"']?work[\"']?\s*:\s*([\"'])(.*?)\1", content, flags=re.DOTALL)
        return final.group(2), work.group(2) if work else ""
    numeric = re.search(
        r"[\"']?final_answer[\"']?\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*[+-]?\d+)?)",
        content,
    )
    if numeric:
        work = re.search(r"[\"']?work[\"']?\s*:\s*([\"'])(.*?)\1", content, flags=re.DOTALL)
        return numeric.group(1), work.group(2) if work else ""
    raise StructuredAnswerError(content)
