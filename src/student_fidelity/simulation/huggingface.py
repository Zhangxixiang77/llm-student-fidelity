"""Local Hugging Face inference with immutable revisions and optional dependencies."""

from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Any, Literal

from student_fidelity.simulation.personas import Persona
from student_fidelity.simulation.simulators import (
    Item,
    SimulationAnswer,
    StructuredAnswerError,
    StudentSimulator,
    build_student_prompt,
    parse_structured_answer,
)

_FULL_REVISION = re.compile(r"^[0-9a-f]{40}$")


def validate_huggingface_revision(revision: str) -> str:
    """Reject mutable branches and abbreviated commit identifiers."""

    if not _FULL_REVISION.fullmatch(revision):
        raise ValueError("Hugging Face revision must be a full 40-character commit SHA")
    return revision


class HuggingFaceLocalSimulator(StudentSimulator):
    """Run a causal instruction model locally without network access during generation."""

    def __init__(
        self,
        model: str,
        revision: str,
        cache_dir: str | Path | None = None,
        device: Literal["auto", "cuda", "cpu"] = "auto",
        dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto",
        max_new_tokens: int = 1024,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        chat_template_kwargs: dict[str, Any] | None = None,
        prompt_strategy: Literal["answer_first", "error_examples"] = "answer_first",
        top_p: float = 1.0,
        seed: int = 42,
    ):
        self.model = model
        self.revision = validate_huggingface_revision(revision)
        if device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be auto, cuda, or cpu")
        if dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError("unsupported Hugging Face dtype")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if prompt_strategy not in {"answer_first", "error_examples"}:
            raise ValueError("prompt_strategy must be answer_first or error_examples")
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.prompt_strategy = prompt_strategy
        self.top_p = top_p
        self.seed = seed
        self._generation_index = 0
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "local Hugging Face inference requires the optional 'hf' dependencies"
            ) from error
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available")
        common = {
            "revision": self.revision,
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
        }
        tokenizer = AutoTokenizer.from_pretrained(self.model, **common)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype]
        device_map = {"auto": "auto", "cuda": {"": 0}, "cpu": None}[self.device]
        model = AutoModelForCausalLM.from_pretrained(
            self.model,
            **common,
            torch_dtype=dtype,
            device_map=device_map,
        )
        model.eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model

    def answer(
        self,
        persona: Persona,
        item: Item,
        history: list[dict[str, Any]],
        temperature: float,
    ) -> SimulationAnswer:
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between zero and two")
        prompt = build_student_prompt(persona, item, history, self.prompt_strategy)
        content, response_metadata = self._generate(prompt, temperature)
        try:
            final_answer, work = parse_structured_answer(content)
        except StructuredAnswerError as error:
            error.response_metadata = response_metadata
            raise
        return SimulationAnswer(
            final_answer, work, content, response_metadata=response_metadata
        )

    def _generate(self, prompt: str, temperature: float) -> tuple[str, dict[str, Any]]:
        self._ensure_loaded()
        torch = self._torch
        generation_seed = self.seed + self._generation_index
        self._generation_index += 1
        torch.manual_seed(generation_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(generation_seed)
        rendered = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        inputs = self._tokenizer(rendered, return_tensors="pt")
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
        input_tokens = int(inputs["input_ids"].shape[-1])
        do_sample = temperature > 0
        generation: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if do_sample:
            generation.update(temperature=temperature, top_p=self.top_p)
        with torch.inference_mode():
            output = self._model.generate(**inputs, **generation)
        output_ids = output[0, input_tokens:]
        content = self._tokenizer.decode(output_ids, skip_special_tokens=True)
        output_tokens = int(output_ids.shape[-1])
        eos_id = self._tokenizer.eos_token_id
        ended_with_eos = eos_id is not None and output_tokens > 0 and bool(
            output_ids[-1].item() == eos_id
        )
        finish_reason = (
            "length" if output_tokens >= self.max_new_tokens and not ended_with_eos else "stop"
        )
        return content, {
            "provider_model": self.model,
            "provider_revision": self.revision,
            "backend": "huggingface_local",
            "finish_reason": finish_reason,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            "generation_seed": generation_seed,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": self.top_p if do_sample else None,
        }

    def close(self) -> None:
        """Release one candidate before loading the next model."""

        self._model = None
        self._tokenizer = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self._torch = None

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "huggingface_local",
            "model": self.model,
            "revision": self.revision,
            "network_access": False,
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
            "device": self.device,
            "dtype": self.dtype,
            "max_new_tokens": self.max_new_tokens,
            "chat_template_kwargs": self.chat_template_kwargs,
            "prompt_strategy": self.prompt_strategy,
            "top_p": self.top_p,
            "seed": self.seed,
        }
