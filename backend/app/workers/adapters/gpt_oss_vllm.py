from __future__ import annotations

import asyncio
import gc
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

from ...contracts import InterviewTurnInput
from ...schemas import (
    SCHEMA_VERSION,
    WorkerHealth,
    WorkerModelSpec,
    WorkerRole,
)
from ..base import WorkerAdapter
from .transformers_generation_llm import (
    SAFETY_SYSTEM_PROMPT,
    SUPPORTED_ROLES,
    TransformersGenerationLlmAdapter,
)
from .transformers_interview_llm import (
    SYSTEM_PROMPT,
    TURN_INSTRUCTION,
    TransformersInterviewLlmAdapter,
)


class GptOssVllmRuntime:
    """Single-GPU GPT-OSS runtime using Harmony prompts and vLLM MXFP4."""

    def __init__(
        self,
        llm: Any,
        encoding: Any,
        sampling_params_type: Any,
        torch: Any,
        parameters: dict[str, Any],
    ) -> None:
        self.llm = llm
        self.encoding = encoding
        self.sampling_params_type = sampling_params_type
        self.torch = torch
        self.parameters = parameters
        self.last_prompt_tokens = 0
        self.last_generated_tokens = 0

    @classmethod
    def load(cls, source: str, model: WorkerModelSpec) -> GptOssVllmRuntime:
        # The app already isolates every GPU worker in its own process. Keeping
        # vLLM in that process makes cancellation and VRAM release predictable.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
        os.environ.setdefault("DO_NOT_TRACK", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if model.model_path:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        import torch
        from openai_harmony import (
            HarmonyEncodingName,
            load_harmony_encoding,
        )
        from vllm import LLM, SamplingParams

        if not model.device.lower().startswith("cuda"):
            raise RuntimeError("gpt_oss_vllm_cuda_required")
        if not torch.cuda.is_available():
            raise RuntimeError("gpt_oss_vllm_cuda_unavailable")
        capability = torch.cuda.get_device_capability()
        if capability < (8, 0):
            raise RuntimeError(
                "gpt_oss_vllm_requires_compute_capability_80_or_newer"
            )

        parameters = dict(model.parameters)
        llm = LLM(
            model=source,
            tokenizer=source,
            revision=None if model.model_path else model.model_revision,
            tokenizer_revision=None if model.model_path else model.model_revision,
            dtype="bfloat16",
            hf_overrides={"torch_dtype": torch.bfloat16},
            quantization="mxfp4",
            tensor_parallel_size=1,
            gpu_memory_utilization=float(
                parameters.get("gpu_memory_utilization", 0.78)
            ),
            max_model_len=int(parameters.get("max_model_len", 8192)),
            max_num_seqs=int(parameters.get("max_num_seqs", 1)),
            max_num_batched_tokens=int(
                parameters.get("max_num_batched_tokens", 8192)
            ),
            enforce_eager=bool(parameters.get("enforce_eager", True)),
            enable_prefix_caching=bool(
                parameters.get("enable_prefix_caching", True)
            ),
            swap_space=int(parameters.get("swap_space_gib", 0)),
            cpu_offload_gb=float(parameters.get("cpu_offload_gib", 0)),
            disable_custom_all_reduce=True,
            disable_log_stats=True,
            trust_remote_code=False,
            seed=int(parameters.get("seed", 42)),
        )
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        torch.backends.cuda.matmul.allow_tf32 = True
        return cls(llm, encoding, SamplingParams, torch, parameters)

    def generate(
        self,
        developer_prompt: str,
        user_prompt: str,
        *,
        max_new_tokens: int,
        reasoning_effort: str,
        temperature: float,
    ) -> str:
        from openai_harmony import (
            Conversation,
            DeveloperContent,
            Message,
            ReasoningEffort,
            Role,
            SystemContent,
        )

        effort = {
            "low": ReasoningEffort.LOW,
            "medium": ReasoningEffort.MEDIUM,
            "high": ReasoningEffort.HIGH,
        }.get(reasoning_effort.lower(), ReasoningEffort.LOW)
        conversation = Conversation.from_messages(
            [
                Message.from_role_and_content(
                    Role.SYSTEM,
                    SystemContent.new()
                    .with_reasoning_effort(effort)
                    .with_conversation_start_date(date.today().isoformat()),
                ),
                Message.from_role_and_content(
                    Role.DEVELOPER,
                    DeveloperContent.new().with_instructions(developer_prompt),
                ),
                Message.from_role_and_content(Role.USER, user_prompt),
            ]
        )
        prompt_ids = self.encoding.render_conversation_for_completion(
            conversation,
            Role.ASSISTANT,
        )
        sampling = self.sampling_params_type(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=float(self.parameters.get("top_p", 1.0)),
            seed=int(self.parameters.get("seed", 42)),
            stop_token_ids=self.encoding.stop_tokens_for_assistant_actions(),
        )
        outputs = self.llm.generate(
            [{"prompt_token_ids": prompt_ids}],
            sampling_params=sampling,
            use_tqdm=False,
        )
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("gpt_oss_vllm_empty_generation")
        completion = outputs[0].outputs[0]
        completion_ids = list(completion.token_ids)
        self.last_prompt_tokens = len(prompt_ids)
        self.last_generated_tokens = len(completion_ids)
        if completion.finish_reason == "length":
            raise RuntimeError("gpt_oss_token_limit_reached")
        messages = self.encoding.parse_messages_from_completion_tokens(
            completion_ids,
            Role.ASSISTANT,
        )
        return self.extract_final_message(messages)

    @classmethod
    def extract_final_message(cls, messages: Any) -> str:
        for message in reversed(list(messages)):
            value = message.to_dict() if hasattr(message, "to_dict") else message
            if not isinstance(value, dict):
                continue
            if str(value.get("channel", "")).lower() != "final":
                continue
            text = cls._content_text(value.get("content"))
            if text:
                return text.strip()
        # Never fall back to the analysis channel: it is not user-facing output.
        raise RuntimeError("gpt_oss_final_message_missing")

    @classmethod
    def _content_text(cls, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "".join(cls._content_text(item) for item in value)
        if isinstance(value, dict):
            for key in ("text", "content", "value"):
                if key in value:
                    return cls._content_text(value[key])
        return ""

    def close(self) -> None:
        llm = self.llm
        self.llm = None
        engine = getattr(llm, "llm_engine", None)
        shutdown = getattr(engine, "shutdown", None)
        if not callable(shutdown):
            shutdown = getattr(
                getattr(engine, "engine_core", None),
                "shutdown",
                None,
            )
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass
        if engine is not None:
            llm.llm_engine = None
        shutdown = None
        engine = None
        del llm
        distributed = getattr(self.torch, "distributed", None)
        if distributed is not None and distributed.is_initialized():
            distributed.destroy_process_group()
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            self.torch.cuda.ipc_collect()


class GptOssInterviewLlmAdapter(TransformersInterviewLlmAdapter):
    def __init__(self, role: WorkerRole) -> None:
        super().__init__(role)
        self._runtime: GptOssVllmRuntime | None = None

    async def load(self, model: WorkerModelSpec) -> WorkerHealth:
        if model.schema_version != SCHEMA_VERSION or model.worker != self.role:
            raise ValueError("worker_model_schema_mismatch")
        source = model.model_path or model.model_id
        if model.model_path and not Path(model.model_path).is_dir():
            raise FileNotFoundError(
                f"gpt_oss_model_path_not_found: {model.model_path}"
            )
        runtime = await asyncio.to_thread(GptOssVllmRuntime.load, source, model)
        self._runtime = runtime
        self._model = runtime.llm
        self._tokenizer = runtime.encoding
        self._torch = runtime.torch
        self._multimodal = False
        self._model_spec = model
        return await self.healthcheck()

    def _generate(
        self,
        turn: InterviewTurnInput,
        metadata: dict[str, Any],
    ) -> tuple[str, int, int]:
        assert self._runtime is not None
        assert self._model_spec is not None
        compact_turn = {
            "transcript": [
                {"speaker": entry.speaker, "text": entry.text}
                for entry in turn.transcript[-12:]
            ],
            "acquired_information": turn.state.acquired_information,
            "asked_topics": turn.state.asked_topics,
            "next_topics": turn.state.next_topics,
            "visitor_char_count": turn.state.visitor_char_count,
            "elapsed_seconds": turn.state.elapsed_seconds,
            "target_transcript_chars": turn.target_transcript_chars,
            "remaining_time_seconds": turn.remaining_time_seconds,
        }
        user_prompt = (
            f"{TURN_INSTRUCTION}\n"
            + json.dumps(compact_turn, ensure_ascii=False, separators=(",", ":"))
        )
        parameters = self._model_spec.parameters
        max_tokens = max(
            int(metadata.get("max_new_tokens", 320)),
            int(parameters.get("interview_min_new_tokens", 256)),
        )
        text = self._runtime.generate(
            SYSTEM_PROMPT,
            user_prompt,
            max_new_tokens=max_tokens,
            reasoning_effort=str(
                parameters.get("interview_reasoning_effort", "low")
            ),
            temperature=float(parameters.get("interview_temperature", 0.7)),
        )
        return (
            text,
            self._runtime.last_prompt_tokens,
            self._runtime.last_generated_tokens,
        )

    async def unload(self) -> None:
        runtime = self._runtime
        self._runtime = None
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._model_spec = None
        self._multimodal = False
        self._cancelled.clear()
        if runtime is not None:
            await asyncio.to_thread(runtime.close)


class GptOssGenerationLlmAdapter(TransformersGenerationLlmAdapter):
    def __init__(self, role: WorkerRole) -> None:
        super().__init__(role)
        self._runtime: GptOssVllmRuntime | None = None

    async def load(self, model: WorkerModelSpec) -> WorkerHealth:
        if model.schema_version != SCHEMA_VERSION or model.worker != self.role:
            raise ValueError("worker_model_schema_mismatch")
        source = model.model_path or model.model_id
        if model.model_path and not Path(model.model_path).is_dir():
            raise FileNotFoundError(
                f"gpt_oss_model_path_not_found: {model.model_path}"
            )
        runtime = await asyncio.to_thread(GptOssVllmRuntime.load, source, model)
        self._runtime = runtime
        self._model = runtime.llm
        self._tokenizer = runtime.encoding
        self._torch = runtime.torch
        self._multimodal = False
        self._model_spec = model
        return await self.healthcheck()

    def _generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_new_tokens: int,
    ) -> str:
        assert self._runtime is not None
        assert self._model_spec is not None
        parameters = self._model_spec.parameters
        safety = system_prompt == SAFETY_SYSTEM_PROMPT
        if safety:
            effort = str(parameters.get("safety_reasoning_effort", "low"))
            temperature = float(parameters.get("safety_temperature", 0.0))
            minimum = int(parameters.get("safety_min_new_tokens", 128))
        else:
            effort = str(parameters.get("planning_reasoning_effort", "medium"))
            temperature = float(parameters.get("planning_temperature", 0.7))
            minimum = int(parameters.get("planning_min_new_tokens", 256))
        return self._runtime.generate(
            system_prompt,
            user_prompt,
            max_new_tokens=max(max_new_tokens, minimum),
            reasoning_effort=effort,
            temperature=temperature,
        )

    def _generate_image_observation(
        self,
        image_path: Path,
        max_new_tokens: int,
    ) -> str:
        raise RuntimeError("gpt_oss_20b_is_text_only")

    async def unload(self) -> None:
        runtime = self._runtime
        self._runtime = None
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._model_spec = None
        self._multimodal = False
        self._cancelled.clear()
        if runtime is not None:
            await asyncio.to_thread(runtime.close)


def create_worker(role: WorkerRole) -> WorkerAdapter:
    if role == WorkerRole.INTERVIEW_LLM:
        return GptOssInterviewLlmAdapter(role)
    if role in SUPPORTED_ROLES:
        return GptOssGenerationLlmAdapter(role)
    raise ValueError(f"unsupported_gpt_oss_role: {role.value}")
