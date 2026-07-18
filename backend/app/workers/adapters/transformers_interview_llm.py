from __future__ import annotations

import asyncio
import gc
import json
import re
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...contracts import InterviewTurnInput, InterviewTurnOutput
from ...schemas import (
    SCHEMA_VERSION,
    InterviewTheme,
    WorkerHealth,
    WorkerMetrics,
    WorkerModelSpec,
    WorkerProgressEvent,
    WorkerRequest,
    WorkerResult,
    WorkerRole,
)
from ..base import ProgressCallback, WorkerAdapter


SYSTEM_PROMPT = """\
あなたは、現在の来場者へインタビューする案内役です。
集めた会話を別のAIが使い、「未来の本人から届く20秒のメッセージ動画」を後で作ります。

役割を絶対に混同しないでください:
- あなたは未来の本人ではありません。
- 未来から話したり、未来の本人として質問へ答えたりしません。
- 来場者に、未来の本人ならどう答えるか・笑うか・驚くかを想像させません。
- 動画の設定や台本をこの場で決めません。
- あなたの仕事は、現在の本人から動画の材料を自然に聞くことだけです。

会話は conversation_plan が指定する段階に従います。話題を勝手に選ばないでください。
- future_question: 未来の本人へ聞きたい質問を一度だけ受け取る。答えの予想は聞かない。
- present_connection: 現在よくしていること、気になること、大切な人や物を聞く。
- concrete_episode: 実際にあった一場面を一つ聞く。
- future_expansion: これまでの材料を使い、見てみたい未来の方向を一つ聞く。
- future_message: 未来の本人からどんな調子で話しかけられたいかを聞く。

返答の原則:
- 直前の発話を受けた短いリアクションと、答えやすい質問一つを基本にする。
- 同じ話題を深掘りしてよいが、同じ文を言い換えて繰り返さない。
- 来場者の言葉を長く引用・復唱しない。
- 質問は一度に一つ。心理分析、説教、助言、過剰な称賛をしない。
- 来場者が明言していない事実・感情・人物像を作らない。
- 短く不自然な語は音声認識の誤りかもしれないため、引用せず、事実として扱わない。
- 「いや」「いいです」「特にない」などは拒否や迷いとして受け止め、簡単な聞き方へ変える。
- 「何を言っているか分からない」などの指摘には、まず短く謝り、推測せず、普通の言葉で一問だけ言い換える。
- 「未来の自分に聞かれたら」のように話者を逆転させる表現は禁止する。

重要な例:
conversation_plan: future_question から present_connection へ移る
来場者「歯は何本残っていますか」
返答「かなり具体的な質問だね。それは未来の自分に預けておこう。今のあなたが最近よくしていることは何？」

conversation_plan: present_connection を聞く
来場者「いや」
返答「了解、もっと簡単に聞くね。最近よくやっていることはある？」

conversation_plan: present_connection を聞く
来場者「何を言ってるか分からないです」
返答「ごめん、聞き方がややこしかったね。最近よくやっていることを一つだけ教えてもらえる？」

conversation_plan: concrete_episode を聞く
来場者「友達と協力するゲームをしています」
返答「友達と一緒に遊ぶんだね。最近いちばん盛り上がったのはどんな場面？」

出力規則:
- 実際に音声で読み上げる日本語の返答だけを出力する。
- JSON、見出し、話者名、説明、思考過程、引用符は付けない。
- 1〜2文、120文字以内を目安にする。
- 質問でない短いリアクションが自然なら、それも有効とする。
"""


TURN_INSTRUCTION = """\
あなたは現在の来場者へ質問する案内役で、未来の本人ではありません。
conversation_plan の target_theme と question_goal に従い、latest_visitor_utterance を受けた自然な発話を一つ作ってください。
mode が transition なら前の答えを一度受け止めて次の段階へ進み、follow_up なら現在の話を一段だけ深掘りしてください。
mode が support または repair なら、曖昧な語を引用・解釈せず、謝意または了解を短く示して簡単な一問へ言い換えてください。
未来の本人の反応や回答を来場者に想像させてはいけません。読み上げる返答本文だけを出力してください。
"""


THEME_ORDER = (
    InterviewTheme.FUTURE_QUESTION,
    InterviewTheme.PRESENT_CONNECTION,
    InterviewTheme.CONCRETE_EPISODE,
    InterviewTheme.FUTURE_EXPANSION,
    InterviewTheme.FUTURE_MESSAGE,
)


THEME_GUIDANCE: dict[InterviewTheme, tuple[str, str]] = {
    InterviewTheme.FUTURE_QUESTION: (
        "未来の本人へ聞きたい質問を一つ受け取る。未来の回答や反応は予想させない",
        "未来の自分に一つだけ聞くなら、何が気になる？",
    ),
    InterviewTheme.PRESENT_CONNECTION: (
        "動画の材料になる、現在よくしていること・気になること・大切なものを一つ聞く",
        "最近よくやっていることは何？",
    ),
    InterviewTheme.CONCRETE_EPISODE: (
        "現在の話に関係する、実際にあった一場面を一つ聞く",
        "そのことで、最近いちばん印象に残った場面はどんなものだった？",
    ),
    InterviewTheme.FUTURE_EXPANSION: (
        "これまでに出た材料を使い、未来で見てみたい変化や場面を一つ聞く",
        "その話が未来で大きく発展するとしたら、どんな場面を見てみたい？",
    ),
    InterviewTheme.FUTURE_MESSAGE: (
        "未来の本人から、どんな調子や内容で話しかけられたいかを一つ聞く",
        "未来の自分から、どんな調子で話しかけられたらうれしい？",
    ),
}


@dataclass(frozen=True)
class InterviewTurnPlan:
    current_theme: InterviewTheme
    topic_depth: int
    interesting_detail: str | None
    topic_complete: bool
    next_anchor: InterviewTheme | None
    mode: str
    question_theme: InterviewTheme
    question_goal: str
    anchor_question: str


def _latest_visitor_text(turn: InterviewTurnInput) -> str:
    return next(
        (
            entry.text.strip()
            for entry in reversed(turn.transcript)
            if entry.speaker == "visitor" and entry.text.strip()
        ),
        "",
    )


def _planning_text(text: str) -> str:
    compact = re.sub(
        r"^[、。\s]*(?:えっと|うーん|そうですね)[、。\s]*",
        "",
        text,
    )
    return re.sub(r"[\s、。！？!?]", "", compact)


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _is_confused(text: str) -> bool:
    return _contains_any(
        text,
        (
            "何を言ってるか分からない",
            "何言ってるか分からない",
            "何を言っているか分からない",
            "意味が分からない",
            "意味わからない",
            "質問が分からない",
            "よく分からない",
        ),
    )


def _is_reluctant(text: str) -> bool:
    compact = _planning_text(text)
    return compact in {
        "いや",
        "いい",
        "いいです",
        "もういい",
        "結構です",
        "ない",
    } or _contains_any(
        text,
        (
            "話したくない",
            "言いたくない",
            "答えたくない",
            "触れたくない",
            "やめたい",
            "別の話",
        ),
    )


def _needs_support(text: str) -> bool:
    return len(_planning_text(text)) <= 4 or _is_confused(text) or _is_reluctant(
        text
    ) or _contains_any(
        text,
        (
            "特にない",
            "わからない",
            "分からない",
            "思いつかない",
            "決められない",
            "なんでもいい",
        ),
    )


def _is_usable_statement(text: str) -> bool:
    return bool(text) and not _needs_support(text) and len(_planning_text(text)) >= 5


def _has_concrete_material(text: str) -> bool:
    if not _is_usable_statement(text):
        return False
    if len(_planning_text(text)) >= 12:
        return True
    return _contains_any(
        text,
        (
            "ました",
            "だった",
            "行った",
            "作った",
            "遊んだ",
            "起きた",
            "あった",
            "したとき",
            "した時",
            "ことがある",
            "友達",
            "研究",
            "部活",
            "学校",
            "仕事",
        ),
    )


def _extract_interesting_detail(text: str) -> str | None:
    if not _is_usable_statement(text):
        return None
    compact = re.sub(
        r"^[、。\s]*(?:えっと|うーん|そうですね)[、。\s]*",
        "",
        text,
    ).strip("、。！？!? ")
    if not compact:
        return None
    return compact if len(compact) <= 42 else compact[:41].rstrip() + "…"


def _next_theme(theme: InterviewTheme) -> InterviewTheme | None:
    index = THEME_ORDER.index(theme)
    return THEME_ORDER[index + 1] if index + 1 < len(THEME_ORDER) else None


def plan_interview_turn(turn: InterviewTurnInput) -> InterviewTurnPlan:
    state = turn.state
    current_theme = state.current_theme
    base_depth = state.topic_depth
    if state.topic_complete and state.next_anchor is not None:
        current_theme = state.next_anchor
        base_depth = 0

    visitor_text = _latest_visitor_text(turn)
    usable_statement = _is_usable_statement(visitor_text)
    topic_depth = base_depth + (1 if usable_statement else 0)
    interesting_detail = (
        _extract_interesting_detail(visitor_text) or state.interesting_detail
    )
    next_anchor = _next_theme(current_theme)
    confused = _is_confused(visitor_text)
    support_needed = _needs_support(visitor_text)

    if current_theme == InterviewTheme.FUTURE_QUESTION:
        # The opening answer is a request for the future video, not a topic to
        # repeatedly reinterpret or ask the visitor to answer themselves.
        topic_complete = usable_statement and next_anchor is not None
    else:
        topic_complete = bool(
            next_anchor is not None
            and not confused
            and (
                topic_depth >= 3
                or (topic_depth >= 2 and _has_concrete_material(visitor_text))
            )
        )

    if confused:
        mode = "repair"
        question_theme = current_theme
    elif topic_complete and next_anchor is not None:
        mode = "transition"
        question_theme = next_anchor
    elif support_needed:
        mode = "support"
        question_theme = current_theme
    else:
        mode = "follow_up"
        question_theme = current_theme
    question_goal, anchor_question = THEME_GUIDANCE[question_theme]
    return InterviewTurnPlan(
        current_theme=current_theme,
        topic_depth=topic_depth,
        interesting_detail=interesting_detail,
        topic_complete=topic_complete,
        next_anchor=next_anchor,
        mode=mode,
        question_theme=question_theme,
        question_goal=question_goal,
        anchor_question=anchor_question,
    )


def build_turn_prompt(turn: InterviewTurnInput) -> str:
    plan = plan_interview_turn(turn)
    visitor_text = _latest_visitor_text(turn)
    if _is_confused(visitor_text):
        visitor_signal = "confused_by_previous_question"
    elif _is_reluctant(visitor_text):
        visitor_signal = "declining_or_uncertain"
    elif _is_usable_statement(visitor_text):
        visitor_signal = "usable_statement"
    else:
        visitor_signal = "short_or_asr_uncertain"
    recent_transcript = [
        {"speaker": entry.speaker, "text": entry.text}
        for entry in turn.transcript[-8:]
    ]
    redaction_by_signal = {
        "confused_by_previous_question": (
            "[来場者が直前の質問を理解できないと伝えた。推測せず言い換える]"
        ),
        "declining_or_uncertain": (
            "[来場者は短く断るか迷っている。元の語句を引用・解釈しない]"
        ),
        "short_or_asr_uncertain": (
            "[短く曖昧な音声認識結果。元の語句を引用・事実化しない]"
        ),
    }
    safe_visitor_text = redaction_by_signal.get(visitor_signal, visitor_text)
    if safe_visitor_text != visitor_text:
        for entry in reversed(recent_transcript):
            if entry["speaker"] == "visitor":
                entry["text"] = safe_visitor_text
                break
    payload = {
        "role_reminder": (
            "You are the present-day interviewer, not the future person. "
            "Do not answer or simulate the future person's reaction."
        ),
        "conversation_plan": {
            "mode": plan.mode,
            "current_theme": plan.current_theme.value,
            "target_theme": plan.question_theme.value,
            "question_goal": plan.question_goal,
            "visitor_signal": visitor_signal,
            "future_question_captured": (
                plan.current_theme == InterviewTheme.FUTURE_QUESTION
                and plan.topic_complete
            ),
        },
        "latest_visitor_utterance": safe_visitor_text,
        "recent_transcript": recent_transcript,
        "response_constraints": {
            "spoken_japanese_only": True,
            "maximum_questions": 1,
            "do_not_quote_uncertain_asr": True,
            "do_not_invent_facts_or_emotions": True,
            "do_not_ask_visitor_to_play_future_self": True,
        },
    }
    return (
        f"{TURN_INSTRUCTION}\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )



class TransformersInterviewLlmAdapter(WorkerAdapter):
    role = WorkerRole.INTERVIEW_LLM

    def __init__(self, role: WorkerRole) -> None:
        if role != self.role:
            raise ValueError(f"unsupported_transformers_interview_role: {role.value}")
        self._model_spec: WorkerModelSpec | None = None
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None
        self._multimodal = False
        self._cancelled: set[str] = set()

    async def load(self, model: WorkerModelSpec) -> WorkerHealth:
        if model.schema_version != SCHEMA_VERSION or model.worker != self.role:
            raise ValueError("worker_model_schema_mismatch")
        source = model.model_path or model.model_id
        if model.model_path and not Path(model.model_path).is_dir():
            raise FileNotFoundError(
                f"interview_llm_model_path_not_found: {model.model_path}"
            )
        (
            self._model,
            self._tokenizer,
            self._torch,
            self._multimodal,
        ) = await asyncio.to_thread(self._create_model, source, model)
        self._model_spec = model
        return await self.healthcheck()

    @staticmethod
    def _create_model(
        source: str,
        model: WorkerModelSpec,
    ) -> tuple[Any, Any, Any, bool]:
        import torch

        if model.device.lower().startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("interview_llm_cuda_unavailable")

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(model.dtype.lower(), torch.bfloat16)
        local_only = bool(model.model_path)
        multimodal = bool(model.parameters.get("multimodal", False))
        if multimodal:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            tokenizer = AutoProcessor.from_pretrained(
                source,
                local_files_only=local_only,
            )
            loaded_model = Qwen3VLForConditionalGeneration.from_pretrained(
                source,
                dtype=dtype,
                local_files_only=local_only,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                source,
                local_files_only=local_only,
            )
            loaded_model = AutoModelForCausalLM.from_pretrained(
                source,
                dtype=dtype,
                local_files_only=local_only,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
        loaded_model.to(model.device)
        loaded_model.eval()
        loaded_model.generation_config.temperature = None
        loaded_model.generation_config.top_p = None
        loaded_model.generation_config.top_k = None
        text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        if text_tokenizer.pad_token_id is None:
            text_tokenizer.pad_token_id = text_tokenizer.eos_token_id

        if model.device.lower().startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
        TransformersInterviewLlmAdapter._warmup(
            loaded_model,
            tokenizer,
            torch,
            multimodal,
        )
        return loaded_model, tokenizer, torch, multimodal

    @staticmethod
    def _warmup(
        model: Any,
        tokenizer: Any,
        torch: Any,
        multimodal: bool,
    ) -> None:
        messages = [
            {"role": "system", "content": "日本語の短い返答だけを返してください。"},
            {"role": "user", "content": "準備できていますか？"},
        ]
        if multimodal:
            messages = TransformersInterviewLlmAdapter._text_only_vl_messages(messages)
        text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                use_cache=True,
                pad_token_id=text_tokenizer.pad_token_id,
                eos_token_id=text_tokenizer.eos_token_id,
            )

    @staticmethod
    def _text_only_vl_messages(
        messages: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
            }
            for message in messages
        ]

    async def healthcheck(self) -> WorkerHealth:
        return WorkerHealth(
            worker=self.role,
            loaded=self._model is not None,
            ready=(
                self._model is not None
                and self._tokenizer is not None
                and self._model_spec is not None
            ),
            backend=self._model_spec.backend if self._model_spec else None,
            model_id=self._model_spec.model_id if self._model_spec else None,
        )

    async def run(
        self,
        request: WorkerRequest,
        progress: ProgressCallback | None = None,
    ) -> WorkerResult:
        if request.schema_version != SCHEMA_VERSION:
            raise ValueError("worker_request_schema_mismatch")
        if request.worker != self.role:
            raise ValueError("worker_role_mismatch")
        if (
            self._model is None
            or self._tokenizer is None
            or self._torch is None
            or self._model_spec is None
        ):
            raise RuntimeError("worker_not_loaded")
        if request.model.catalog_id != self._model_spec.catalog_id:
            raise ValueError("worker_loaded_model_mismatch")

        turn_value = request.metadata.get("interview_turn")
        if not isinstance(turn_value, dict):
            raise ValueError("interview_turn_input_required")
        turn = InterviewTurnInput.model_validate(turn_value)
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if request.request_id in self._cancelled:
            self._cancelled.discard(request.request_id)
            raise asyncio.CancelledError
        if progress:
            await progress(
                WorkerProgressEvent(
                    request_id=request.request_id,
                    worker=self.role,
                    progress=0.15,
                    message="会話の続きを考えています",
                )
            )

        started = time.perf_counter()
        if self._model_spec.device.lower().startswith("cuda"):
            self._torch.cuda.reset_peak_memory_stats()
        raw_text, prompt_tokens, generated_tokens = await asyncio.to_thread(
            self._generate,
            turn,
            request.metadata,
        )
        if request.request_id in self._cancelled:
            self._cancelled.discard(request.request_id)
            raise asyncio.CancelledError

        parsed = self._parse_output(raw_text, turn)
        output_path = output_dir / "interview-turn.json"
        output_path.write_text(
            json.dumps(parsed.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        peak_vram_mb = 0
        if self._model_spec.device.lower().startswith("cuda"):
            peak_vram_mb = int(
                self._torch.cuda.max_memory_allocated() / 1024 / 1024
            )

        if progress:
            await progress(
                WorkerProgressEvent(
                    request_id=request.request_id,
                    worker=self.role,
                    progress=1.0,
                    message="返事が決まりました",
                )
            )
        return WorkerResult(
            request_id=request.request_id,
            worker=self.role,
            backend=self._model_spec.backend,
            model_id=self._model_spec.model_id,
            model_revision=self._model_spec.model_revision,
            implemented=True,
            output_paths={"interview_turn": str(output_path)},
            metadata={
                "interview_turn": parsed.model_dump(mode="json"),
                "next_utterance": parsed.next_utterance,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
            },
            metrics=WorkerMetrics(
                processing_time_ms=elapsed_ms,
                peak_vram_mb=peak_vram_mb,
                peak_cpu_memory_mb=self._peak_cpu_memory_mb(),
            ),
        )

    def _generate(
        self,
        turn: InterviewTurnInput,
        metadata: dict[str, Any],
    ) -> tuple[str, int, int]:
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._torch is not None
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_turn_prompt(turn)},
        ]
        if self._multimodal:
            messages = self._text_only_vl_messages(messages)
        text_tokenizer = getattr(self._tokenizer, "tokenizer", self._tokenizer)
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max(64, int(metadata.get("max_new_tokens", 320))),
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.05,
                pad_token_id=text_tokenizer.pad_token_id,
                eos_token_id=text_tokenizer.eos_token_id,
            )
        output_ids = generated[0, prompt_tokens:]
        raw_text = text_tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
        ).strip()
        return raw_text, prompt_tokens, int(output_ids.shape[-1])

    @staticmethod
    def _parse_output(
        raw_text: str,
        turn: InterviewTurnInput,
    ) -> InterviewTurnOutput:
        plan = plan_interview_turn(turn)
        utterance = TransformersInterviewLlmAdapter._recover_model_utterance(
            raw_text
        )
        if not utterance:
            return TransformersInterviewLlmAdapter._fallback_output(turn)
        next_utterance, _ = TransformersInterviewLlmAdapter._shape_reply(
            utterance,
            turn,
            [plan.question_theme.value],
        )
        output = InterviewTurnOutput(
            acquired_information=(
                TransformersInterviewLlmAdapter._record_visitor_information(
                    turn,
                    plan,
                )
            ),
            asked_topics=list(
                dict.fromkeys(
                    [*turn.state.asked_topics, plan.question_theme.value]
                )
            ),
            next_topics=[],
            visitor_char_count=turn.state.visitor_char_count,
            elapsed_seconds=turn.state.elapsed_seconds,
            should_end=False,
            end_reason="continue",
            next_utterance=next_utterance,
        )
        TransformersInterviewLlmAdapter._apply_plan(output, plan)
        return output

    @staticmethod
    def _record_visitor_information(
        turn: InterviewTurnInput,
        plan: InterviewTurnPlan,
    ) -> dict[str, Any]:
        recorded = dict(turn.state.acquired_information)
        visitor_text = _latest_visitor_text(turn)
        if not _is_usable_statement(visitor_text):
            return recorded
        key_by_theme = {
            InterviewTheme.FUTURE_QUESTION: "future_questions",
            InterviewTheme.PRESENT_CONNECTION: "present_details",
            InterviewTheme.CONCRETE_EPISODE: "concrete_episodes",
            InterviewTheme.FUTURE_EXPANSION: "future_preferences",
            InterviewTheme.FUTURE_MESSAGE: "message_preferences",
        }
        key = key_by_theme[plan.current_theme]
        existing = recorded.get(key)
        values = list(existing) if isinstance(existing, list) else []
        if visitor_text not in values:
            values.append(visitor_text)
        recorded[key] = values
        return recorded

    @staticmethod
    def _apply_plan(
        output: InterviewTurnOutput,
        plan: InterviewTurnPlan,
    ) -> None:
        output.current_theme = plan.current_theme
        output.topic_depth = plan.topic_depth
        output.interesting_detail = plan.interesting_detail
        output.topic_complete = plan.topic_complete
        output.next_anchor = plan.next_anchor

    @staticmethod
    def _recover_model_utterance(raw_text: str) -> str:
        raw = raw_text.strip()
        if not raw:
            return ""
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw).strip()

        quoted = re.search(
            r'"next_utterance"\s*:\s*"((?:\\.|[^"\\])*)',
            raw,
            flags=re.DOTALL,
        )
        if quoted:
            try:
                return " ".join(
                    json.loads(f'"{quoted.group(1)}"').split()
                ).strip()
            except json.JSONDecodeError:
                pass

        labelled = re.search(
            r"(?:next_utterance|返答)\s*[:：]\s*([^\n{}]+)",
            raw,
        )
        if labelled:
            return " ".join(labelled.group(1).strip(' "').split())

        # Broken JSON must never be spoken aloud; plain model prose is still a
        # valid conversational choice and is kept even when it is not a question.
        if raw.startswith(("{", "[")) or "\"next_utterance\"" in raw:
            return ""
        return " ".join(raw.split()).strip(' "')

    @staticmethod
    def _fallback_output(turn: InterviewTurnInput) -> InterviewTurnOutput:
        plan = plan_interview_turn(turn)
        next_utterance, topic = TransformersInterviewLlmAdapter._fallback_question(
            turn
        )
        output = InterviewTurnOutput(
            acquired_information=(
                TransformersInterviewLlmAdapter._record_visitor_information(
                    turn,
                    plan,
                )
            ),
            asked_topics=list(
                dict.fromkeys([*turn.state.asked_topics, topic])
            ),
            next_topics=[],
            visitor_char_count=turn.state.visitor_char_count,
            elapsed_seconds=turn.state.elapsed_seconds,
            should_end=False,
            end_reason="continue",
            next_utterance=next_utterance,
        )
        TransformersInterviewLlmAdapter._apply_plan(output, plan)
        return output

    @staticmethod
    def _shape_reply(
        text: str,
        turn: InterviewTurnInput,
        proposed_topics: list[str],
    ) -> tuple[str, str | None]:
        del proposed_topics
        compact = " ".join(text.split()).strip()
        if compact:
            return compact, None
        return TransformersInterviewLlmAdapter._fallback_question(turn)

    @staticmethod
    def _fallback_question(
        turn: InterviewTurnInput,
        proposed_topics: list[str] | None = None,
    ) -> tuple[str, str]:
        del proposed_topics
        plan = plan_interview_turn(turn)
        visitor_text = _latest_visitor_text(turn)
        support_questions = {
            InterviewTheme.FUTURE_QUESTION: (
                "仕事、毎日の生活、ちょっと変なことでも大丈夫。"
                "未来の自分に一つだけ聞くなら、何が気になる？"
            ),
            InterviewTheme.PRESENT_CONNECTION: (
                "最近よくやっていることと、最近ちょっと気になることなら、"
                "どちらが話しやすい？"
            ),
            InterviewTheme.CONCRETE_EPISODE: (
                "学校、研究室、友達との時間の中で、最近覚えている出来事はある？"
            ),
            InterviewTheme.FUTURE_EXPANSION: (
                "大成功する未来と、思い切りSFな未来なら、どちらを見てみたい？"
            ),
            InterviewTheme.FUTURE_MESSAGE: (
                "笑わせてほしいのと、驚かせてほしいのなら、どちらが近い？"
            ),
        }
        if _is_confused(visitor_text):
            return (
                f"ごめん、聞き方がややこしかったね。"
                f"{plan.anchor_question}",
                plan.question_theme.value,
            )
        if _needs_support(visitor_text):
            return (
                f"了解、もっと簡単に聞くね。"
                f"{support_questions[plan.question_theme]}",
                plan.question_theme.value,
            )
        if (
            plan.current_theme == InterviewTheme.FUTURE_QUESTION
            and plan.topic_complete
        ):
            return (
                "その質問は未来の自分に預けておくね。"
                "今のあなたが最近よくしていることは何？",
                plan.question_theme.value,
            )
        return plan.anchor_question, plan.question_theme.value

    @staticmethod
    def _peak_cpu_memory_mb() -> int:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) // 1024

    async def cancel(self, request_id: str) -> None:
        self._cancelled.add(request_id)

    async def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        torch = self._torch
        self._torch = None
        self._model_spec = None
        self._multimodal = False
        self._cancelled.clear()
        await asyncio.to_thread(gc.collect)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def create_worker(role: WorkerRole) -> WorkerAdapter:
    if role == WorkerRole.INTERVIEW_LLM:
        return TransformersInterviewLlmAdapter(role)
    from .transformers_generation_llm import TransformersGenerationLlmAdapter

    return TransformersGenerationLlmAdapter(role)
