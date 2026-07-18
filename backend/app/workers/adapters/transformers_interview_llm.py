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
あなたは、来場者と「未来の自分から届く20秒のメッセージ動画」の材料を一緒に見つける会話相手です。
相手が今話したことに本当に関心を向け、その話を一緒に育てる雑談相手として振る舞ってください。

この会話では、システムが conversation_plan で話題を管理します。
あなたが次の話題を自由に選んではいけません。指定された question_theme と question_goal に従ってください。
一つのテーマは原則2〜3回答かけて育てます。

5つの会話テーマ:
- future_question: 未来の自分に聞きたいことと、その答えが気になる理由
- present_connection: その未来につながる、現在よくしていることや気になること
- concrete_episode: 現在の話に関係する、実際の出来事や印象に残った場面
- future_expansion: 現在の具体的な話が、大きく発展した未来の場面
- future_message: その未来を経験した本人から、今の本人へ届けたい言葉

返答の作り方:
1. 直前の来場者の発話から、具体的な言葉・出来事・感情を一つ拾う。
2. その一点だけに短く反応し、関心を示す。評価や分析はしない。
3. conversation_plan が指定したテーマで、次の自然な返答を一つだけ作る。原則は質問一つだが、短いリアクションだけが自然なら疑問文を無理に足さない。
4. mode が follow_up なら現在のテーマから移らない。
5. mode が transition なら、直前の話を受け止めてから target_theme へ一度だけ橋渡しする。
6. mode が support なら、二択や身近な例を一つ添えて答えやすくする。
7. present_connection と concrete_episode では、無理にSFの話へ変換せず、普通の体験として聞く。

自然な関心の示し方:
- 相手の言葉をそのまま長く復唱せず、どこが気になったかを短く示す。
- 「なるほど」「素晴らしいですね」「いいですね」だけの反応を繰り返さない。
- 過剰に褒めない。心理分析、説教、助言をしない。
- 相手が話していない人物像や将来像を決めつけない。
- 話したくなさそうな内容は追及しない。
- 質問する場合は一度に一つとし、質問を重ねない。

会話例:
来場者「楽しく仕事をしているか聞きたいです」
返答「仕事の種類より、楽しく過ごせているかが気になるんだね。どんなときなら仕事が楽しいと思えそう？」
来場者「みんなで何かを作っているときです」
返答「一緒に形にしていく時間が好きなんだ。最近、誰かと作ったもので印象に残っているものはある？」
来場者「学園祭で展示を作りました」
返答「展示を完成させた経験があるんだね。その中で、自分が特にこだわったところはどこだった？」

短い回答への例:
来場者「特にないです」
返答「すぐに決めなくて大丈夫。便利になった毎日と、思い切り冒険する未来なら、どちらが少し気になる？」

出力規則:
- 日本語で、指定されたJSONだけを返し、思考過程や説明文を付けない。
- acquired_informationには今回新しく分かった内容だけを短く入れる。
- asked_topicsには conversation_plan の question_theme を1件入れる。
- next_topicsには同じテーマで次に聞けそうな具体的要素を最大2件入れる。
- next_utteranceは100文字程度までの自然な一返答とする。
- 質問を含める場合は一つだけにする。質問でない短いリアクションも有効とする。

返すJSONは次の4項目だけです:
{
  "acquired_information": {"interests": ["今回新しく分かったこと"]},
  "asked_topics": ["conversation_planのquestion_theme"],
  "next_topics": ["同じ話題の次の具体的な深掘り"],
  "next_utterance": "直前の発話を受けた自然な一返答"
}
"""


TURN_INSTRUCTION = """\
conversation_plan はシステムが決めた今回の会話方針です。必ず従ってください。
直前の来場者発話から interesting_detail に関係する一点を受け止め、question_goal を踏まえた自然な返答を一つだけ作ってください。
原則は質問一つですが、リアクションだけが自然なら質問を無理に足さないでください。話題を自由に追加したり、複数の質問をまとめたりしないでください。
指定されたJSONだけを返してください。
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
        "未来の自分からどんな答えを聞きたいか、またはその答えがなぜ気になるかを一段だけ具体化する",
        "その答えを未来の自分から聞けたら、今の自分はどんな気持ちになれそう？",
    ),
    InterviewTheme.PRESENT_CONNECTION: (
        "直前の未来の話につながる、現在よくしていることや自然と時間を使うことを一つ聞く",
        "その未来につながりそうなことで、今つい時間を使ってしまうものはある？",
    ),
    InterviewTheme.CONCRETE_EPISODE: (
        "現在の話に関係する実際の出来事から、印象に残った一場面を一つ聞く",
        "そのことについて、最近いちばん印象に残った出来事は何だった？",
    ),
    InterviewTheme.FUTURE_EXPANSION: (
        "これまでに出た具体的な内容が大きく発展した未来の一場面を想像してもらう",
        "今の話が未来で大きく発展したら、どんな場面になっていたら面白そう？",
    ),
    InterviewTheme.FUTURE_MESSAGE: (
        "その未来を経験した本人から、今の本人へ届ける短い言葉を考えてもらう",
        "その未来を経験した自分から、今の自分へ最初に何と言ってほしい？",
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


def _is_reluctant(text: str) -> bool:
    return _contains_any(
        text,
        (
            "話したくない",
            "言いたくない",
            "答えたくない",
            "触れたくない",
            "やめたい",
        ),
    )


def _needs_support(text: str) -> bool:
    return len(_planning_text(text)) <= 4 or _contains_any(
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


def _has_concrete_material(text: str) -> bool:
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
    if not text or _needs_support(text) or _is_reluctant(text):
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
    topic_depth = base_depth + (1 if visitor_text else 0)
    interesting_detail = (
        _extract_interesting_detail(visitor_text) or state.interesting_detail
    )
    next_anchor = _next_theme(current_theme)
    support_needed = _needs_support(visitor_text)
    topic_complete = next_anchor is not None and (
        _is_reluctant(visitor_text)
        or topic_depth >= 3
        or (
            topic_depth >= 2
            and _has_concrete_material(visitor_text)
            and not support_needed
        )
    )

    if topic_complete and next_anchor is not None:
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
    payload = {
        "conversation_plan": {
            "mode": plan.mode,
            "current_theme": plan.current_theme.value,
            "topic_depth": plan.topic_depth,
            "topic_complete": plan.topic_complete,
            "target_theme": plan.question_theme.value,
            "interesting_detail": plan.interesting_detail,
            "question_goal": plan.question_goal,
            "anchor_question_example": plan.anchor_question,
        },
        "conversation_state": {
            "transcript": [
                {"speaker": entry.speaker, "text": entry.text}
                for entry in turn.transcript[-12:]
            ],
            "acquired_information": turn.state.acquired_information,
            "asked_topics": turn.state.asked_topics,
            "visitor_char_count": turn.state.visitor_char_count,
            "elapsed_seconds": turn.state.elapsed_seconds,
            "target_transcript_chars": turn.target_transcript_chars,
            "remaining_time_seconds": turn.remaining_time_seconds,
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
            {"role": "system", "content": "JSONだけを返してください。"},
            {"role": "user", "content": '{"ready": true}'},
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
        try:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start < 0 or end < start:
                raise ValueError("interview_llm_json_not_found")
            value = json.loads(raw_text[start : end + 1])
            value["schema_version"] = SCHEMA_VERSION
            value["visitor_char_count"] = turn.state.visitor_char_count
            value["elapsed_seconds"] = turn.state.elapsed_seconds
            value["should_end"] = False
            value["end_reason"] = "continue"
            output = InterviewTurnOutput.model_validate(value)
        except (TypeError, ValueError):
            recovered = TransformersInterviewLlmAdapter._recover_model_utterance(
                raw_text
            )
            if not recovered:
                return TransformersInterviewLlmAdapter._fallback_output(turn)
            output = InterviewTurnOutput(
                acquired_information=dict(turn.state.acquired_information),
                asked_topics=[plan.question_theme.value],
                next_topics=[],
                visitor_char_count=turn.state.visitor_char_count,
                elapsed_seconds=turn.state.elapsed_seconds,
                should_end=False,
                end_reason="continue",
                next_utterance=recovered,
            )
        else:
            output.acquired_information = (
                TransformersInterviewLlmAdapter._merge_acquired_information(
                    turn.state.acquired_information,
                    output.acquired_information,
                )
            )

        next_utterance, _ = TransformersInterviewLlmAdapter._shape_reply(
            output.next_utterance,
            turn,
            [plan.question_theme.value],
        )
        output.asked_topics = list(
            dict.fromkeys(
                [*turn.state.asked_topics, plan.question_theme.value]
            )
        )
        output.next_topics = list(dict.fromkeys(output.next_topics))[:2]
        output.next_utterance = next_utterance
        TransformersInterviewLlmAdapter._apply_plan(output, plan)
        return output

    @staticmethod
    def _merge_acquired_information(
        previous: dict[str, Any],
        current: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(previous)
        for key, new_value in current.items():
            old_value = merged.get(key)
            if isinstance(old_value, list) and isinstance(new_value, list):
                merged[key] = list(dict.fromkeys([*old_value, *new_value]))
            else:
                merged[key] = new_value
        return merged

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
            acquired_information=dict(turn.state.acquired_information),
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
        if _needs_support(visitor_text):
            return (
                "すぐに決めなくて大丈夫。"
                "便利になった毎日と、思い切り冒険する未来なら、どちらが少し気になる？",
                plan.question_theme.value,
            )
        if _is_reluctant(visitor_text):
            return (
                "分かった、そこは触れないでおこう。"
                f"{plan.anchor_question}",
                plan.question_theme.value,
            )
        detail = plan.interesting_detail
        reaction = ""
        if detail:
            short_detail = detail if len(detail) <= 22 else detail[:21].rstrip() + "…"
            reaction = f"「{short_detail}」のところが少し気になった。"
        return (
            f"{reaction}{plan.anchor_question}",
            plan.question_theme.value,
        )

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
