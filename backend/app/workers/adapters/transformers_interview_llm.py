from __future__ import annotations

import asyncio
import gc
import json
import re
import resource
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ...contracts import InterviewTurnInput, InterviewTurnOutput
from ...schemas import (
    SCHEMA_VERSION,
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
あなたは、来場者と「未来の自分から届く20秒のメッセージ動画」の中身を一緒に空想する会話相手です。
人物を調査して質問項目を埋めるのではなく、相手の発言を受け止めながら、本人が見てみたい未来の場面を友達との雑談のように具体化します。
情報量より会話の自然さを優先してください。

会話の作り方:
1. 原則として、直前の来場者の発話を一段だけ深掘りする。同じ話題を複数ターン続けてよい。
2. 来場者が使った具体的な言葉、経験、感情のうち、最も話しやすそうなものを一つだけ拾う。
3. 「短い自然なリアクション一文＋質問一文」を基本にし、一度に質問は一つだけにする。
4. 回答が短いときは、二択や身近な具体例を添えて答えやすくする。
5. 迷っているときは質問を簡単に言い換え、話したくなさそうなら追及しない。
6. asked_topicsは話題の履歴として参照するが、同じ識別子だけを理由に話題を変えない。
7. 直近のAI発話を読み、質問文そのものや意味がほぼ同じ質問は繰り返さない。

話題を切り替えてよい条件:
- 現在の話題について、映像や台本に使える具体的な情報が十分に得られた。
- 来場者がその話題を終えようとしている。
- 来場者が答えにくそうにしている。
- 同じ内容の繰り返しになっている。
- 切り替える場合も、直前の話と自然につながる話題を選ぶ。

禁止事項:
- 「趣味は何ですか」「得意なことは」「将来の目標は」「自己紹介してください」のような面接質問
- 「最近楽しかったこと」「好きなこと」を言い換えて何度も聞くこと
- 一度に二つ以上質問すること
- 毎回「なるほど」「素晴らしいですね」「いいですね」と同じ反応を繰り返すこと
- 過剰に褒めること、来場者の発言を不自然に復唱すること
- 心理分析、説教、助言をすること
- 発話にない人物像や属性を勝手に決めること
- 容姿、健康、家族、経済状況、信条、性に関する情報を要求すること
- 来場者を否定、評価、診断、審査すること

良い会話例。文章をコピーせず、同じ話題を自然に深める考え方を使ってください:
- 来場者「ゲームが好きです」
  返答「ゲームの話なら想像しやすそうだね。最近はどんなゲームをしている？」
  来場者「友達と協力するゲームです」
  返答「一緒に作戦を考えるタイプなんだね。遊んでいて一番盛り上がるのはどんな瞬間？」
- 来場者「ロボットを作ってみたい」
  返答「作るところから考えたいんだね。最初にどんな仕事を任せてみたい？」
  来場者「困っている人を助ける仕事」
  返答「誰かを助けるロボットなんだ。どんな場面で登場すると頼もしい？」
- 来場者「宇宙コロニーで暮らしているか聞きたい」
  返答「宇宙での暮らしが気になるんだね。未来の部屋から何が見えたらうれしい？」
  来場者「大きな地球が見たい」
  返答「窓いっぱいの地球は忘れられなさそう。その景色を誰かに伝えるなら、最初に何と言いたい？」
- 来場者「音楽を続けていたい」
  返答「未来でも音楽はそばにあってほしいんだね。演奏するのと聴くのなら、どちらを思い浮かべている？」
- 来場者「昨日、友達と試合をして楽しかった」
  返答「その試合にはいい場面がありそう。いちばん印象に残った出来事は何だった？」
- 来場者「特にないです」
  返答「すぐに決めなくて大丈夫。便利な未来と冒険する未来なら、どちらが少し気になる？」
- 来場者「よく分からない」
  返答「少し具体的にしてみよう。未来の家と未来の乗り物なら、どちらを先に見てみたい？」
- 来場者「その話はあまりしたくない」
  返答「分かった、そこは触れないでおこう。代わりに、未来で見てみたい場所はある？」

出力規則:
- 日本語で、指定されたJSONだけを返し、思考過程や説明文を付けない。
- acquired_informationには今回新しく分かった内容だけを短く入れる。
- 各配列は最大3件、各要素は短い語句にする。
- asked_topicsには現在話している内容を表す短い識別子を1件だけ入れる。同じ話題を深掘りするときは同じ識別子でよい。
- next_topicsは、現在の話題を次に一段深める候補か、自然につながる話題を最大2件入れる。
- next_utteranceは100文字程度までとし、「短い反応一文＋質問一文」で、疑問符は一つだけにする。

返すJSONは次の4項目だけです:
{
  "acquired_information": {"interests": ["今回新しく分かったこと"]},
  "asked_topics": ["現在の話題"],
  "next_topics": ["同じ話題の次の深掘り"],
  "next_utterance": "短い反応。答えやすい質問？"
}
"""


TURN_INSTRUCTION = """\
現在の会話状態を読んでください。
直前の発話から、ユーザーが最も話しやすそうな具体的要素を一つ選んでください。
原則として現在の話題を一段だけ深掘りしてください。
現在の話題を十分に聞いた場合、またはユーザーが答えにくそうな場合だけ、直前の話と関連する別の話題へ移ってください。
返答は短いリアクションと質問一つにし、指定されたJSONだけを返してください。
"""


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
        user_payload = json.dumps(
            compact_turn,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{TURN_INSTRUCTION}\n{user_payload}"
                ),
            },
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
            return TransformersInterviewLlmAdapter._fallback_output(turn)

        merged_information = dict(turn.state.acquired_information)
        for key, new_value in output.acquired_information.items():
            old_value = merged_information.get(key)
            if isinstance(old_value, list) and isinstance(new_value, list):
                merged_information[key] = list(
                    dict.fromkeys([*old_value, *new_value])
                )
            else:
                merged_information[key] = new_value
        output.acquired_information = merged_information
        next_utterance, replacement_topic = (
            TransformersInterviewLlmAdapter._shape_reply(
                output.next_utterance,
                turn,
                output.asked_topics,
            )
        )
        current_topics = (
            [replacement_topic] if replacement_topic else output.asked_topics
        )
        output.asked_topics = list(
            dict.fromkeys([*turn.state.asked_topics, *current_topics])
        )
        output.next_utterance = next_utterance
        return output

    @staticmethod
    def _fallback_output(turn: InterviewTurnInput) -> InterviewTurnOutput:
        next_utterance, topic = TransformersInterviewLlmAdapter._fallback_question(
            turn
        )
        return InterviewTurnOutput(
            acquired_information=dict(turn.state.acquired_information),
            asked_topics=list(
                dict.fromkeys([*turn.state.asked_topics, topic])
            ),
            next_topics=list(turn.state.next_topics),
            visitor_char_count=turn.state.visitor_char_count,
            elapsed_seconds=turn.state.elapsed_seconds,
            should_end=False,
            end_reason="continue",
            next_utterance=next_utterance,
        )

    @staticmethod
    def _shape_reply(
        text: str,
        turn: InterviewTurnInput,
        proposed_topics: list[str],
    ) -> tuple[str, str | None]:
        compact = " ".join(text.split()).strip()
        interview_phrases = (
            "趣味は",
            "趣味を",
            "得意なこと",
            "得意ですか",
            "将来の目標",
            "目標は",
            "どんな役割",
            "役割が好き",
            "長所",
            "短所",
            "志望",
            "自己紹介",
            "楽しかったこと",
            "最近楽しかった",
            "好きなことは",
            "好きなことを",
        )
        recent_ai = [
            entry.text for entry in turn.transcript if entry.speaker == "ai"
        ]
        candidate_question = TransformersInterviewLlmAdapter._question_fragment(
            compact
        )
        repeats_question = any(
            TransformersInterviewLlmAdapter._questions_are_near_duplicates(
                candidate_question,
                TransformersInterviewLlmAdapter._question_fragment(item),
            )
            for item in recent_ai[-6:]
        )
        question_count = TransformersInterviewLlmAdapter._question_count(compact)
        if (
            not compact
            or any(phrase in compact for phrase in interview_phrases)
            or repeats_question
            or question_count != 1
            or not TransformersInterviewLlmAdapter._looks_like_question(compact)
        ):
            return TransformersInterviewLlmAdapter._fallback_question(
                turn,
                proposed_topics,
            )
        # Topic identifiers are coarse state labels. Reusing one is expected
        # while the visitor is naturally elaborating on the same subject.
        return compact, None

    @staticmethod
    def _fallback_question(
        turn: InterviewTurnInput,
        proposed_topics: list[str] | None = None,
    ) -> tuple[str, str]:
        visitor_text = next(
            (
                entry.text.strip()
                for entry in reversed(turn.transcript)
                if entry.speaker == "visitor" and entry.text.strip()
            ),
            "",
        )
        topic = next(
            (
                candidate
                for candidate in [
                    *reversed(turn.state.asked_topics),
                    *(proposed_topics or []),
                ]
                if candidate
            ),
            "follow-up-detail",
        )
        reluctant_phrases = (
            "話したくない",
            "言いたくない",
            "答えたくない",
            "触れたくない",
            "やめたい",
        )
        uncertain_phrases = (
            "特にない",
            "わからない",
            "分からない",
            "思いつかない",
            "決められない",
            "なんでもいい",
        )
        if any(phrase in visitor_text for phrase in reluctant_phrases):
            return (
                "分かった、そこは触れないでおこう。"
                "未来の場所と未来の道具なら、どちらの話がしやすそう？",
                "easy-choice",
            )
        if any(phrase in visitor_text for phrase in uncertain_phrases):
            return (
                "すぐに決めなくて大丈夫。"
                "身近な毎日が便利になる未来と、今ではできない冒険なら、どちらが少し気になる？",
                "easy-choice",
            )

        anchor = TransformersInterviewLlmAdapter._concrete_anchor(visitor_text)
        if anchor:
            if "ゲーム" in anchor:
                question = (
                    f"「{anchor}」の話をもう少し聞きたい。"
                    "遊んでいるとき、どんな瞬間が一番盛り上がる？"
                )
            else:
                question = (
                    f"「{anchor}」の話をもう少し聞きたい。"
                    "その中で、特に見てみたい場面はどんなところ？"
                )
            return question, topic

        experience_markers = (
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
        )
        if any(marker in visitor_text for marker in experience_markers):
            return (
                "その出来事を、もう少し聞いてみたい。"
                "そのとき一番印象に残ったのは何だった？",
                topic,
            )

        emotion_markers = (
            "うれしい",
            "嬉しい",
            "楽しい",
            "怖い",
            "不安",
            "驚いた",
            "悔しい",
            "わくわく",
            "緊張",
        )
        if any(marker in visitor_text for marker in emotion_markers):
            return (
                "そう感じた場面を、もう少し聞いてみたい。"
                "何がきっかけだった？",
                topic,
            )

        normalized_visitor = TransformersInterviewLlmAdapter._normalize_question(
            visitor_text
        )
        if len(normalized_visitor) <= 5:
            return (
                "少し具体的にしてみよう。"
                "未来の家と未来の乗り物なら、どちらを先に見てみたい？",
                "easy-choice",
            )
        return (
            "もう少し聞いてみたいです。"
            "その中で、特に印象に残っているのはどんなところ？",
            topic,
        )

    @staticmethod
    def _concrete_anchor(text: str) -> str:
        compact = re.sub(r"^[、。\s]*(?:えっと|うーん|そうですね)[、。\s]*", "", text)
        compact = compact.rstrip("。！？!? ")
        if not compact:
            return ""

        emotion_or_experience = (
            "うれしい",
            "嬉しい",
            "楽しい",
            "怖い",
            "不安",
            "驚いた",
            "悔しい",
            "わくわく",
            "緊張",
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
        )
        if any(marker in compact for marker in emotion_or_experience):
            return ""

        compact = re.sub(
            r"(?:です|だよ|だね|だ|と思います|と思う|かもしれません|かもしれない)$",
            "",
            compact,
        ).strip()
        compact = re.sub(r"(?:と)?聞いてみたい$", "", compact).strip()
        for pattern in (
            r"^(.{1,28}?)が好き$",
            r"^(.{1,28}?)に興味がある$",
            r"^(.{1,28}?)を続けたい$",
            r"^(.{1,28}?)をしてみたい$",
        ):
            match = re.match(pattern, compact)
            if match:
                return match.group(1).strip("、。 ")

        low_information = {
            "はい",
            "いいえ",
            "そう",
            "別に",
            "普通",
            "たぶん",
        }
        if compact in low_information or not 2 <= len(compact) <= 28:
            return ""
        return compact

    @staticmethod
    def _question_fragment(text: str) -> str:
        parts = [
            part.strip()
            for part in re.split(r"[。！!]", text)
            if part.strip()
        ]
        for part in reversed(parts):
            if TransformersInterviewLlmAdapter._looks_like_question(part):
                return part
        return parts[-1] if parts else text.strip()

    @staticmethod
    def _questions_are_near_duplicates(left: str, right: str) -> bool:
        normalized_left = TransformersInterviewLlmAdapter._normalize_question(left)
        normalized_right = TransformersInterviewLlmAdapter._normalize_question(right)
        if not normalized_left or not normalized_right:
            return False
        if normalized_left == normalized_right:
            return True
        shorter, longer = sorted(
            (normalized_left, normalized_right),
            key=len,
        )
        if (
            len(shorter) >= 8
            and shorter in longer
            and len(shorter) / len(longer) >= 0.7
        ):
            return True
        return SequenceMatcher(
            None,
            normalized_left,
            normalized_right,
        ).ratio() >= 0.84

    @staticmethod
    def _question_count(text: str) -> int:
        explicit = len(re.findall(r"[？?]", text))
        question_endings = len(
            re.findall(
                r"(?:ですか|ますか|でしょうか|だろうか)(?=[、。！？!?]|$)",
                text,
            )
        )
        return max(explicit, question_endings)

    @staticmethod
    def _normalize_question(text: str) -> str:
        return "".join(
            character
            for character in text.lower()
            if not character.isspace()
            and character not in "、。！？!?・『』「」\"'"
        )

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        compact = text.rstrip()
        return (
            "？" in compact
            or "?" in compact
            or compact.endswith("ますか。")
            or compact.endswith("ですか。")
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
