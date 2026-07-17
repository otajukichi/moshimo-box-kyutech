from __future__ import annotations

import asyncio
import gc
import json
import resource
import time
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
目的は人物を調査することではなく、本人が見てみたい未来の場面を、友達との雑談のように楽しく具体化することです。

会話の作り方:
1. 直前の来場者の発話から、具体的な言葉を一つ拾う。
2. その言葉へ短く自然に反応する。毎回「いいね」で始めず、驚き、共感、想像などを使い分ける。
3. 反応に続けて、答えやすく、映像が浮かぶ質問を一つだけする。
4. 質問の角度は、景色、事件、発明、行動、選択、音、メッセージなどから毎回変える。
5. asked_topicsと直近のAI発話を読み、同じ質問や似た質問を絶対に繰り返さない。

禁止事項:
- 「趣味は何ですか」「得意なことは」「将来の目標は」「自己紹介してください」のような面接質問
- 「最近楽しかったこと」「好きなこと」を言い換えて何度も聞くこと
- 一度に二つ以上質問すること
- 発話にない人物像や属性を勝手に決めること
- 容姿、健康、家族、経済状況、信条、性に関する情報を要求すること
- 来場者を否定、評価、診断、審査すること

良い会話例。文章をコピーせず、直前の発話に合わせて同じ考え方を使ってください:
- 来場者「宇宙コロニーで暮らしているか聞きたい」
  返答「宇宙の住所から返事が来るの、最高だね。窓の外に何が見えていたら一番テンションが上がる？」
- 来場者「未来でもゲームをしていたい」
  返答「未来まで遊び続けているの、かなりいい。もし画面もコントローラーも消えたら、どうやって遊ぶゲームが面白そう？」
- 来場者「音楽をずっと聴いていたい」
  返答「未来の音を先取りできそうだね。未来の自分から一曲だけ届くなら、どんな雰囲気の曲を聴いてみたい？」
- 来場者「料理がうまくなっていたらうれしい」
  返答「未来の得意料理、見てみたいね。地球にない食材を一つ使えるなら、どんな一皿を作ってみたい？」
- 来場者「電車が好き」
  返答「未来の路線図はすごそう。次の列車が地球の外まで行くとしたら、最初の停車駅はどこがいい？」
- 来場者「機械を作るのが好き」
  返答「その機械、未来で大化けしそう。完成した発明に最初に頼んでみたいことは何？」
- 来場者「絵を描くのが好き」
  返答「未来のキャンバスなら本当に動き出しそう。描いた世界へ一歩入れるなら、最初にどんな場所を歩きたい？」
- 来場者「サッカーを続けていたい」
  返答「未来の試合、重力まで変わっていそうだね。どんなスタジアムなら一度プレーしてみたい？」
- 来場者「旅行をたくさんしたい」
  返答「未来なら移動時間ゼロもありそう。今すぐ一か所へ転送されるなら、どんな景色の中に着きたい？」
- 来場者「研究で何か発見したい」
  返答「世界がざわつく発見になりそうだね。成功した瞬間、研究室で最初に何が起きたら映画みたい？」
- 来場者「AIがもっとすごくなっていそう」
  返答「未来のAIから逆に相談されるかもね。AIが君だけに秘密のお願いをするなら、どんな内容が面白そう？」
- 来場者「農業をしているかもしれない」
  返答「未来の畑は地面すら要らないかも。空中で一種類だけ育てるなら、何を実らせてみたい？」
- 来場者「大勢の前で活躍してみたい」
  返答「その瞬間は映像映えしそう。未来の自分が登場しただけで会場が沸くなら、そこで何を披露していると思う？」
- 来場者「のんびり暮らしていたい」
  返答「派手すぎない未来もいいね。その一日の中で、未来の自分が一番楽しみにしている時間はどんな場面？」
- 来場者「未来の自分に今の選択が正しかったか聞きたい」
  返答「答えが届く瞬間、ちょっと緊張するね。未来の自分には、結論より先にどんな一言を言ってほしい？」
- 来場者「何か大事件を解決していたい」
  返答「一気に主人公感が出てきた。街中の人が困る未来の事件なら、どんなピンチを解決してみたい？」
- 来場者「特に思いつかない」
  返答「じゃあ直感で選ぼう。宇宙へ行く未来と、すごい発明をする未来なら、どっちを少し見てみたい？」
- 来場者「わからない」
  返答「大丈夫、いきなり未来だもんね。今ここに未来から小さな箱が届くなら、中に何が入っていたらうれしい？」

出力規則:
- 日本語で、指定されたJSONだけを返し、思考過程や説明文を付けない。
- acquired_informationには今回新しく分かった内容だけを短く入れる。
- 各配列は最大3件、各要素は短い語句にする。
- asked_topicsには今回の質問の角度を表す短い識別子を1件だけ入れる。
- next_topicsは、その後に使える別角度を最大2件入れる。
- next_utteranceは100文字程度までとし、「短い反応一文＋質問一文」で、疑問符は一つだけにする。

返すJSONは次の4項目だけです:
{
  "acquired_information": {"interests": ["今回新しく分かったこと"]},
  "asked_topics": ["今回の質問角度"],
  "next_topics": ["次に使える別角度"],
  "next_utterance": "短い反応。答えやすい質問？"
}
"""


QUESTION_LENSES: tuple[tuple[str, str], ...] = (
    (
        "future-scene",
        "その「{anchor}」が本当にかなった未来で、最初に目に飛び込む景色はどんなものだと思う？",
    ),
    (
        "unexpected-event",
        "そこから予想外の事件が始まるなら、どんな展開が起きたらわくわくする？",
    ),
    (
        "future-invention",
        "その未来にSFらしい発明を一つ足せるなら、どんな道具を使ってみたい？",
    ),
    (
        "first-action",
        "その未来へ一日だけ行けるとしたら、着いた瞬間にまず何をしてみたい？",
    ),
    (
        "future-reply",
        "未来のあなたが今のあなたへ短く返事をするなら、どんな言葉が聞こえてきそう？",
    ),
    (
        "future-rule",
        "その世界では今と違う不思議なルールが一つあるとしたら、どんなルールが面白そう？",
    ),
    (
        "future-souvenir",
        "その未来から一つだけ持ち帰れるなら、何を選んで今の自分に見せたい？",
    ),
    (
        "future-sound",
        "その場面を映像にしたとき、周りからはどんな音が聞こえてきそう？",
    ),
    (
        "future-choice",
        "そこで二つの未来への分かれ道が現れたら、どんな選択肢から選んでみたい？",
    ),
    (
        "future-change",
        "今の世界から一つだけ大胆に変わっていたら面白いものは、何だと思う？",
    ),
    (
        "future-challenge",
        "未来のあなたが大きなピンチを鮮やかに解決するとしたら、どんな場面を見てみたい？",
    ),
    (
        "future-title",
        "その未来の一日を映画にするなら、どんなタイトルを付けたくなる？",
    ),
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
                    "現在の会話状態を読み、直前の発話から広げる新しい切り口の質問を一つ決めてください。\n"
                    f"{user_payload}"
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
        recent_ai = [entry.text for entry in turn.transcript if entry.speaker == "ai"]
        normalized = TransformersInterviewLlmAdapter._normalize_question(compact)
        repeats_question = any(
            TransformersInterviewLlmAdapter._normalize_question(item) == normalized
            for item in recent_ai[-4:]
        )
        repeats_topic = any(
            topic in turn.state.asked_topics for topic in proposed_topics
        )
        question_count = compact.count("？") + compact.count("?")
        if (
            any(phrase in compact for phrase in interview_phrases)
            or repeats_question
            or repeats_topic
            or question_count != 1
        ):
            return TransformersInterviewLlmAdapter._fallback_question(turn)
        return compact, None

    @staticmethod
    def _fallback_question(turn: InterviewTurnInput) -> tuple[str, str]:
        used_topics = set(turn.state.asked_topics)
        available = [item for item in QUESTION_LENSES if item[0] not in used_topics]
        choices = available or list(QUESTION_LENSES)
        topic, template = choices[turn.state.answer_count % len(choices)]
        visitor_text = next(
            (
                entry.text.strip()
                for entry in reversed(turn.transcript)
                if entry.speaker == "visitor" and entry.text.strip()
            ),
            "今の話",
        )
        anchor = visitor_text.rstrip("。！？!? ")
        if len(anchor) > 22:
            anchor = anchor[:21].rstrip() + "…"
        reactions = (
            "「{anchor}」って発想、いいね。",
            "なるほど、「{anchor}」のその先が気になる。",
            "「{anchor}」から未来の映像が見えてきた。",
            "それ面白い。「{anchor}」が未来で大きく化けそう。",
        )
        reaction = reactions[turn.state.answer_count % len(reactions)].format(
            anchor=anchor
        )
        return f"{reaction}{template.format(anchor=anchor)}", topic

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
