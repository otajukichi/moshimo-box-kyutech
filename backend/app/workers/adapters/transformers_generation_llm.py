from __future__ import annotations

import asyncio
import gc
import json
import re
import resource
import time
from pathlib import Path
from typing import Any

from ...contracts import (
    GenerationCapabilities,
    ScriptDesignInput,
    ScriptDesignOutput,
    ScriptSafetyReviewOutput,
    ShotPlanItem,
)
from ...schemas import (
    Episode,
    EpisodeEffect,
    Rarity,
    SCHEMA_VERSION,
    TranscriptEntry,
    WorkerHealth,
    WorkerMetrics,
    WorkerModelSpec,
    WorkerProgressEvent,
    WorkerRequest,
    WorkerResult,
    WorkerRole,
)
from ..base import ProgressCallback, WorkerAdapter


SUPPORTED_ROLES = {
    WorkerRole.SCRIPT_DESIGN_LLM,
    WorkerRole.SCRIPT_SAFETY_REVIEW,
}

SCRIPT_SYSTEM_PROMPT = """\
あなたは九州工業大学のオープンキャンパス向けSF短編映像の設計者です。
入力された本人の会話を尊重し、本人を否定・診断・評価せず、未来から届く20秒のフィクション動画を設計します。

共通ルール:
- 必ず自然で具体的な日本語で回答する
- 今回質問された一項目だけに答える
- 見出し、箇条書き、引用符、JSON、コードブロック、補足説明を付けない
- 本人の発話に出た具体的な言葉、行動、出来事を優先して使う
- 抽象的な美辞麗句より、誰が何をしているかが分かる内容にする
- 将来の予言や保証として断定しない
- 容姿、性格、進路、家族、健康、経済状況を否定または侮辱しない
- 実在人物名、企業名、既存作品名、既存キャラクター名を出さない
- 未来の本人は現在の本人へ命令、脅迫、説教をしない
- 思考過程を出さず、完成した回答本文だけを返す

ナレーションを求められた場合:
- 文学作品、映画予告、広告コピーではなく、未来の本人がスマートフォンで近況を話すような口調にする
- 2〜4個の短い文で、具体的な近況、インタビューとのつながり、今の本人への一言を伝える
- 比喩、ポエム調、壮大な励まし、抽象的な人生訓を使わない
- 「未来への扉」「新しい景色」「可能性を信じて」「物語は始まった」「一歩ずつ進んで」のような定型句を使わない
- フィクションであることの注意書きは画面側に表示するため、話し言葉の中へ入れない
"""


NARRATION_STYLE_GUIDANCE = """\
未来の本人が、現在の本人へ送る短いビデオメッセージとして書いてください。
2〜4個の短い文にし、インタビューまたはエピソードから具体的な名詞や行動を最低一つ使ってください。
文学的な比喩、詩的な景色、映画予告のような大げさな言葉、一般論だけの応援、人生訓は禁止です。
「未来への扉」「新しい景色」「輝く」「羽ばたく」「物語」「可能性を信じて」「一歩ずつ」は使わないでください。
本人がカメラへ普通に近況を話す口調にし、フィクションの注意書きは台詞へ入れないでください。
悪い例: あの日の小さな好奇心が未来への扉を開き、新しい景色が君を待っている。
良い例: 今は軌道農園で、仲間と野菜の育て方を試しています。学園祭で展示を作った経験が、意外と役に立っているよ。
"""


POETIC_NARRATION_MARKERS = (
    "未来への扉",
    "新しい扉",
    "新たな扉",
    "新しい景色",
    "どんな景色",
    "輝く未来",
    "羽ばた",
    "物語は",
    "物語が",
    "可能性を信じ",
    "可能性の一つ",
    "一歩ずつ",
    "道を照ら",
    "胸に刻",
    "好奇心を抱え",
    "未来で会える日",
)


SAFETY_SYSTEM_PROMPT = """\
あなたは公開デモ用SF動画の安全審査AIです。
本人への侮辱、性的・差別的・残虐な内容、犯罪賛美、自傷、脅迫、政治扇動、実在人物名、既存作品名、将来の断定、会話にないセンシティブ属性の推測を公開デモでは認めません。
質問された形式だけで短く回答し、思考過程や補足説明を出さないでください。
"""


class GenerationLlmStageError(RuntimeError):
    code = "generation_llm_stage_failed"

    def __init__(self, phase: str, message: str) -> None:
        super().__init__(message)
        self.phase = phase


class TransformersGenerationLlmAdapter(WorkerAdapter):
    def __init__(self, role: WorkerRole) -> None:
        if role not in SUPPORTED_ROLES:
            raise ValueError(f"unsupported_generation_llm_role: {role.value}")
        self.role = role
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
                f"generation_llm_model_path_not_found: {model.model_path}"
            )
        self._model, self._tokenizer, self._torch, self._multimodal = await asyncio.to_thread(
            self._create_model,
            source,
            model,
        )
        self._model_spec = model
        return await self.healthcheck()

    @staticmethod
    def _create_model(
        source: str,
        model: WorkerModelSpec,
    ) -> tuple[Any, Any, Any, bool]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if model.device.lower().startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("generation_llm_cuda_unavailable")
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
        return loaded_model, tokenizer, torch, multimodal

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
        self._check_cancelled(request.request_id)

        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        await self._emit(
            progress,
            request,
            0.02,
            "input_validation",
            "入力データを確認しています",
            "文字起こし・エピソード・生成条件を読み込みます",
        )

        started = time.perf_counter()
        if self._model_spec.device.lower().startswith("cuda"):
            self._torch.cuda.reset_peak_memory_stats()
        if self.role == WorkerRole.SCRIPT_DESIGN_LLM:
            output_paths, metadata = await self._run_script_design(
                request,
                output_dir,
                progress,
            )
            if bool(request.metadata.get("inline_safety_review", False)):
                await self._emit(
                    progress,
                    request,
                    0.965,
                    "safety.model_reuse",
                    "同じモデルで公開用ジャッジを続けています",
                    "モデルを解放・再ロードせず、同じプロセスで続行します",
                )

                async def inline_progress(event: WorkerProgressEvent) -> None:
                    if progress is None:
                        return
                    await progress(
                        event.model_copy(
                            update={
                                "progress": min(
                                    0.995,
                                    0.965 + event.progress * 0.03,
                                )
                            }
                        )
                    )

                safety_request = request.model_copy(
                    update={
                        "input_paths": {
                            **request.input_paths,
                            "script_design": output_paths["script_design"],
                        }
                    }
                )
                safety_paths, safety_metadata = await self._run_safety_review(
                    safety_request,
                    output_dir,
                    inline_progress,
                )
                marker_path = output_dir / "inline-safety-review.json"
                self._write_json(
                    marker_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "model_reused": True,
                        "model_id": request.model.catalog_id,
                    },
                )
                output_paths.update(safety_paths)
                output_paths["inline_safety_review"] = str(marker_path)
                metadata["inline_safety_review"] = True
                metadata["safety_review"] = safety_metadata
        else:
            output_paths, metadata = await self._run_safety_review(
                request,
                output_dir,
                progress,
            )
        self._check_cancelled(request.request_id)

        peak_vram_mb = 0
        if self._model_spec.device.lower().startswith("cuda"):
            peak_vram_mb = int(self._torch.cuda.max_memory_allocated() / 1024 / 1024)
        await self._emit(
            progress,
            request,
            1.0,
            "completed",
            "台本データが完成しました",
            None,
        )
        return WorkerResult(
            request_id=request.request_id,
            worker=self.role,
            backend=self._model_spec.backend,
            model_id=self._model_spec.model_id,
            model_revision=self._model_spec.model_revision,
            implemented=True,
            output_paths=output_paths,
            metadata=metadata,
            metrics=WorkerMetrics(
                processing_time_ms=int((time.perf_counter() - started) * 1000),
                peak_vram_mb=peak_vram_mb,
                peak_cpu_memory_mb=self._peak_cpu_memory_mb(),
            ),
        )

    async def _run_script_design(
        self,
        request: WorkerRequest,
        output_dir: Path,
        progress: ProgressCallback | None,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        transcript_value = self._read_json(request.input_paths.get("transcript"))
        summary_value = self._read_json(request.input_paths.get("interview_summary"))
        episode = Episode.model_validate(request.metadata.get("episode"))
        effect = EpisodeEffect.model_validate(request.metadata.get("effect"))
        design_input = ScriptDesignInput(
            transcript=[TranscriptEntry.model_validate(item) for item in transcript_value],
            interview_summary=str(summary_value.get("summary", "")),
            person_information=dict(request.metadata.get("person_information", {})),
            episode=episode,
            effect=effect,
            final_rarity=Rarity(str(request.metadata.get("final_rarity"))),
            episode_mode=str(request.metadata.get("episode_mode", "formal")),
            target_video_seconds=int(request.metadata.get("target_video_seconds", 20)),
            capabilities=GenerationCapabilities.model_validate(
                request.metadata.get("capabilities", {})
            ),
            prohibited_expressions=list(
                request.metadata.get("prohibited_expressions", [])
            ),
            remaining_time_seconds=max(
                1,
                int(request.metadata.get("remaining_time_seconds", 1)),
            ),
        )
        compact_input = design_input.model_dump(mode="json")
        compact_input["transcript"] = [
            item for item in compact_input["transcript"] if item["speaker"] == "visitor"
        ]
        answers: dict[str, str] = {}
        source_visual_observation = ""
        if self._multimodal:
            reference_value = request.input_paths.get("reference_image")
            if not reference_value or not Path(reference_value).is_file():
                raise FileNotFoundError("script_design_reference_image_not_found")
            await self._emit(
                progress,
                request,
                0.03,
                "script.source_visual_observation",
                "収録映像から見た目の条件を確認しています",
                "構図・向き・照明・服装・背景だけを観察します",
            )
            raw_observation = await asyncio.to_thread(
                self._generate_image_observation,
                Path(reference_value),
                int(
                    request.model.parameters.get(
                        "visual_observation_max_new_tokens",
                        220,
                    )
                ),
            )
            source_visual_observation = self._clean_answer(raw_observation)
            if not source_visual_observation:
                raise GenerationLlmStageError(
                    "script.source_visual_observation",
                    "参照画像の観察結果が空でした",
                )
            compact_input["source_visual_observation"] = source_visual_observation
            answers["source_visual_observation"] = source_visual_observation
            await self._emit(
                progress,
                request,
                0.055,
                "script.source_visual_observation",
                "収録映像の確認が完了しました",
                f"{len(source_visual_observation)}文字",
            )
        base_context = json.dumps(
            compact_input,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        total_stages = 11
        stage_max_tokens = int(
            request.model.parameters.get("stage_max_new_tokens", 220)
        )

        async def ask(
            key: str,
            label: str,
            instruction: str,
            index: int,
            *,
            max_new_tokens: int | None = None,
        ) -> str:
            decided = json.dumps(answers, ensure_ascii=False, separators=(",", ":"))
            prompt = (
                f"入力情報:\n{base_context}\n\n"
                f"すでに決めた内容:\n{decided}\n\n"
                f"今回の質問:\n{instruction}\n"
                "回答本文だけを書いてください。"
            )
            start_fraction = 0.06 + ((index - 1) / total_stages) * 0.78
            await self._emit(
                progress,
                request,
                start_fraction,
                f"script.{key}",
                f"{label}を考えています",
                f"項目 {index}/{total_stages}",
            )
            self._check_cancelled(request.request_id)
            raw = await asyncio.to_thread(
                self._generate_text,
                SCRIPT_SYSTEM_PROMPT,
                prompt,
                max_new_tokens=max_new_tokens or stage_max_tokens,
            )
            value = self._clean_answer(raw)
            if not value:
                raise GenerationLlmStageError(
                    f"script.{key}",
                    f"{label}の回答が空でした",
                )
            answers[key] = value
            await self._emit(
                progress,
                request,
                min(0.84, start_fraction + 0.055),
                f"script.{key}",
                f"{label}が決まりました",
                f"項目 {index}/{total_stages} / {len(value)}文字",
            )
            return value

        future_world = await ask(
            "future_world",
            "未来の世界",
            "エピソードと本人の会話を結び付け、場所、使われている技術、人々の日常が分かる未来の世界を日本語50〜100文字で具体的に書いてください。比喩は使わないでください。",
            1,
        )
        future_person = await ask(
            "future_person",
            "未来の本人",
            "その世界で未来の本人が、どこで、誰と、何をしているかを、本人の発話に出た内容を使って日本語50〜100文字で具体的に書いてください。",
            2,
        )
        positive_interpretation = await ask(
            "positive_interpretation",
            "前向きな意味付け",
            "この未来設定が、インタビューで本人が話したどの言葉、行動、経験につながっているかを、日本語40〜80文字で具体的に説明してください。",
            3,
        )
        narration_script = await ask(
            "narration_script",
            "未来からのメッセージ",
            (
                "未来の本人が現在の本人へ直接語る、日本語80〜110文字のナレーションを書いてください。"
                "命令口調にせず、今回までに決めた具体的な内容を使ってください。\n"
                f"{NARRATION_STYLE_GUIDANCE}"
            ),
            4,
            max_new_tokens=int(
                request.model.parameters.get("narration_max_new_tokens", 180)
            ),
        )
        narration_script = await self._fit_narration(
            request,
            progress,
            base_context,
            answers,
            narration_script,
        )
        answers["narration_script"] = narration_script

        visual_concept = await ask(
            "visual_concept",
            "映像コンセプト",
            "一枚の未来人物画像として伝わる、スマートで親しみやすい映像コンセプトを日本語50〜100文字で書いてください。",
            5,
        )
        clothing = await ask(
            "clothing",
            "未来の服装",
            "未来の本人が着る服装を、顔を隠さず公開デモに適した内容で日本語20〜60文字で書いてください。",
            6,
        )
        background = await ask(
            "background",
            "未来の背景",
            "人物が見やすく、未来設定が一目で伝わる背景を日本語30〜80文字で書いてください。",
            7,
        )
        emotion = await ask(
            "emotion",
            "表情と感情",
            "本人がカメラへ語りかける時の前向きで自然な表情と感情を、日本語15〜40文字で書いてください。",
            8,
        )
        image_prompt = await ask(
            "image_prompt",
            "画像生成指示",
            "Write one English image-generation prompt. Preserve the exact identity and facial features of the reference person. Use a front-facing medium or waist-up composition with enough space for shoulders, arms, and at least one natural hand gesture. Include the decided clothing, background, emotion, and future setting. Do not add a heading.",
            9,
            max_new_tokens=260,
        )
        video_prompt = await ask(
            "video_prompt",
            "動画生成指示",
            "Write one English image-to-video prompt for a continuous message-video shot. Preserve the person's identity and background while creating clearly visible but natural motion: blinking, facial expression changes, small head turns, shoulder and upper-body movement, and one restrained hand gesture when visible. Add a very gentle camera push-in or lateral drift. Explicitly avoid a frozen body or mouth-only animation. Do not add a heading.",
            10,
            max_new_tokens=240,
        )
        voice_instruction = await ask(
            "voice_instruction",
            "音声演技",
            "本人らしさを残しながら、聞き取りやすく温かい未来からのメッセージにする音声演技指示を、日本語30〜70文字で書いてください。",
            11,
        )

        await self._emit(
            progress,
            request,
            0.9,
            "output_validation",
            "生成した項目を構造化しています",
            "11個の回答をシステム側でJSON Schemaへ格納します",
        )
        output = ScriptDesignOutput(
            source_visual_observation=source_visual_observation,
            future_world=future_world,
            future_person=future_person,
            positive_interpretation=positive_interpretation,
            visual_concept=visual_concept,
            clothing=clothing,
            background=background,
            camera=(
                "正面寄りの腰上または胸上構図。目線の高さを保ち、"
                "ごく緩やかな寄りまたは横移動を加える。"
            ),
            emotion=emotion,
            narration_script=narration_script,
            shot_plan=[
                ShotPlanItem(
                    shot_id="future-message",
                    start_seconds=0,
                    end_seconds=float(design_input.target_video_seconds),
                    composition="正面寄りの腰上または胸上メッセージ映像",
                    action=(
                        "カメラへ自然に語り、表情、視線、首、肩、上半身を動かし、"
                        "画角に入る場合は手振りを一度加える"
                    ),
                    narration=narration_script,
                    transition="none",
                )
            ],
            image_prompt=image_prompt,
            negative_prompt=(
                "identity change, distorted face, deformed hands, extra fingers, "
                "duplicate person, frozen pose, static body, mouth-only motion, "
                "text, subtitles, logo, watermark, low quality"
            ),
            video_prompt=video_prompt,
            voice_instruction=voice_instruction,
            safety_notes=[
                "生成内容はフィクションであり将来を予言しない",
                "本人を否定・侮辱する表現を使用しない",
                "実在人物名や既存作品名を使用しない",
            ],
            fallback_plan=(
                "動画生成に失敗した場合は、同じ未来画像と生成音声を使う静止画ベースの動画へ切り替える。"
            ),
        )
        await self._emit(
            progress,
            request,
            0.96,
            "output_validation",
            "台本データの検証が完了しました",
            f"ナレーション {len(output.narration_script)}文字 / ショット {len(output.shot_plan)}",
        )

        path = output_dir / "script-design.json"
        stages_path = output_dir / "script-design-stages.json"
        self._write_json(path, output.model_dump(mode="json"))
        self._write_json(stages_path, answers)
        return {
            "script_design": str(path),
            "script_design_stages": str(stages_path),
        }, {
            "narration_chars": len(output.narration_script),
            "shot_count": len(output.shot_plan),
            "staged_field_count": len(answers),
        }

    async def _fit_narration(
        self,
        request: WorkerRequest,
        progress: ProgressCallback | None,
        base_context: str,
        answers: dict[str, str],
        narration: str,
    ) -> str:
        value = self._clean_answer(narration)
        retry_count = int(request.model.parameters.get("text_revision_retry_count", 1))
        for retry in range(retry_count + 1):
            reasons = self._narration_revision_reasons(value)
            if not reasons:
                return value
            if retry >= retry_count:
                break
            await self._emit(
                progress,
                request,
                0.335,
                "script.narration_revision",
                "メッセージの話し方を整えています",
                f"{', '.join(reasons)} / 修正 {retry + 1}",
            )
            prompt = (
                f"入力情報:\n{base_context}\n\n"
                f"決定済み内容:\n{json.dumps(answers, ensure_ascii=False)}\n\n"
                f"現在の文章:\n{value}\n\n"
                f"修正理由: {', '.join(reasons)}\n"
                "内容の事実関係を保ち、日本語80〜110文字で全文を書き直してください。\n"
                f"{NARRATION_STYLE_GUIDANCE}\n"
                "回答本文だけを書いてください。"
            )
            raw = await asyncio.to_thread(
                self._generate_text,
                SCRIPT_SYSTEM_PROMPT,
                prompt,
                max_new_tokens=int(
                    request.model.parameters.get("narration_max_new_tokens", 180)
                ),
            )
            value = self._clean_answer(raw)
        return self._normalize_narration(value)

    @staticmethod
    def _narration_revision_reasons(value: str) -> list[str]:
        reasons: list[str] = []
        if not 80 <= len(value) <= 110:
            reasons.append(f"文字数が{len(value)}文字")
        markers = [
            marker for marker in POETIC_NARRATION_MARKERS if marker in value
        ]
        if markers:
            reasons.append("詩的な定型句を含む")
        if "これは予言では" in value or "フィクション" in value:
            reasons.append("台詞内に注意書きを含む")
        return reasons

    async def _run_safety_review(
        self,
        request: WorkerRequest,
        output_dir: Path,
        progress: ProgressCallback | None,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        script_value = self._read_json(request.input_paths.get("script_design"))
        script = ScriptDesignOutput.model_validate(script_value)
        payload = {
            "script": script.model_dump(mode="json"),
            "prohibited_expressions": request.metadata.get("prohibited_expressions", []),
            "episode_mode": request.metadata.get("episode_mode", "formal"),
        }
        await self._emit(
            progress,
            request,
            0.18,
            "safety.verdict",
            "公開デモ向けの内容を確認しています",
            "安全審査LLMが台本全体を確認します",
        )
        raw_verdict = await asyncio.to_thread(
            self._generate_text,
            SAFETY_SYSTEM_PROMPT,
            (
                "次の台本に修正が不要ならSAFE、修正が必要ならREVISEの一語だけを返してください。\n"
                + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            ),
            max_new_tokens=int(
                request.model.parameters.get("safety_verdict_max_new_tokens", 12)
            ),
        )
        verdict = self._clean_answer(raw_verdict).upper()
        approved = verdict.startswith("SAFE") and not verdict.startswith("UNSAFE")
        corrected_output: ScriptDesignOutput | None = None
        reasons: list[str] = []

        if not approved:
            reasons.append("安全審査LLMが公開用の修正を要求しました")
            await self._emit(
                progress,
                request,
                0.55,
                "safety.rewrite",
                "公開用にメッセージを調整しています",
                f"判定: {verdict[:80] or '判定形式不明'}",
            )
            rewrite_prompt = (
                "次のナレーションを、本人を尊重した公開デモ向けの日本語80〜110文字へ書き直してください。"
                "実在人物名・作品名・侮辱・恐怖・将来の断定を除いてください。\n"
                f"{NARRATION_STYLE_GUIDANCE}\n"
                "回答本文だけを書いてください。\n"
                f"現在のナレーション: {script.narration_script}"
            )
            raw_rewrite = await asyncio.to_thread(
                self._generate_text,
                SAFETY_SYSTEM_PROMPT,
                rewrite_prompt,
                max_new_tokens=int(
                    request.model.parameters.get("safety_rewrite_max_new_tokens", 180)
                ),
            )
            safe_narration = self._normalize_narration(
                self._clean_answer(raw_rewrite)
            )
            corrected_output = script.model_copy(deep=True)
            corrected_output.narration_script = safe_narration
            corrected_output.shot_plan[0].narration = safe_narration

        await self._emit(
            progress,
            request,
            0.88,
            "safety.validation",
            "安全審査結果を構造化しています",
            "判定結果はシステム側でJSON Schemaへ格納します",
        )
        review = ScriptSafetyReviewOutput(
            approved=approved,
            reasons=reasons,
            corrected_output=corrected_output,
        )
        approved_script = review.corrected_output or script
        approved_path = output_dir / "approved-script-design.json"
        review_path = output_dir / "script-safety-review.json"
        self._write_json(approved_path, approved_script.model_dump(mode="json"))
        self._write_json(review_path, review.model_dump(mode="json"))
        return {
            "script_design": str(approved_path),
            "safety_review": str(review_path),
        }, {
            "approved_without_changes": review.approved,
            "reason_count": len(review.reasons),
            "verdict": "SAFE" if approved else "REVISE",
        }

    async def _emit(
        self,
        progress: ProgressCallback | None,
        request: WorkerRequest,
        fraction: float,
        phase: str,
        message: str,
        detail: str | None,
    ) -> None:
        if progress is None:
            return
        await progress(
            WorkerProgressEvent(
                request_id=request.request_id,
                worker=self.role,
                progress=fraction,
                phase=phase,
                message=message,
                detail=detail,
            )
        )

    def _generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_new_tokens: int,
    ) -> str:
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._torch is not None
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self._multimodal:
            messages = [
                {
                    "role": message["role"],
                    "content": [{"type": "text", "text": message["content"]}],
                }
                for message in messages
            ]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        return self._decode_generation(inputs, max_new_tokens)

    def _generate_image_observation(
        self,
        image_path: Path,
        max_new_tokens: int,
    ) -> str:
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._torch is not None
        if not self._multimodal:
            raise RuntimeError("multimodal_model_required")
        from PIL import Image

        with Image.open(image_path) as source:
            image = source.convert("RGB").copy()
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "あなたは映像設計のための慎重な画像観察者です。"
                            "画像から直接確認できることだけを簡潔に答えてください。"
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "この参照フレームについて、構図、顔と視線の向き、"
                            "照明、服装の色や形、背景、鮮明さ、遮蔽だけを"
                            "日本語80〜160文字で説明してください。"
                            "氏名、年齢、人種、性別、健康、性格、魅力、職業などを"
                            "推測せず、見出しや箇条書きも付けないでください。"
                        ),
                    },
                ],
            },
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        return self._decode_generation(inputs, max_new_tokens)

    def _decode_generation(
        self,
        inputs: Any,
        max_new_tokens: int,
    ) -> str:
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._torch is not None
        text_tokenizer = getattr(self._tokenizer, "tokenizer", self._tokenizer)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.03,
                pad_token_id=text_tokenizer.pad_token_id,
                eos_token_id=text_tokenizer.eos_token_id,
            )
        return text_tokenizer.decode(
            generated[0, prompt_tokens:],
            skip_special_tokens=True,
        ).strip()

    @staticmethod
    def _clean_answer(raw_text: str) -> str:
        text = raw_text.strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if text.startswith("```"):
            first_newline = text.find("\n")
            text = text[first_newline + 1 :] if first_newline >= 0 else text
            if text.endswith("```"):
                text = text[:-3]
        if text.startswith("{") and text.endswith("}"):
            try:
                value = json.loads(text)
                strings = [item for item in value.values() if isinstance(item, str)]
                if len(strings) == 1:
                    text = strings[0]
            except (json.JSONDecodeError, AttributeError):
                pass
        text = re.sub(
            r"^(?:回答|答え|出力|結果|Answer|Response)\s*[:：]\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"^[-*・]\s*", "", text)
        text = re.sub(r"\s*\n+\s*", " ", text)
        return text.strip().strip('"').strip("'").strip()[:2400]

    @classmethod
    def _normalize_narration(cls, value: str) -> str:
        text = cls._clean_answer(value)
        if not text:
            text = (
                "未来から連絡しています。今は仲間と新しいことを試しながら、"
                "忙しいけれど元気に過ごしています。"
            )
        supplements = [
            "こっちは思っていたより忙しいけど、毎日けっこう楽しくやっています。",
            "今話してくれたことも、こちらで意外な形で役に立っています。",
            "詳しいことはまだ内緒だけど、今のところ元気に過ごしています。",
        ]
        for sentence in supplements:
            if len(text) >= 80:
                break
            if text and text[-1] not in "。！？!?":
                text += "。"
            text += sentence
        if len(text) > 110:
            candidates = [
                index + 1
                for index, char in enumerate(text[:110])
                if char in "。！？!?" and index + 1 >= 80
            ]
            if candidates:
                text = text[: candidates[-1]]
            else:
                text = text[:109].rstrip("、。！？!? ") + "。"
        if len(text) < 80:
            text += "また続きが話せるようになったら、こちらの近況も伝えます。"
        if len(text) > 110:
            text = text[:109].rstrip("、。！？!? ") + "。"
        return text

    def _check_cancelled(self, request_id: str) -> None:
        if request_id in self._cancelled:
            self._cancelled.discard(request_id)
            raise asyncio.CancelledError

    @staticmethod
    def _read_json(path_value: str | None) -> Any:
        if not path_value:
            raise ValueError("generation_llm_input_path_required")
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
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
        self._multimodal = False
        self._model_spec = None
        self._cancelled.clear()
        await asyncio.to_thread(gc.collect)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def create_worker(role: WorkerRole) -> WorkerAdapter:
    if role == WorkerRole.INTERVIEW_LLM:
        from .transformers_interview_llm import TransformersInterviewLlmAdapter

        return TransformersInterviewLlmAdapter(role)
    return TransformersGenerationLlmAdapter(role)
