from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor

from src.cache_utils import get_teacache_coefficients
from src.dist import set_multi_gpus_devices
from src.fm_solvers import FlowDPMSolverMultistepScheduler
from src.fm_solvers_unipc import FlowUniPCMultistepScheduler
from src.pipeline_wan_fun_inpaint_audio_2512 import WanFunInpaintAudioPipeline
from src.utils import filter_kwargs, get_image_to_video_latent3, save_videos_grid
from src.wan_image_encoder import CLIPModel
from src.wan_text_encoder import WanT5EncoderModel
from src.wan_transformer3d_audio_2512 import (
    WanTransformerAudioMask3DModel as WanTransformer,
)
from src.wan_vae import AutoencoderKLWan
from src.wav2vec2 import Wav2Vec2Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunked EchoMimicV3 Flash inference")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--transformer-path", required=True)
    parser.add_argument("--wav2vec-model-path", required=True)
    parser.add_argument("--target-seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--sample-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--chunk-frames", type=int, default=113)
    parser.add_argument("--overlap-frames", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--sampler-name", choices=["Flow", "Flow_Unipc", "Flow_DPM++"], default="Flow_Unipc")
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--audio-guidance-scale", type=float, default=2.0)
    parser.add_argument("--audio-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--teacache-threshold", type=float, default=0.1)
    parser.add_argument("--num-skip-start-steps", type=int, default=5)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--weight-dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--enable-riflex", action="store_true")
    parser.add_argument("--riflex-k", type=int, default=6)
    return parser.parse_args()


def aligned_frame_count(value: int, ratio: int) -> int:
    if value <= 1:
        return 1
    return ((value - 1) // ratio) * ratio + 1


def aligned_frame_count_ceil(value: int, ratio: int) -> int:
    if value <= 1:
        return 1
    return math.ceil((value - 1) / ratio) * ratio + 1


def sample_size_for(image: Image.Image, requested: list[int]) -> tuple[int, int]:
    width, height = image.size
    requested_area = requested[0] * requested[1]
    original_area = width * height
    if requested_area < original_area:
        scale = math.sqrt(original_area / requested_area)
        width = width / scale
        height = height / scale
    width = max(16, int(width) // 16 * 16)
    height = max(16, int(height) // 16 * 16)
    return height, width


def normalize_loudness(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio)
    if not np.isfinite(loudness) or abs(loudness) > 100:
        return audio
    return pyln.normalize.loudness(audio, loudness, -23.0)


def audio_embeddings(
    audio: np.ndarray,
    sample_rate: int,
    frame_count: int,
    wav2vec_path: str,
) -> torch.Tensor:
    encoder = Wav2Vec2Model.from_pretrained(wav2vec_path, local_files_only=True).to("cpu")
    encoder.feature_extractor._freeze_parameters()
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        wav2vec_path,
        local_files_only=True,
    )
    values = np.squeeze(extractor(audio, sampling_rate=sample_rate).input_values)
    values_tensor = torch.from_numpy(values).float().unsqueeze(0)
    with torch.no_grad():
        outputs = encoder(values_tensor, seq_len=frame_count, output_hidden_states=True)
    stacked = torch.stack(outputs.hidden_states[1:], dim=1).squeeze(0)
    stacked = rearrange(stacked, "b s d -> s b d")
    indices = (torch.arange(5) - 2).unsqueeze(0) + torch.arange(frame_count).unsqueeze(1)
    indices = torch.clamp(indices, min=0, max=stacked.shape[0] - 1)
    result = stacked[indices].unsqueeze(0).cpu()
    del encoder, extractor, outputs, stacked
    return result


def frame_to_image(frame: np.ndarray) -> Image.Image:
    pixels = np.clip(frame.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.weight_dtype == "bfloat16" else torch.float16
    device = set_multi_gpus_devices(1, 1)
    config = OmegaConf.load(args.config_path)

    transformer = WanTransformer.from_pretrained(
        os.path.join(
            args.base_model_path,
            config["transformer_additional_kwargs"].get("transformer_subpath", "transformer"),
        ),
        transformer_additional_kwargs=OmegaConf.to_container(
            config["transformer_additional_kwargs"]
        ),
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )
    state_dict = load_file(args.transformer_path)
    state_dict = state_dict.get("state_dict", state_dict)
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    print(f"Transformer loaded: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.base_model_path, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(
            args.base_model_path,
            config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
        ),
        local_files_only=True,
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(
            args.base_model_path,
            config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
        ),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    ).eval()
    clip_encoder = CLIPModel.from_pretrained(
        os.path.join(
            args.base_model_path,
            config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"),
        )
    ).to(dtype).eval()

    scheduler_class = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[args.sampler_name]
    scheduler_config = OmegaConf.to_container(config["scheduler_kwargs"])
    if args.sampler_name in {"Flow_Unipc", "Flow_DPM++"}:
        scheduler_config["shift"] = 1
    scheduler = scheduler_class(**filter_kwargs(scheduler_class, scheduler_config))
    pipeline = WanFunInpaintAudioPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip_encoder,
    )
    pipeline.to(device=device)

    coefficients = get_teacache_coefficients(args.base_model_path)
    if coefficients is not None:
        pipeline.transformer.enable_teacache(
            coefficients,
            args.num_inference_steps,
            args.teacache_threshold,
            num_skip_start_steps=args.num_skip_start_steps,
            offload=False,
        )

    audio, sample_rate = librosa.load(args.audio_path, sr=16000, mono=True)
    audio = normalize_loudness(audio, sample_rate)
    duration_seconds = len(audio) / sample_rate
    requested_frames = max(1, int(min(duration_seconds, args.target_seconds) * args.fps))
    temporal_ratio = int(vae.config.temporal_compression_ratio)
    total_frames = aligned_frame_count(requested_frames, temporal_ratio)
    chunk_frames = aligned_frame_count(args.chunk_frames, temporal_ratio)
    if args.overlap_frames <= 0 or args.overlap_frames >= chunk_frames:
        raise ValueError("overlap_frames must be positive and smaller than chunk_frames")
    embeddings = audio_embeddings(audio, sample_rate, total_frames, args.wav2vec_model_path)

    reference: Image.Image | list[Image.Image] = Image.open(args.image_path).convert("RGB")
    height, width = sample_size_for(reference, args.sample_size)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    combined: np.ndarray | None = None
    start_frame = 0
    chunk_index = 0

    with torch.no_grad():
        while start_frame < total_frames:
            remaining = total_frames - start_frame
            current_frames = (
                chunk_frames
                if remaining >= chunk_frames
                else aligned_frame_count_ceil(remaining, temporal_ratio)
            )
            if current_frames <= args.overlap_frames and combined is not None:
                break
            if args.enable_riflex:
                latent_frames = (current_frames - 1) // temporal_ratio + 1
                pipeline.transformer.enable_riflex(k=args.riflex_k, L_test=latent_frames)

            input_video, input_mask, clip_image = get_image_to_video_latent3(
                reference,
                None,
                video_length=current_frames,
                sample_size=[height, width],
            )
            chunk_audio = embeddings[:, start_frame : start_frame + current_frames]
            if chunk_audio.shape[1] < current_frames:
                padding = chunk_audio[:, -1:].repeat(
                    1,
                    current_frames - chunk_audio.shape[1],
                    1,
                    1,
                    1,
                )
                chunk_audio = torch.cat([chunk_audio, padding], dim=1)
            chunk_audio = chunk_audio.to(device=device, dtype=dtype)
            sample = pipeline(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_frames=current_frames,
                audio_embeds=chunk_audio,
                audio_scale=args.audio_scale,
                height=height,
                width=width,
                generator=generator,
                neg_scale=1.0,
                neg_steps=0,
                use_dynamic_cfg=False,
                use_dynamic_acfg=False,
                guidance_scale=args.guidance_scale,
                audio_guidance_scale=args.audio_guidance_scale,
                num_inference_steps=args.num_inference_steps,
                video=input_video,
                mask_video=input_mask,
                clip_image=clip_image,
                cfg_skip_ratio=0.0,
                shift=args.shift,
            ).videos
            sample_np = sample.detach().cpu().numpy() if torch.is_tensor(sample) else np.asarray(sample)

            if combined is None:
                combined = sample_np
            else:
                overlap = min(args.overlap_frames, combined.shape[2], sample_np.shape[2])
                mix = np.linspace(0.0, 1.0, overlap, dtype=np.float32).reshape(1, 1, -1, 1, 1)
                combined[:, :, -overlap:] = (
                    combined[:, :, -overlap:] * (1.0 - mix)
                    + sample_np[:, :, :overlap] * mix
                )
                combined = np.concatenate([combined, sample_np[:, :, overlap:]], axis=2)

            overlap = min(args.overlap_frames, sample_np.shape[2])
            reference = [
                frame_to_image(sample_np[0, :, index])
                for index in range(sample_np.shape[2] - overlap, sample_np.shape[2])
            ]
            chunk_index += 1
            print(
                f"Chunk {chunk_index}: start={start_frame} frames={current_frames} total={total_frames}",
                flush=True,
            )
            if start_frame + current_frames >= total_frames:
                break
            start_frame += current_frames - args.overlap_frames
            del sample, sample_np, chunk_audio, input_video, input_mask
            torch.cuda.empty_cache()

    if combined is None:
        raise RuntimeError("EchoMimicV3 produced no frames")
    combined = combined[:, :, :total_frames]
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    save_videos_grid(torch.from_numpy(combined), str(output_path), fps=args.fps)
    print(f"Saved {combined.shape[2]} frames to {output_path}", flush=True)


if __name__ == "__main__":
    main()
