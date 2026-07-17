# Generation model stack

Model environments and revisions are pinned by the installer scripts and
`config/model-catalog.yaml`. Downloaded weights and third-party repositories
remain outside Git.

| Stage | Fast | Balanced | Full/quality |
| --- | --- | --- | --- |
| Interview ASR | Kotoba-Whisper v2.0 faster | same | same |
| Interview LLM | Qwen3 4B Instruct | same | GPT-OSS 20B; Qwen models remain selectable |
| Script design and review | Qwen3 4B Instruct | Qwen3-VL 8B | GPT-OSS 20B; Qwen3-VL remains selectable |
| Voice clone TTS | Fish Audio S2 Pro | same | same |
| Future image | FLUX.2 Klein 4B | FLUX.2 Klein 9B | FLUX.2 Klein 9B |
| Audio-driven video | MuseTalk 1.5 | MuseTalk 1.5 | EchoMimicV3 Flash or MuseTalk 1.5 |

Install a reproducible group instead of invoking every model installer by hand:

~~~bash
./scripts/install-models.sh fast
./scripts/install-models.sh balanced
./scripts/install-models.sh gpt-oss
./scripts/install-models.sh full
~~~

`balanced` is the normal demo target. `gpt-oss` adds only GPT-OSS 20B, while
`full` also installs EchoMimicV3. Every installer reuses existing Conda
environments and resumable model directories. When GPT-OSS is installed, the
`quality` preset selects it for interview, script design, and inline safety
review; every Qwen option remains available in detailed model settings.

GPT-OSS is text-only in this app, so script design does not pass a reference
image to it. The A100 MIG runtime uses a separate `gpt-oss` Conda environment
and vLLM 0.10.2's CUDA 12.8 MXFP4 runtime with Transformers 4.56.1, one
sequence, an 8192-token context, and a 0.78 GPU-memory limit. This avoids the roughly 48 GB BF16
Transformers path and leaves headroom inside the 40192 MiB MIG instance. The
installer rejects a non-CUDA-12 PyTorch runtime before downloading weights.
Interview and staged planning use low reasoning effort. Harmony
analysis messages are parsed but never exposed.

After any manual model change, rebuild the ignored machine-local catalog:

~~~bash
./scripts/activate-installed-models.sh
./scripts/doctor.sh
~~~

The activation step checks required files and imports, but does not load model
weights onto the GPU. The app still performs worker load and health checks at
runtime.

GPT-OSS weights use Apache-2.0 and the OpenAI gpt-oss usage policy. Fish Audio S2 Pro and gated FLUX weights have their own terms. Accept the
official terms with the lab account before download. Do not redistribute model
weights through this repository. `NOTICE` retains notices already required by
the integrated components.
