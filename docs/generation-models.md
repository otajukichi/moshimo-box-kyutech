# Generation model stack

Model environments and revisions are pinned by the installer scripts and
`config/model-catalog.yaml`. Downloaded weights and third-party repositories
remain outside Git.

| Stage | Fast | Balanced | Full/quality |
| --- | --- | --- | --- |
| Interview ASR | Kotoba-Whisper v2.0 faster | same | same |
| Interview LLM | Qwen3 4B Instruct | same | selectable Qwen3-VL 8B |
| Script design and review | Qwen3 4B Instruct | Qwen3-VL 8B | Qwen3-VL 8B |
| Voice clone TTS | Fish Audio S2 Pro | same | same |
| Future image | FLUX.2 Klein 4B | FLUX.2 Klein 9B | FLUX.2 Klein 9B |
| Audio-driven video | MuseTalk 1.5 | MuseTalk 1.5 | EchoMimicV3 Flash or MuseTalk 1.5 |

Install a reproducible group instead of invoking every model installer by hand:

~~~bash
./scripts/install-models.sh fast
./scripts/install-models.sh balanced
./scripts/install-models.sh full
~~~

`balanced` is the normal demo target. `full` additionally installs the slower
EchoMimicV3 option. Every installer reuses existing Conda environments and
resumable model directories.

After any manual model change, rebuild the ignored machine-local catalog:

~~~bash
./scripts/activate-installed-models.sh
./scripts/doctor.sh
~~~

The activation step checks required files and imports, but does not load model
weights onto the GPU. The app still performs worker load and health checks at
runtime.

Fish Audio S2 Pro and gated FLUX weights have their own terms. Accept the
official terms with the lab account before download. Do not redistribute model
weights through this repository. `NOTICE` retains notices already required by
the integrated components.
