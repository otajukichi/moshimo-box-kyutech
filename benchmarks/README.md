# Local model benchmarks

Model selection is deferred to the next stage. Store reproducible benchmark
commands and aggregate, non-personal results here. Raw recordings and generated
media belong in `private-inputs/` or `results/`, which are ignored by Git.

Record exact model revision, command, environment, load/unload time, peak CPU and
GPU memory, storage use, processing time, failures, rerun stability, and license
review before marking a model as validated in the local catalog.

## First ASR candidate

Install the pinned faster-whisper backend and exact Kotoba model revision:

```bash
./scripts/install-asr-kotoba.sh
```

Run the local benchmark with a private Japanese recording. The result stores
timing, memory, error counts, and transcript character count, but not the
transcript itself:

```bash
<workspace>/env/moshimo-box-kyutech/app/bin/python \
  benchmarks/asr/benchmark_faster_whisper.py \
  benchmarks/private-inputs/sample.webm --runs 3
```
