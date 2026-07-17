# Local models

Model weights, caches, and downloaded model repositories live here and are
ignored by Git. Prefer stable catalog IDs for directory names:

```text
models/
  asr/<catalog-id>/
  llm/<catalog-id>/
  tts/<catalog-id>/
  image/<catalog-id>/
  video/<catalog-id>/
  shared-cache/
```

Do not place visitor media or benchmark recordings here. Session media belongs
under `data/sessions`, and private benchmark inputs under
`benchmarks/private-inputs`; both are ignored by Git.
