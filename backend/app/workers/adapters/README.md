# Worker adapters

Real model integrations live below this package. An adapter implements
`WorkerAdapter` and exposes a factory with this signature:

```python
def create_worker(role: WorkerRole) -> WorkerAdapter:
    ...
```

Register the factory in `config/model-catalog.yaml` as
`module.path:create_worker`. Keep model-specific imports inside the factory or
adapter methods so the shared application can load the catalog without importing
heavy frameworks.

The first real adapter is `faster_whisper_asr.py`. It serves both ASR roles,
while the interview path keeps its process loaded between answer turns.
