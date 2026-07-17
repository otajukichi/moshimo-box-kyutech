from __future__ import annotations

from importlib import import_module
from typing import Callable, cast

from ..schemas import WorkerRole
from .base import WorkerAdapter


WorkerFactory = Callable[[WorkerRole], WorkerAdapter]


def create_worker_adapter(role: WorkerRole, entrypoint: str) -> WorkerAdapter:
    module_name, separator, attribute_name = entrypoint.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError(f"invalid_worker_entrypoint: {entrypoint}")
    module = import_module(module_name)
    factory = cast(WorkerFactory, getattr(module, attribute_name))
    adapter = factory(role)
    if not isinstance(adapter, WorkerAdapter):
        raise TypeError(f"worker_factory_returned_invalid_adapter: {entrypoint}")
    if adapter.role != role:
        raise ValueError(f"worker_factory_role_mismatch: {entrypoint}")
    return adapter
