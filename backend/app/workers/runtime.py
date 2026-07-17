from __future__ import annotations

import argparse
import asyncio
import logging
import os

import uvicorn
from fastapi import FastAPI, HTTPException, Request

from ..schemas import (
    WorkerHealth,
    WorkerModelSpec,
    WorkerProgressEvent,
    WorkerRequest,
    WorkerRole,
)
from .base import WorkerAdapter
from .factory import create_worker_adapter


LOGGER = logging.getLogger(__name__)


def _error_detail(exc: Exception, phase: str) -> dict[str, str]:
    message = str(exc).strip() or type(exc).__name__
    return {
        "code": str(getattr(exc, "code", "worker_runtime_error")),
        "phase": str(getattr(exc, "phase", phase)),
        "message": message[:1200],
        "exception_type": type(exc).__name__,
    }


def create_runtime_app(role: WorkerRole, expected_key: str, auth_header: str) -> FastAPI:
    app = FastAPI(title=f"Moshimo Worker: {role.value}", docs_url=None, redoc_url=None)
    adapter: WorkerAdapter | None = None
    progress_events: dict[str, list[dict[str, object]]] = {}

    def authorize(request: Request) -> None:
        if request.headers.get(auth_header) != expected_key:
            raise HTTPException(status_code=401, detail="worker_auth_failed")

    @app.get("/health")
    async def health(request: Request):
        authorize(request)
        if adapter is None:
            return WorkerHealth(worker=role, loaded=False, ready=False).model_dump(
                mode="json"
            )
        return (await adapter.healthcheck()).model_dump(mode="json")

    @app.post("/load")
    async def load(request: Request, model: WorkerModelSpec):
        nonlocal adapter
        authorize(request)
        if model.worker != role:
            raise HTTPException(status_code=422, detail="worker_role_mismatch")
        if adapter is not None:
            await adapter.unload()
        adapter = create_worker_adapter(role, model.adapter_entrypoint)
        try:
            return (await adapter.load(model)).model_dump(mode="json")
        except Exception as exc:
            LOGGER.exception("worker load failed: role=%s", role.value)
            raise HTTPException(
                status_code=500,
                detail=_error_detail(exc, "model_load"),
            ) from exc

    @app.get("/progress/{request_id}")
    async def progress(request: Request, request_id: str):
        authorize(request)
        return {"events": progress_events.get(request_id, [])}

    @app.post("/run")
    async def run(request: Request, payload: WorkerRequest):
        authorize(request)
        if adapter is None:
            raise HTTPException(status_code=409, detail="worker_not_loaded")
        progress_events[payload.request_id] = []

        async def record_progress(event: WorkerProgressEvent) -> None:
            progress_events[payload.request_id].append(event.model_dump(mode="json"))

        try:
            result = await adapter.run(payload, record_progress)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOGGER.exception("worker run failed: role=%s", role.value)
            raise HTTPException(
                status_code=500,
                detail=_error_detail(exc, "inference"),
            ) from exc
        return result.model_dump(mode="json")

    @app.post("/cancel/{request_id}")
    async def cancel(request: Request, request_id: str):
        authorize(request)
        if adapter is not None:
            await adapter.cancel(request_id)
        return {"ok": True}

    @app.post("/unload")
    async def unload(request: Request):
        nonlocal adapter
        authorize(request)
        if adapter is not None:
            await adapter.unload()
            adapter = None
        return {"ok": True}

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--role", required=True, choices=[item.value for item in WorkerRole])
    args = parser.parse_args()

    expected_key = os.environ["MOSHIMO_WORKER_KEY"]
    auth_header = os.environ.get("MOSHIMO_WORKER_AUTH_HEADER", "X-Moshimo-Worker-Key")
    uvicorn.run(
        create_runtime_app(WorkerRole(args.role), expected_key, auth_header),
        host=args.host,
        port=args.port,
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
