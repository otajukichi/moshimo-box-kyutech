from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(level: str, log_root: Path) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        log_root / "application.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[stream_handler, file_handler],
        force=True,
    )


def session_event(
    logger: logging.Logger,
    event: str,
    *,
    session_id: str,
    **metadata: object,
) -> None:
    fields = " ".join(f"{key}={value}" for key, value in sorted(metadata.items()))
    logger.info("event=%s session_id=%s %s", event, session_id, fields)
