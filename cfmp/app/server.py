"""Entry point: ``python -m app.server``."""

from __future__ import annotations

import logging

import uvicorn

from .config import LOG_LEVEL_MAP, load_settings


def main() -> None:
    settings = load_settings()
    level = LOG_LEVEL_MAP.get(settings.log_level, "INFO")
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # One worker: the model cache lives in-process. Training and prediction
    # are sync routes, so FastAPI runs them in the threadpool and the event
    # loop stays responsive.
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",  # noqa: S104 — Supervisor's internal network
        port=settings.port,
        log_level=level.lower(),
        access_log=settings.log_level in ("trace", "debug"),
        workers=1,
    )


if __name__ == "__main__":
    main()
