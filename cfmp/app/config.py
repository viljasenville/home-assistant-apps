"""Settings, populated from the environment by the s6 run script."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    data_dir: str
    port: int
    api_token: str
    num_threads: int
    max_history_rows: int
    keep_model_versions: int
    log_level: str
    timezone: str
    default_model: str
    base_temp: float

    @property
    def instances_dir(self) -> str:
        return os.path.join(self.data_dir, "instances")


def load_settings() -> Settings:
    return Settings(
        data_dir=os.environ.get("CFM_DATA_DIR", "/data"),
        port=_int("CFM_PORT", 8099),
        api_token=os.environ.get("CFM_API_TOKEN", "").strip(),
        num_threads=_int("CFM_NUM_THREADS", 2),
        max_history_rows=_int("CFM_MAX_HISTORY_ROWS", 70000),
        keep_model_versions=_int("CFM_KEEP_MODEL_VERSIONS", 3),
        log_level=os.environ.get("CFM_LOG_LEVEL", "info").lower(),
        timezone=os.environ.get("CFM_TIMEZONE", "UTC"),
        default_model=os.environ.get("CFM_DEFAULT_MODEL", "lightgbm").strip() or "lightgbm",
        base_temp=_float("CFM_BASE_TEMP", 17.0),
    )


# Home Assistant log levels do not all map onto Python ones.
LOG_LEVEL_MAP = {
    "trace": "DEBUG",
    "debug": "DEBUG",
    "info": "INFO",
    "notice": "INFO",
    "warning": "WARNING",
    "error": "ERROR",
    "fatal": "CRITICAL",
}
