"""Backend registry.

Adding a model means importing it here and appending it to ``_BACKENDS``.
Everything else — discovery, availability, the HTTP contract — follows.
"""

from __future__ import annotations

from .base import FitResult, ForecastBackend, InsufficientData
from .lightgbm_backend import LightGBMBackend
from .profile_backend import ProfileBackend

_BACKENDS: dict[str, ForecastBackend] = {
    backend.id: backend()
    for backend in (LightGBMBackend, ProfileBackend)
}


def all_backends() -> list[ForecastBackend]:
    return list(_BACKENDS.values())


def get(model: str) -> ForecastBackend | None:
    return _BACKENDS.get(model)


def exists(model: str) -> bool:
    return model in _BACKENDS


def resolve_default(preferred: str) -> str:
    """The default backend id.

    Falls back to the first available backend when the configured one cannot
    run here, so a deployment without LightGBM still answers requests that
    omit ``model``.
    """
    backend = _BACKENDS.get(preferred)
    if backend is not None and backend.available():
        return preferred
    for candidate in _BACKENDS.values():
        if candidate.available():
            return candidate.id
    return preferred


__all__ = [
    "FitResult",
    "ForecastBackend",
    "InsufficientData",
    "all_backends",
    "exists",
    "get",
    "resolve_default",
]
