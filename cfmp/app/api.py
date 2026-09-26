"""FastAPI application and routes. See docs/openapi.yaml for the contract."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

from . import __version__, backends, errors, timeseries as TS
from .config import Settings, load_settings
from .schemas import (
    Error,
    Health,
    HourForecast,
    InstanceInfo,
    ModelInfo,
    ModelList,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)
from .store import InstanceStore

_LOGGER = logging.getLogger("consumption_forecast")

settings: Settings = load_settings()
store = InstanceStore(settings.instances_dir, keep_versions=settings.keep_model_versions)

ERRORS = {
    400: {"model": Error},
    401: {"model": Error},
    404: {"model": Error},
    409: {"model": Error},
}

app = FastAPI(
    title="Consumption Forecast Model Service",
    version=__version__,
    description=(
        "HTTP API for an out-of-Home-Assistant model service. The integration "
        "owns the data; this service only trains models and produces forecasts."
    ),
)
errors.install(app)


def require_token(request: Request) -> None:
    if not settings.api_token:
        return
    header = request.headers.get("authorization", "")
    supplied = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if supplied != settings.api_token:
        raise errors.unauthorized()


AUTH = [Depends(require_token)]


def _default_model() -> str:
    return backends.resolve_default(settings.default_model)


def _resolve_backend(model: str | None):
    """Pick the backend for a request, or fail the way the contract says."""
    chosen = model or _default_model()
    backend = backends.get(chosen)
    if backend is None or not backend.available():
        raise errors.model_not_available(chosen)
    return backend


def _now_iso() -> str:
    return datetime.now(ZoneInfo(settings.timezone)).isoformat(timespec="seconds")


@app.get("/health", response_model=Health, response_model_exclude_none=True, tags=["meta"])
def get_health() -> Health:
    """Liveness probe. Unauthenticated so probes stay simple."""
    return Health(version=__version__, default_model=_default_model())


@app.get("/models", response_model=ModelList, response_model_exclude_none=True,
         dependencies=AUTH, tags=["models"], responses={401: {"model": Error}})
def list_models() -> ModelList:
    return ModelList(
        default=_default_model(),
        models=[ModelInfo(**b.info()) for b in backends.all_backends()],
    )


@app.get("/models/{model}", response_model=ModelInfo, response_model_exclude_none=True,
         dependencies=AUTH, tags=["models"], responses=ERRORS)
def get_model(model: str) -> ModelInfo:
    backend = backends.get(model)
    if backend is None:
        raise errors.not_found(f"Model '{model}' is not known to this service.")
    return ModelInfo(**backend.info())


@app.post("/train", response_model=TrainResponse, response_model_exclude_none=True,
          dependencies=AUTH, tags=["training"], responses=ERRORS)
def train_model(req: TrainRequest) -> TrainResponse | JSONResponse:
    store.validate_id(req.model_id)
    backend = _resolve_backend(req.model)
    base_temp = req.base_temp if req.base_temp is not None else settings.base_temp

    df = TS.to_frame(
        [h.model_dump() for h in req.series], max_rows=settings.max_history_rows
    )
    usable = int(df["energy"].notna().sum())
    _LOGGER.info(
        "Training %s/%s: %d usable hours (%s … %s)",
        backend.id, req.model_id, usable, df.index[0], df.index[-1],
    )

    lock = store.train_lock(backend.id, req.model_id)
    if not lock.acquire(blocking=False):
        raise errors.conflict(
            "training_in_progress",
            f"Training for '{req.model_id}' on '{backend.id}' is already running.",
        )
    started = time.perf_counter()
    try:
        try:
            fit = backend.fit(
                df,
                timezone=settings.timezone,
                base_temp=base_temp,
                num_threads=settings.num_threads,
            )
        except backends.InsufficientData as err:
            # A valid request with a negative answer, not a client error.
            _LOGGER.info(
                "Not enough data for %s/%s: %d < %d",
                backend.id, req.model_id, err.n_hours, err.required,
            )
            body = TrainResponse(
                trained=False,
                model=backend.id,
                model_id=req.model_id,
                reason="insufficient_data",
                n_hours=err.n_hours,
            )
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=body.model_dump(exclude_none=True),
            )

        trained_at = _now_iso()
        meta = {
            "model": backend.id,
            "model_id": req.model_id,
            "val_mae": fit.val_mae,
            "n_hours": fit.n_hours,
            "trained_at": trained_at,
            "base_temp": base_temp,
            "timezone": settings.timezone,
            **fit.extra,
        }
        importance = backend.feature_importance(fit.state)
        if importance:
            meta["feature_importance"] = importance
        store.save(backend, req.model_id, fit.state, meta, settings.timezone)
    finally:
        lock.release()

    _LOGGER.info(
        "Trained %s/%s in %.1f s, val_mae %s",
        backend.id, req.model_id, time.perf_counter() - started, fit.val_mae,
    )
    return TrainResponse(
        trained=True,
        model=backend.id,
        model_id=req.model_id,
        val_mae=fit.val_mae,
        n_hours=fit.n_hours,
        trained_at=trained_at,
        baseline_val_mae=fit.extra.get("baseline_val_mae"),
    )


@app.post("/predict", response_model=PredictResponse, response_model_exclude_none=True,
          dependencies=AUTH, tags=["forecast"], responses=ERRORS)
def predict(req: PredictRequest) -> PredictResponse:
    store.validate_id(req.model_id)

    # Without an explicit model, use whichever backend holds this instance.
    model = req.model or store.find_model_for(req.model_id)
    if model is None:
        raise errors.not_found(
            f"No trained model found for model_id '{req.model_id}'. Train first."
        )
    backend = _resolve_backend(model)

    try:
        state, meta = store.load(backend, req.model_id)
    except FileNotFoundError as err:
        raise errors.not_found(str(err)) from err

    timezone = meta.get("timezone", settings.timezone)
    base_temp = meta.get("base_temp", settings.base_temp)

    future_records = [h.model_dump() for h in req.future]
    offset = TS.offset_of(future_records)
    future = TS.to_frame(future_records)
    origin = future.index[0]

    tail_records = [h.model_dump() for h in (req.history_tail or [])]
    if backend.autoregressive:
        if not tail_records:
            raise errors.conflict(
                "history_tail_required",
                f"Model '{backend.id}' needs a history_tail of at least "
                f"{backend.lag_hours} hours for its lag features.",
            )
        history = TS.to_frame(tail_records)
        history = history.loc[: origin - pd.Timedelta(hours=1)]
        if len(history) < backend.lag_hours:
            raise errors.conflict(
                "history_tail_too_short",
                f"history_tail covers {len(history)} hours before the forecast "
                f"origin; model '{backend.id}' needs {backend.lag_hours}.",
            )
    elif tail_records:
        history = TS.to_frame(tail_records).loc[: origin - pd.Timedelta(hours=1)]
    else:
        history = future.iloc[:0]

    combined = TS.merge_for_prediction(history, future) if len(history) else future
    values = backend.predict(
        state,
        combined,
        timezone=timezone,
        base_temp=base_temp,
        future_index=future.index,
        origin=origin,
    )

    return PredictResponse(
        model=backend.id,
        hourly=[
            HourForecast(ts=TS.isoformat(ts, offset), kwh=round(float(v), 4))
            for ts, v in zip(future.index, values)
        ],
    )


@app.get(
    "/models/{model}/instances/{model_id}",
    response_model=InstanceInfo,
    response_model_exclude_none=True,
    dependencies=AUTH,
    tags=["training"],
    responses=ERRORS,
)
def get_instance(model: str, model_id: str) -> InstanceInfo:
    store.validate_id(model_id)
    if not backends.exists(model):
        raise errors.not_found(f"Model '{model}' is not known to this service.")
    meta = store.meta(model, model_id)
    if meta is None:
        raise errors.not_found(
            f"No trained instance '{model_id}' for model '{model}'."
        )
    return InstanceInfo(
        model=model,
        model_id=model_id,
        val_mae=meta.get("val_mae"),
        n_hours=meta.get("n_hours"),
        trained_at=meta.get("trained_at"),
        feature_importance=meta.get("feature_importance"),
        baseline_val_mae=meta.get("baseline_val_mae"),
    )


@app.delete(
    "/models/{model}/instances/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=AUTH,
    tags=["training"],
    responses={401: {"model": Error}},
)
def delete_instance(model: str, model_id: str) -> Response:
    """Idempotent: 204 whether or not the instance existed."""
    store.validate_id(model_id)
    store.delete(model, model_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
