"""Request and response models, mirroring docs/openapi.yaml."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Base(BaseModel):
    # The contract uses `model` and `model_id`; pydantic reserves the
    # `model_` prefix, so that protection is lifted here.
    model_config = ConfigDict(protected_namespaces=())


class HourActual(Base):
    """One historical hour, with the measured energy."""

    ts: str = Field(description="Start of the hour, ISO 8601 with offset")
    energy: float = Field(description="Consumption for the hour (kWh)")
    out_temp: float = Field(description="Outdoor temperature (°C)")
    target: float | None = Field(
        default=None,
        description="Indoor target/threshold temperature (°C); base_temp is used if absent",
    )


class HourFuture(Base):
    """One future hour to forecast."""

    ts: str
    out_temp: float = Field(description="Forecast outdoor temperature (°C)")
    target: float | None = None


class HourForecast(Base):
    ts: str
    kwh: float = Field(description="Forecast consumption for the hour (kWh)")


class ModelInfo(Base):
    id: str
    name: str
    description: str | None = None
    available: bool
    min_hours: int | None = None
    autoregressive: bool | None = None
    lag_hours: int | None = Field(
        default=None,
        description="Hours of history_tail required for lag features (0 when none)",
    )


class ModelList(Base):
    default: str
    models: list[ModelInfo]


class Health(Base):
    status: Literal["ok"] = "ok"
    version: str
    default_model: str | None = None


class TrainRequest(Base):
    model: str | None = Field(
        default=None, description="Backend id from GET /models; omit for the default"
    )
    model_id: str = Field(description="Id to store the trained model under")
    base_temp: float | None = Field(
        default=None, description="Heating threshold (°C) for hours without a target"
    )
    series: list[HourActual] = Field(min_length=24, description="Chronological hourly history")


class TrainResponse(Base):
    trained: bool
    model: str
    model_id: str | None = None
    val_mae: float | None = Field(
        default=None, description="Time-ordered validation MAE in kWh per hour"
    )
    n_hours: int | None = None
    trained_at: str | None = None
    reason: str | None = Field(default=None, description="Present when trained is false")
    baseline_val_mae: float | None = Field(
        default=None,
        description="Hour-of-week baseline MAE on the same holdout, where the backend reports it",
    )


class PredictRequest(Base):
    model: str | None = Field(
        default=None, description="Backend id; omit to use the trained instance's model"
    )
    model_id: str
    history_tail: list[HourActual] | None = Field(
        default=None,
        description="Recent actual hours for lag features, oldest first",
    )
    future: list[HourFuture] = Field(min_length=1, description="Future hours to forecast")


class PredictResponse(Base):
    model: str
    hourly: list[HourForecast]


class InstanceInfo(Base):
    model: str
    model_id: str
    val_mae: float | None = None
    n_hours: int | None = None
    trained_at: str | None = None
    feature_importance: dict[str, float] | None = None
    baseline_val_mae: float | None = None


class Error(Base):
    error: str = Field(description="Short machine-readable code")
    message: str | None = None
