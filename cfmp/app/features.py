"""Feature pipeline shared by the tree-based backends.

Two design rules govern this module.

First, training and prediction must see identical features, so both go
through ``build_matrix``.

Second, no feature may use information that is unavailable at forecast time.
A one-hour energy lag is not known when forecasting twelve hours ahead, so
energy-derived features are limited to long lags (``energy_lag168``, the same
hour a week earlier) and *origin-anchored* aggregates computed at the moment
the forecast is issued.

During training a horizon is drawn at random for every row and the origin
follows from it. This matches the training distribution to the prediction
situation: without it the model learns to lean on recent observations and
degrades sharply at longer horizons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COOLING_BASE_C = 21.0

# Hours of history a prediction call must supply for these features to be
# fully populated. Reported to callers as the model's lag depth.
LAG_HOURS = 168

FEATURE_COLUMNS: list[str] = [
    "hour",
    "dow",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "out_temp",
    "out_temp_lag1",
    "out_temp_lag3",
    "out_temp_mean24",
    "out_temp_min24",
    "out_temp_max24",
    "hdd",
    "cdd",
    "target",
    "horizon",
    "energy_lag168",
    "origin_last",
    "origin_mean24",
    "origin_mean168",
]


def _row_features(df: pd.DataFrame, timezone: str, base_temp: float) -> pd.DataFrame:
    """Features that depend only on the hour being forecast."""
    local = df.index.tz_convert(timezone)
    hour = local.hour.to_numpy(dtype=float)
    dow = local.dayofweek.to_numpy(dtype=float)
    month = local.month.to_numpy(dtype=float)

    out = pd.DataFrame(index=df.index)
    out["hour"] = hour
    out["dow"] = dow
    out["is_weekend"] = (dow >= 5).astype(float)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)

    temp = df["out_temp"]
    out["out_temp"] = temp
    out["out_temp_lag1"] = temp.shift(1)
    out["out_temp_lag3"] = temp.shift(3)
    # Thermal inertia: yesterday's weather still drives today's consumption.
    out["out_temp_mean24"] = temp.rolling(24, min_periods=3).mean()
    out["out_temp_min24"] = temp.rolling(24, min_periods=3).min()
    out["out_temp_max24"] = temp.rolling(24, min_periods=3).max()

    # Per-hour target when supplied, otherwise the request's base_temp.
    threshold = df["target"].fillna(base_temp)
    out["hdd"] = np.maximum(0.0, threshold - temp)
    out["cdd"] = np.maximum(0.0, temp - COOLING_BASE_C)
    out["target"] = df["target"]

    out["energy_lag168"] = df["energy"].shift(LAG_HOURS)
    return out


def _origin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates as of the moment a forecast is issued.

    Indexed by origin: row ``o`` describes the state where the most recent
    observation is ``o - 1 h``.
    """
    prev = df["energy"].shift(1)
    out = pd.DataFrame(index=df.index)
    out["origin_last"] = prev
    out["origin_mean24"] = prev.rolling(24, min_periods=6).mean()
    out["origin_mean168"] = prev.rolling(168, min_periods=24).mean()
    return out


def build_matrix(
    df: pd.DataFrame,
    *,
    timezone: str,
    base_temp: float,
    targets: pd.DatetimeIndex,
    origins: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Assemble the feature matrix for (target, origin) pairs.

    Element *i* means "forecast targets[i] given that the latest observation
    is origins[i] - 1 h".
    """
    if len(targets) != len(origins):
        raise ValueError("targets and origins have different lengths.")

    rows = _row_features(df, timezone, base_temp).reindex(targets)
    anchors = _origin_features(df).reindex(origins)

    horizon = (targets - origins).total_seconds() / 3600.0 + 1.0
    rows["horizon"] = np.asarray(horizon, dtype=float)
    for col in ("origin_last", "origin_mean24", "origin_mean168"):
        rows[col] = anchors[col].to_numpy()

    return rows[FEATURE_COLUMNS].astype(float)


def sample_training_pairs(
    index: pd.DatetimeIndex,
    *,
    min_horizon: int,
    max_horizon: int,
    seed: int = 0,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Draw a horizon per training row and derive its origin."""
    rng = np.random.default_rng(seed)
    horizons = rng.integers(min_horizon, max_horizon + 1, size=len(index))
    return index, pd.DatetimeIndex(index - pd.to_timedelta(horizons - 1, unit="h"))


def prediction_pairs(
    future_index: pd.DatetimeIndex, origin: pd.Timestamp
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """At prediction time every hour shares one origin."""
    return future_index, pd.DatetimeIndex([origin] * len(future_index))


def hour_of_week(index: pd.DatetimeIndex, timezone: str) -> np.ndarray:
    local = index.tz_convert(timezone)
    return (local.dayofweek * 24 + local.hour).to_numpy()


def mae(actual: np.ndarray, pred: np.ndarray) -> float | None:
    mask = np.isfinite(actual) & np.isfinite(pred)
    if not mask.any():
        return None
    return float(np.mean(np.abs(pred[mask] - actual[mask])))
