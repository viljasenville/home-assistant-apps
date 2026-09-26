"""Hourly series handling shared by every backend.

The API speaks in ``HourActual`` / ``HourFuture`` objects; internally
everything is a gap-free hourly DataFrame on a UTC index with the columns
``energy``, ``out_temp`` and ``target``.
"""

from __future__ import annotations

from datetime import timedelta, timezone as dt_timezone

import numpy as np
import pandas as pd

COLUMNS = ("energy", "out_temp", "target")


def to_frame(records: list[dict], *, max_rows: int | None = None) -> pd.DataFrame:
    """Build a gap-free hourly frame. Gaps become NaN rows so lags stay aligned."""
    if not records:
        raise ValueError("Empty series.")

    df = pd.DataFrame.from_records(records)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    df["ts"] = df["ts"].dt.floor("h")
    df = df.dropna(subset=["ts"]).drop_duplicates(subset=["ts"], keep="last")
    df = df.sort_values("ts").set_index("ts")

    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan

    if max_rows is not None and len(df) > max_rows:
        df = df.iloc[-max_rows:]

    full = pd.date_range(df.index[0], df.index[-1], freq="h", tz="UTC")
    return df.reindex(full)[list(COLUMNS)]


def offset_of(records: list[dict]) -> dt_timezone:
    """The UTC offset the caller used, so responses echo it back.

    The contract requires an explicit offset on every timestamp; returning
    forecasts in the caller's own offset keeps them directly comparable with
    what was sent.
    """
    for rec in records:
        ts = pd.to_datetime(rec["ts"], format="ISO8601")
        if ts.tzinfo is not None:
            return dt_timezone(timedelta(seconds=ts.utcoffset().total_seconds()))
    return dt_timezone.utc


def isoformat(ts: pd.Timestamp, offset: dt_timezone) -> str:
    return ts.tz_convert(offset).isoformat()


def merge_for_prediction(history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    """One continuous frame: energy known up to the origin, weather throughout."""
    combined = pd.concat([history, future])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    full = pd.date_range(combined.index[0], combined.index[-1], freq="h", tz="UTC")
    return combined.reindex(full)
