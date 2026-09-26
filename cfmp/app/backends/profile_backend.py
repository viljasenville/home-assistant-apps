"""Hour-of-week profile backend.

Deliberately simple: an average for each hour of the week plus a linear
heating-degree term. It needs no third-party library, trains in
milliseconds, and works on two weeks of data, so it stays available when a
heavier backend cannot run on the hardware.

It is also the reference point for the others — `val_mae` from this backend
is what a gradient-boosted model has to beat to be worth its container.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .. import features as F
from .base import FitResult, ForecastBackend, InsufficientData

VALIDATION_DAYS = 7


class ProfileState:
    def __init__(self, profile: list[float], hdd_coef: float, overall: float) -> None:
        self.profile = profile          # 168 values, Monday 00:00 first
        self.hdd_coef = hdd_coef
        self.overall = overall


def _fit_arrays(
    energy: np.ndarray, how: np.ndarray, hdd: np.ndarray
) -> tuple[np.ndarray, float, float]:
    ok = np.isfinite(energy) & np.isfinite(hdd)
    energy, how, hdd = energy[ok], how[ok], hdd[ok]
    overall = float(np.mean(energy)) if energy.size else 0.0

    # Least squares on the heating term first, then a profile over the residual.
    if hdd.size and np.ptp(hdd) > 1e-9:
        design = np.column_stack([hdd, np.ones_like(hdd)])
        coef, _, _, _ = np.linalg.lstsq(design, energy, rcond=None)
        hdd_coef = float(coef[0])
    else:
        hdd_coef = 0.0

    residual = energy - hdd_coef * hdd
    profile = np.full(168, float(np.mean(residual)) if residual.size else overall)
    for key in range(168):
        sel = how == key
        if sel.any():
            profile[key] = float(np.mean(residual[sel]))
    return profile, hdd_coef, overall


class ProfileBackend(ForecastBackend):
    id = "profile"
    name = "Hour-of-week profile"
    description = (
        "Average consumption per hour of the week plus a linear "
        "heating-degree term. No external dependencies, trains in "
        "milliseconds, and needs only two weeks of history. Serves as the "
        "accuracy reference the other backends are compared against."
    )
    min_hours = 336
    # Purely calendar and weather driven, so no history_tail is needed.
    autoregressive = False
    lag_hours = 0

    def _hdd(self, df: pd.DataFrame, base_temp: float) -> np.ndarray:
        threshold = df["target"].fillna(base_temp)
        return np.maximum(0.0, (threshold - df["out_temp"]).to_numpy())

    def fit(
        self,
        df: pd.DataFrame,
        *,
        timezone: str,
        base_temp: float,
        num_threads: int,
    ) -> FitResult:
        labelled = df.index[df["energy"].notna()]
        if len(labelled) < self.min_hours:
            raise InsufficientData(len(labelled), self.min_hours)

        hdd_all = pd.Series(self._hdd(df, base_temp), index=df.index)
        split_at = labelled[-1] - pd.Timedelta(days=VALIDATION_DAYS)
        train_idx = labelled[labelled <= split_at]
        valid_idx = labelled[labelled > split_at]
        if len(train_idx) < 24 * 7 or len(valid_idx) < 24:
            train_idx, valid_idx = labelled, labelled[:0]

        profile, hdd_coef, overall = _fit_arrays(
            df["energy"].reindex(train_idx).to_numpy(),
            F.hour_of_week(train_idx, timezone),
            hdd_all.reindex(train_idx).to_numpy(),
        )

        val_mae = None
        if len(valid_idx) >= 24:
            pred = profile[F.hour_of_week(valid_idx, timezone)] + hdd_coef * hdd_all.reindex(
                valid_idx
            ).to_numpy()
            val_mae = F.mae(df["energy"].reindex(valid_idx).to_numpy(), np.maximum(pred, 0.0))

        # Refit on everything once validated, so the stored model uses all data.
        profile, hdd_coef, overall = _fit_arrays(
            df["energy"].reindex(labelled).to_numpy(),
            F.hour_of_week(labelled, timezone),
            hdd_all.reindex(labelled).to_numpy(),
        )
        return FitResult(
            state=ProfileState(profile.tolist(), hdd_coef, overall),
            val_mae=val_mae,
            n_hours=len(labelled),
            extra={"hdd_coef": round(hdd_coef, 5)},
        )

    def predict(
        self,
        state: ProfileState,
        df: pd.DataFrame,
        *,
        timezone: str,
        base_temp: float,
        future_index: pd.DatetimeIndex,
        origin: pd.Timestamp,
    ) -> np.ndarray:
        rows = df.reindex(future_index)
        hdd = self._hdd(rows, base_temp)
        profile = np.asarray(state.profile, dtype=float)
        pred = profile[F.hour_of_week(future_index, timezone)] + state.hdd_coef * np.nan_to_num(
            hdd
        )
        return np.maximum(pred, 0.0)

    def save(self, state: ProfileState, directory: str) -> None:
        with open(os.path.join(directory, "profile.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "profile": state.profile,
                    "hdd_coef": state.hdd_coef,
                    "overall": state.overall,
                },
                fh,
            )

    def load(self, directory: str) -> ProfileState:
        with open(os.path.join(directory, "profile.json"), encoding="utf-8") as fh:
            raw = json.load(fh)
        return ProfileState(raw["profile"], float(raw["hdd_coef"]), float(raw["overall"]))
