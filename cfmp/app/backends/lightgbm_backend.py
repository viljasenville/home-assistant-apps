"""LightGBM backend: gradient-boosted trees over the shared feature pipeline."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .. import features as F
from .base import FitResult, ForecastBackend, InsufficientData

# L1 is more robust to consumption spikes than squared error.
DEFAULT_PARAMS: dict = {
    "objective": "regression_l1",
    "metric": "l1",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
}

MAX_ROUNDS = 1500
EARLY_STOPPING_ROUNDS = 75
VALIDATION_DAYS = 14
MIN_HORIZON = 1
MAX_HORIZON = 48


class LightGBMState:
    def __init__(self, booster, best_iteration: int, params: dict) -> None:
        self.booster = booster
        self.best_iteration = best_iteration
        self.params = params


class LightGBMBackend(ForecastBackend):
    id = "lightgbm"
    name = "LightGBM"
    description = (
        "Gradient-boosted trees at hour level. Calendar, weather and "
        "origin-anchored history features, trained across a sampled horizon "
        "range so accuracy holds out to 48 hours. Direct multi-step: every "
        "hour is predicted from the same forecast origin rather than fed back "
        "into the model, so errors do not compound along the horizon."
    )
    min_hours = 504
    # A history_tail is required for the lag features, hence true — though
    # forecasting is direct rather than recursive.
    autoregressive = True
    lag_hours = F.LAG_HOURS

    @classmethod
    def available(cls) -> bool:
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            return False
        return True

    def fit(
        self,
        df: pd.DataFrame,
        *,
        timezone: str,
        base_temp: float,
        num_threads: int,
    ) -> FitResult:
        import lightgbm as lgb

        labelled = df.index[df["energy"].notna()]
        if len(labelled) < self.min_hours:
            raise InsufficientData(len(labelled), self.min_hours)

        # A random split would leak the future into the past through the lags.
        split_at = labelled[-1] - pd.Timedelta(days=VALIDATION_DAYS)
        train_idx = labelled[labelled <= split_at]
        valid_idx = labelled[labelled > split_at]
        if len(train_idx) < 24 * 14 or len(valid_idx) < 24:
            train_idx, valid_idx = labelled, labelled[:0]

        params = {**DEFAULT_PARAMS, "num_threads": num_threads, "seed": 0}

        t_targets, t_origins = F.sample_training_pairs(
            train_idx, min_horizon=MIN_HORIZON, max_horizon=MAX_HORIZON, seed=0
        )
        x_train = F.build_matrix(
            df, timezone=timezone, base_temp=base_temp, targets=t_targets, origins=t_origins
        )
        y_train = df["energy"].reindex(t_targets).to_numpy()
        train_set = lgb.Dataset(x_train, label=y_train, free_raw_data=False)

        callbacks = [lgb.log_evaluation(period=0)]
        valid_sets: list = []
        x_valid = y_valid = v_targets = None

        if len(valid_idx) >= 24:
            v_targets, v_origins = F.sample_training_pairs(
                valid_idx, min_horizon=MIN_HORIZON, max_horizon=MAX_HORIZON, seed=1
            )
            x_valid = F.build_matrix(
                df, timezone=timezone, base_temp=base_temp, targets=v_targets, origins=v_origins
            )
            y_valid = df["energy"].reindex(v_targets).to_numpy()
            valid_sets = [lgb.Dataset(x_valid, label=y_valid, reference=train_set)]
            callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False))

        booster = lgb.train(
            params,
            train_set,
            num_boost_round=MAX_ROUNDS,
            valid_sets=valid_sets,
            valid_names=["valid"] if valid_sets else None,
            callbacks=callbacks,
        )
        best_iteration = int(booster.best_iteration or booster.num_trees())

        val_mae = None
        extra: dict = {"num_trees": int(booster.num_trees()), "max_horizon": MAX_HORIZON}
        if valid_sets:
            pred = np.asarray(booster.predict(x_valid, num_iteration=best_iteration))
            val_mae = F.mae(y_valid, pred)
            # The hour-of-week baseline on the same holdout, so the caller can
            # see whether the trees earn their keep.
            how_train = F.hour_of_week(train_idx, timezone)
            profile = (
                pd.Series(df["energy"].reindex(train_idx).to_numpy(), index=how_train)
                .groupby(level=0)
                .mean()
            )
            overall = float(np.nanmean(df["energy"].reindex(train_idx).to_numpy()))
            baseline = profile.reindex(F.hour_of_week(v_targets, timezone)).fillna(overall)
            extra["baseline_val_mae"] = F.mae(y_valid, baseline.to_numpy())

        return FitResult(
            state=LightGBMState(booster, best_iteration, params),
            val_mae=val_mae,
            n_hours=len(labelled),
            extra=extra,
        )

    def predict(
        self,
        state: LightGBMState,
        df: pd.DataFrame,
        *,
        timezone: str,
        base_temp: float,
        future_index: pd.DatetimeIndex,
        origin: pd.Timestamp,
    ) -> np.ndarray:
        targets, origins = F.prediction_pairs(future_index, origin)
        x = F.build_matrix(
            df, timezone=timezone, base_temp=base_temp, targets=targets, origins=origins
        )
        out = np.asarray(
            state.booster.predict(x, num_iteration=state.best_iteration), dtype=float
        )
        return np.maximum(out, 0.0)

    def save(self, state: LightGBMState, directory: str) -> None:
        state.booster.save_model(
            os.path.join(directory, "model.txt"), num_iteration=state.best_iteration
        )
        with open(os.path.join(directory, "backend.json"), "w", encoding="utf-8") as fh:
            json.dump({"best_iteration": state.best_iteration, "params": state.params}, fh)

    def load(self, directory: str) -> LightGBMState:
        import lightgbm as lgb

        booster = lgb.Booster(model_file=os.path.join(directory, "model.txt"))
        with open(os.path.join(directory, "backend.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        return LightGBMState(booster, int(meta["best_iteration"]), meta.get("params", {}))

    def feature_importance(self, state: LightGBMState) -> dict[str, float] | None:
        gains = state.booster.feature_importance(importance_type="gain")
        return {name: float(v) for name, v in zip(F.FEATURE_COLUMNS, gains)}
