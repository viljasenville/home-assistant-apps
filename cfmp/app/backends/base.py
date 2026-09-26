"""Backend interface.

A backend is one model implementation. Adding XGBoost or a scikit-learn
model means adding a module here and registering it; the HTTP contract does
not change, because callers discover backends at runtime through
``GET /models``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class InsufficientData(Exception):
    """Not enough usable history to train. Surfaces as 422 insufficient_data."""

    def __init__(self, n_hours: int, required: int) -> None:
        super().__init__(f"{n_hours} usable hours, {required} required")
        self.n_hours = n_hours
        self.required = required


@dataclass
class FitResult:
    state: object
    val_mae: float | None
    n_hours: int
    extra: dict = field(default_factory=dict)


class ForecastBackend(ABC):
    """One model implementation.

    Class attributes describe the backend to callers; the instance methods
    do the work. State is whatever the backend needs to persist, and the
    backend itself owns its serialisation format.
    """

    id: str
    name: str
    description: str
    min_hours: int
    autoregressive: bool
    lag_hours: int = 0

    @classmethod
    def available(cls) -> bool:
        """Whether this backend's library is installed in this deployment."""
        return True

    @classmethod
    def info(cls) -> dict:
        return {
            "id": cls.id,
            "name": cls.name,
            "description": cls.description,
            "available": cls.available(),
            "min_hours": cls.min_hours,
            "autoregressive": cls.autoregressive,
            "lag_hours": cls.lag_hours,
        }

    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        *,
        timezone: str,
        base_temp: float,
        num_threads: int,
    ) -> FitResult: ...

    @abstractmethod
    def predict(
        self,
        state: object,
        df: pd.DataFrame,
        *,
        timezone: str,
        base_temp: float,
        future_index: pd.DatetimeIndex,
        origin: pd.Timestamp,
    ) -> np.ndarray: ...

    @abstractmethod
    def save(self, state: object, directory: str) -> None: ...

    @abstractmethod
    def load(self, directory: str) -> object: ...

    def feature_importance(self, state: object) -> dict[str, float] | None:
        return None
