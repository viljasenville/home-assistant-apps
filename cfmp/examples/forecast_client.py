"""Example: calling the service from a Home Assistant integration.

This file belongs to the *integration* (custom_components/...), not to the
add-on. Copy it into your integration and adapt the names.

Two things are worth copying along with the code. First, the model list is
discovered at runtime, so the integration never hard-codes backend names and
picks up new ones without a release. Second, the service is an optional
accelerator: every call is bounded by a timeout and every failure falls back
to the built-in profile model.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

_LOGGER = logging.getLogger(__name__)

# Inside the Supervisor network the add-on resolves by its slug.
DEFAULT_URL = "http://local-consumption_forecast:8099"

PREDICT_TIMEOUT = 20
TRAIN_TIMEOUT = 600


class ForecastServiceError(Exception):
    """The service was unreachable or returned an error."""

    def __init__(self, message: str, *, code: str | None = None, status: int | None = None):
        super().__init__(message)
        self.code = code
        self.status = status


class InsufficientData(ForecastServiceError):
    """The service had too little history to train (HTTP 422)."""


@dataclass(slots=True)
class HourActual:
    ts: datetime
    energy: float
    out_temp: float
    target: float | None = None

    def as_json(self) -> dict:
        row = {
            "ts": self.ts.isoformat(),
            "energy": self.energy,
            "out_temp": self.out_temp,
        }
        if self.target is not None:
            row["target"] = self.target
        return row


@dataclass(slots=True)
class HourFuture:
    ts: datetime
    out_temp: float
    target: float | None = None

    def as_json(self) -> dict:
        row = {"ts": self.ts.isoformat(), "out_temp": self.out_temp}
        if self.target is not None:
            row["target"] = self.target
        return row


class ForecastServiceClient:
    def __init__(
        self,
        hass: HomeAssistant,
        *,
        url: str = DEFAULT_URL,
        token: str | None = None,
        model_id: str,
        model: str | None = None,
    ) -> None:
        self._session = async_get_clientsession(hass)
        self._url = url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._model_id = model_id
        self._model = model

    async def _request(self, method: str, path: str, *, json=None, timeout: int = 20):
        try:
            async with self._session.request(
                method,
                f"{self._url}{path}",
                json=json,
                headers=self._headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status == 204:
                    return None
                body = await resp.json(content_type=None)
                if resp.status == 422:
                    raise InsufficientData(
                        str(body.get("reason", "insufficient_data")),
                        code=body.get("reason"),
                        status=422,
                    )
                if resp.status >= 400:
                    raise ForecastServiceError(
                        body.get("message") or body.get("error", f"HTTP {resp.status}"),
                        code=body.get("error"),
                        status=resp.status,
                    )
                return body
        except asyncio.TimeoutError as err:
            raise ForecastServiceError(f"Timed out after {timeout}s on {path}") from err
        except aiohttp.ClientError as err:
            raise ForecastServiceError(f"Connection error: {err}") from err

    async def available(self) -> bool:
        try:
            async with self._session.get(
                f"{self._url}/health", timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    async def list_models(self) -> list[dict]:
        """Discover the backends this deployment offers.

        Use this to populate the config-flow selector instead of hard-coding
        names: a service that later gains an XGBoost backend exposes it here
        with no change to the integration.
        """
        body = await self._request("GET", "/models", timeout=10)
        return [m for m in body["models"] if m.get("available")]

    async def default_model(self) -> str | None:
        body = await self._request("GET", "/health", timeout=5)
        return body.get("default_model")

    async def train(
        self, series: list[HourActual], *, base_temp: float | None = None
    ) -> dict:
        payload: dict = {
            "model_id": self._model_id,
            "series": [h.as_json() for h in series],
        }
        if self._model:
            payload["model"] = self._model
        if base_temp is not None:
            payload["base_temp"] = base_temp

        body = await self._request("POST", "/train", json=payload, timeout=TRAIN_TIMEOUT)
        _LOGGER.info(
            "Trained %s: val_mae %s over %s hours",
            body.get("model"), body.get("val_mae"), body.get("n_hours"),
        )
        return body

    async def predict(
        self, future: list[HourFuture], history_tail: list[HourActual] | None = None
    ) -> dict[datetime, float]:
        payload: dict = {
            "model_id": self._model_id,
            "future": [h.as_json() for h in future],
        }
        if self._model:
            payload["model"] = self._model
        if history_tail:
            payload["history_tail"] = [h.as_json() for h in history_tail]

        body = await self._request("POST", "/predict", json=payload, timeout=PREDICT_TIMEOUT)
        return {
            datetime.fromisoformat(h["ts"]): h["kwh"] for h in body["hourly"]
        }

    async def instance_info(self, model: str) -> dict:
        """Diagnostics for a trained instance: val_mae, feature importance…"""
        return await self._request(
            "GET", f"/models/{model}/instances/{self._model_id}", timeout=10
        )

    async def delete_instance(self, model: str) -> None:
        """Call when the config entry is removed."""
        await self._request(
            "DELETE", f"/models/{model}/instances/{self._model_id}", timeout=10
        )


async def async_forecast(
    client: ForecastServiceClient,
    future: list[HourFuture],
    history_tail: list[HourActual],
    profile_model,
) -> tuple[dict[datetime, float], str]:
    """Return (forecast, source).

    Publish the source as a sensor attribute so users can see which model
    produced the number.
    """
    try:
        return await client.predict(future, history_tail), "service"
    except ForecastServiceError as err:
        _LOGGER.warning("Forecast service unavailable (%s) — using profile model.", err)
        return profile_model.predict(future), "profile"
