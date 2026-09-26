"""Persistence for trained model instances.

Layout::

    /data/instances/<model>/<model_id>/
        current                 -> version identifier
        <version>/meta.json     -> val_mae, n_hours, trained_at, …
        <version>/…             -> whatever the backend writes

Versions are kept on disk so a bad retrain can be rolled back by hand; the
HTTP contract only ever exposes the current one.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

from .backends import ForecastBackend

_LOGGER = logging.getLogger(__name__)

SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class InstanceStore:
    def __init__(self, root: str, keep_versions: int = 3) -> None:
        self.root = root
        self.keep_versions = max(1, keep_versions)
        os.makedirs(self.root, exist_ok=True)
        self._cache: dict[tuple[str, str], tuple[object, dict]] = {}
        self._lock = threading.RLock()
        self._train_locks: dict[tuple[str, str], threading.Lock] = {}

    @staticmethod
    def validate_id(model_id: str) -> str:
        if not SAFE_ID.match(model_id):
            raise ValueError(
                "model_id may contain only letters, digits, dot, hyphen and "
                "underscore (max 128 characters)."
            )
        return model_id

    def _dir(self, model: str, model_id: str) -> str:
        return os.path.join(self.root, model, self.validate_id(model_id))

    def train_lock(self, model: str, model_id: str) -> threading.Lock:
        with self._lock:
            return self._train_locks.setdefault((model, model_id), threading.Lock())

    def save(
        self,
        backend: ForecastBackend,
        model_id: str,
        state: object,
        meta: dict,
        timezone: str,
    ) -> str:
        base = self._dir(backend.id, model_id)
        version = datetime.now(ZoneInfo(timezone)).strftime("%Y%m%dT%H%M%S")
        target = os.path.join(base, version)
        tmp = target + ".tmp"

        os.makedirs(tmp, exist_ok=True)
        try:
            backend.save(state, tmp)
            with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({**meta, "version": version}, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, target)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

        pointer_tmp = os.path.join(base, ".current.tmp")
        with open(pointer_tmp, "w", encoding="utf-8") as fh:
            fh.write(version)
        os.replace(pointer_tmp, os.path.join(base, "current"))

        with self._lock:
            self._cache.pop((backend.id, model_id), None)
        self._prune(backend.id, model_id)
        _LOGGER.info("Stored %s/%s version %s", backend.id, model_id, version)
        return version

    def _prune(self, model: str, model_id: str) -> None:
        versions = self.list_versions(model, model_id)
        keep = set(versions[-self.keep_versions :]) | {self.current_version(model, model_id)}
        for version in versions:
            if version not in keep:
                shutil.rmtree(os.path.join(self._dir(model, model_id), version), ignore_errors=True)

    def list_versions(self, model: str, model_id: str) -> list[str]:
        base = self._dir(model, model_id)
        if not os.path.isdir(base):
            return []
        return sorted(
            name
            for name in os.listdir(base)
            if os.path.isdir(os.path.join(base, name)) and not name.startswith(".")
        )

    def current_version(self, model: str, model_id: str) -> str | None:
        pointer = os.path.join(self._dir(model, model_id), "current")
        versions = self.list_versions(model, model_id)
        if os.path.isfile(pointer):
            with open(pointer, encoding="utf-8") as fh:
                version = fh.read().strip()
            if version in versions:
                return version
        return versions[-1] if versions else None

    def exists(self, model: str, model_id: str) -> bool:
        return self.current_version(model, model_id) is not None

    def find_model_for(self, model_id: str) -> str | None:
        """Which backend holds a trained instance under this id.

        Lets ``predict`` omit ``model`` and still reach the right instance.
        """
        if not os.path.isdir(self.root):
            return None
        for model in sorted(os.listdir(self.root)):
            if self.exists(model, model_id):
                return model
        return None

    def load(self, backend: ForecastBackend, model_id: str) -> tuple[object, dict]:
        version = self.current_version(backend.id, model_id)
        if version is None:
            raise FileNotFoundError(
                f"No trained model for model_id '{model_id}' on backend '{backend.id}'."
            )

        key = (backend.id, model_id)
        with self._lock:
            cached = self._cache.get(key)
            if cached and cached[1].get("version") == version:
                return cached

        path = os.path.join(self._dir(backend.id, model_id), version)
        state = backend.load(path)
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)

        with self._lock:
            self._cache[key] = (state, meta)
        return state, meta

    def meta(self, model: str, model_id: str) -> dict | None:
        version = self.current_version(model, model_id)
        if version is None:
            return None
        path = os.path.join(self._dir(model, model_id), version, "meta.json")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def delete(self, model: str, model_id: str) -> bool:
        base = self._dir(model, model_id)
        if not os.path.isdir(base):
            return False
        shutil.rmtree(base, ignore_errors=True)
        with self._lock:
            self._cache.pop((model, model_id), None)
        return True
