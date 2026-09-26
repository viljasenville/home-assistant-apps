"""Conformance test: every response is validated against docs/openapi.yaml.

Run on a development machine, not in the container. Beyond checking that
training and prediction work, this validates each response body against the
schema the contract declares for that status code, so a drift between the
implementation and the spec fails here.
"""

from __future__ import annotations

import copy
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

TMP = tempfile.mkdtemp(prefix="cfm-test-")
os.environ.update(
    CFM_DATA_DIR=TMP,
    CFM_API_TOKEN="testtoken",
    CFM_NUM_THREADS="2",
    CFM_TIMEZONE="Europe/Helsinki",
    CFM_LOG_LEVEL="warning",
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import jsonschema  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.api import app  # noqa: E402

AUTH = {"Authorization": "Bearer testtoken"}


def load_spec() -> dict:
    with open(os.path.join(ROOT, "docs", "openapi.yaml"), encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    return normalise_nullable(spec)


def normalise_nullable(node):
    """Translate OpenAPI 3.0 `nullable: true` into JSON Schema 2020-12.

    The spec document declares openapi 3.1.0, where schemas are plain JSON
    Schema and `nullable` is no longer a keyword — the 2020-12 spelling is
    `type: [number, "null"]`. A validator would silently ignore `nullable`
    and then reject a legitimate null, so it is translated here.
    """
    if isinstance(node, dict):
        node = {k: normalise_nullable(v) for k, v in node.items()}
        if node.pop("nullable", False) and "type" in node:
            t = node["type"]
            node["type"] = [t, "null"] if isinstance(t, str) else list(t) + ["null"]
        return node
    if isinstance(node, list):
        return [normalise_nullable(v) for v in node]
    return node


SPEC = load_spec()


def schema_for(path: str, method: str, status: int) -> dict | None:
    """The schema the contract declares for this operation and status."""
    operation = SPEC["paths"][path][method]
    response = operation["responses"].get(str(status))
    if response is None:
        return None
    if "$ref" in response:
        ref = response["$ref"].split("/")[-1]
        response = SPEC["components"]["responses"][ref]
    content = response.get("content")
    if not content:
        return None
    schema = content["application/json"]["schema"]
    # Validate against the whole document so internal $refs resolve.
    return {**copy.deepcopy(schema), "components": SPEC["components"]}


def synth(hours: int = 24 * 200, seed: int = 7) -> pd.DataFrame:
    """A series resembling household consumption: daily rhythm, weekend, heating."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-09-01", periods=hours, freq="h", tz="Europe/Helsinki")
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    doy = idx.dayofyear.to_numpy()

    temp = (
        5.0
        - 12.0 * np.cos(2 * np.pi * (doy - 15) / 365.0)
        + 3.0 * np.sin(2 * np.pi * (hour - 3) / 24.0)
        + rng.normal(0, 1.8, hours)
    )
    hdd = np.maximum(0.0, 17.0 - pd.Series(temp).rolling(12, min_periods=1).mean().to_numpy())
    profile = 0.45 + 0.55 * np.exp(-0.5 * ((hour - 7) / 1.6) ** 2)
    profile += 0.75 * np.exp(-0.5 * ((hour - 19) / 2.2) ** 2)
    profile *= np.where(dow >= 5, 1.18, 1.0)

    energy = 0.32 * hdd + profile + rng.normal(0, 0.12, hours)
    return pd.DataFrame({"ts": idx, "energy": np.maximum(energy, 0.05), "out_temp": temp})


def hours(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    out = []
    for rec in df.to_dict("records"):
        item = {"ts": pd.Timestamp(rec["ts"]).isoformat()}
        for c in cols:
            item[c] = float(rec[c])
        out.append(item)
    return out


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.validated = 0

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        print(f"{'  OK  ' if ok else ' FAIL '} {label}{(' — ' + detail) if detail else ''}")
        if not ok:
            self.failures.append(label)
        return ok

    def conforms(self, label: str, resp, path: str, method: str) -> None:
        """Check the status code's schema, then validate the body against it."""
        declared = SPEC["paths"][path][method]["responses"]
        if str(resp.status_code) not in declared:
            self.check(f"{label}: status {resp.status_code} is declared", False,
                       f"{method.upper()} {path} declares no {resp.status_code}")
            return
        if resp.status_code == 204:
            # Declared with a description and no content, so there is nothing
            # to validate beyond the body being empty.
            self.check(f"{label}: 204 carries no body", resp.content == b"")
            return
        schema = schema_for(path, method, resp.status_code)
        if schema is None:
            self.check(f"{label}: {resp.status_code} declares a schema", False)
            return
        try:
            jsonschema.validate(resp.json(), schema)
            self.validated += 1
            self.check(f"{label}: body matches the {resp.status_code} schema", True)
        except jsonschema.ValidationError as err:
            self.check(
                f"{label}: body matches the {resp.status_code} schema",
                False,
                f"{'.'.join(str(p) for p in err.absolute_path) or '(root)'}: {err.message}",
            )


def main() -> int:
    client = TestClient(app)
    c = Checker()
    data = synth()
    train_df, future_df = data.iloc[:-72], data.iloc[-72:]

    print("── meta and discovery " + "─" * 40)
    r = client.get("/health")
    c.check("GET /health is unauthenticated", r.status_code == 200)
    c.conforms("GET /health", r, "/health", "get")
    c.check("health reports a default model", bool(r.json().get("default_model")),
            r.json().get("default_model", ""))

    r = client.get("/models")
    c.check("GET /models without a token -> 401", r.status_code == 401)
    c.conforms("GET /models (401)", r, "/models", "get")
    c.check("401 body uses the error envelope", r.json().get("error") == "unauthorized",
            str(r.json()))

    r = client.get("/models", headers=AUTH)
    c.conforms("GET /models", r, "/models", "get")
    listed = {m["id"]: m for m in r.json()["models"]}
    c.check("several backends are discoverable", len(listed) >= 2, ", ".join(listed))
    c.check("lightgbm is available", listed.get("lightgbm", {}).get("available") is True)

    r = client.get("/models/lightgbm", headers=AUTH)
    c.conforms("GET /models/lightgbm", r, "/models/{model}", "get")

    r = client.get("/models/xgboost", headers=AUTH)
    c.check("unknown backend -> 404", r.status_code == 404)
    c.conforms("GET /models/xgboost (404)", r, "/models/{model}", "get")

    print("\n── training " + "─" * 50)
    series = hours(train_df, ["energy", "out_temp"])
    r = client.post("/train", json={"model_id": "entry1", "series": series,
                                    "base_temp": 17.0}, headers=AUTH)
    c.conforms("POST /train (default model)", r, "/train", "post")
    body = r.json()
    c.check("trained with the default backend", body.get("trained") is True
            and body["model"] == "lightgbm", str(body.get("model")))
    print(f"        val_mae {body.get('val_mae'):.4f} over {body.get('n_hours')} h"
          f"  |  baseline {body.get('baseline_val_mae'):.4f}")
    c.check("val_mae beats the hour-of-week baseline",
            body["val_mae"] < body["baseline_val_mae"])
    c.check("trained_at carries an explicit offset",
            bool(re.search(r"[+-]\d{2}:\d{2}$", body.get("trained_at", ""))),
            body.get("trained_at", ""))

    r = client.post("/train", json={"model": "profile", "model_id": "entry1",
                                    "series": series}, headers=AUTH)
    c.conforms("POST /train (profile backend)", r, "/train", "post")
    c.check("second backend trains under the same model_id",
            r.json().get("model") == "profile" and r.json().get("trained") is True)

    r = client.post("/train", json={"model": "xgboost", "model_id": "entry1",
                                    "series": series}, headers=AUTH)
    c.check("unavailable backend -> 404", r.status_code == 404)
    c.conforms("POST /train (unknown backend)", r, "/train", "post")

    r = client.post("/train", json={"model_id": "tiny",
                                    "series": series[:400]}, headers=AUTH)
    c.check("too little data -> 422", r.status_code == 422, str(r.status_code))
    c.conforms("POST /train (insufficient)", r, "/train", "post")
    c.check("422 says insufficient_data and trained=false",
            r.json().get("reason") == "insufficient_data" and r.json()["trained"] is False)

    r = client.post("/train", json={"series": series}, headers=AUTH)
    c.check("missing model_id -> 400", r.status_code == 400, str(r.status_code))
    c.conforms("POST /train (malformed)", r, "/train", "post")

    print("\n── forecasting " + "─" * 47)
    tail = hours(train_df.iloc[-192:], ["energy", "out_temp"])
    future = hours(future_df, ["out_temp"])

    r = client.post("/predict", json={"model": "lightgbm", "model_id": "entry1",
                                      "history_tail": tail, "future": future}, headers=AUTH)
    c.conforms("POST /predict", r, "/predict", "post")
    hourly = r.json()["hourly"]
    values = np.array([h["kwh"] for h in hourly])
    actual = future_df["energy"].to_numpy()
    mae = float(np.mean(np.abs(values - actual)))
    print(f"        72 h forecast MAE {mae:.4f} vs mean-only "
          f"{float(np.mean(np.abs(actual - actual.mean()))):.4f}")
    c.check("one forecast per requested hour", len(hourly) == len(future))
    c.check("no negative or NaN values", bool((values >= 0).all() and np.isfinite(values).all()))
    c.check("forecast beats a mean-only baseline",
            mae < float(np.mean(np.abs(actual - actual.mean()))))
    # The synthetic series crosses March, so the correct offset is +02:00
    # (Helsinki before DST) — echoing the caller's own offset, not a fixed one.
    sent_offset = future[0]["ts"][-6:]
    c.check("timestamps echo the caller's offset",
            hourly[0]["ts"].endswith(sent_offset),
            f"sent {sent_offset}, got {hourly[0]['ts']}")

    r = client.post("/predict", json={"model": "profile", "model_id": "entry1",
                                      "future": future}, headers=AUTH)
    c.conforms("POST /predict (no history_tail)", r, "/predict", "post")
    c.check("non-autoregressive backend needs no history_tail",
            r.status_code == 200 and len(r.json()["hourly"]) == len(future))

    r = client.post("/predict", json={"model": "lightgbm", "model_id": "entry1",
                                      "future": future}, headers=AUTH)
    c.check("autoregressive backend without a tail -> 409", r.status_code == 409)
    c.conforms("POST /predict (tail missing)", r, "/predict", "post")

    r = client.post("/predict", json={"model": "lightgbm", "model_id": "entry1",
                                      "history_tail": tail[-30:], "future": future},
                    headers=AUTH)
    c.check("tail shorter than the lag depth -> 409", r.status_code == 409)
    c.conforms("POST /predict (tail too short)", r, "/predict", "post")
    c.check("409 names the reason", r.json().get("error") == "history_tail_too_short",
            str(r.json().get("error")))

    r = client.post("/predict", json={"model_id": "entry1", "history_tail": tail,
                                      "future": future}, headers=AUTH)
    c.check("omitting model resolves the stored instance",
            r.status_code == 200 and r.json()["model"] in ("lightgbm", "profile"))

    r = client.post("/predict", json={"model_id": "nosuch", "future": future}, headers=AUTH)
    c.check("untrained model_id -> 404", r.status_code == 404)
    c.conforms("POST /predict (untrained)", r, "/predict", "post")

    print("\n── instances " + "─" * 49)
    path = "/models/{model}/instances/{model_id}"
    r = client.get("/models/lightgbm/instances/entry1", headers=AUTH)
    c.conforms("GET instance", r, path, "get")
    c.check("instance reports feature importance",
            bool(r.json().get("feature_importance")))

    r = client.get("/models/lightgbm/instances/nosuch", headers=AUTH)
    c.check("unknown instance -> 404", r.status_code == 404)
    c.conforms("GET instance (404)", r, path, "get")

    r = client.delete("/models/lightgbm/instances/entry1", headers=AUTH)
    c.conforms("DELETE instance", r, path, "delete")
    r = client.delete("/models/lightgbm/instances/entry1", headers=AUTH)
    c.check("delete is idempotent", r.status_code == 204)

    r = client.post("/predict", json={"model": "lightgbm", "model_id": "entry1",
                                      "history_tail": tail, "future": future}, headers=AUTH)
    c.check("deleted instance is gone -> 404", r.status_code == 404)

    print()
    print(f"{c.validated} response bodies validated against docs/openapi.yaml")
    if c.failures:
        print(f"FAILED: {len(c.failures)} checks -> {c.failures}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
