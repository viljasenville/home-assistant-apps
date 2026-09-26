# Consumption Forecast Model Provider

A Home Assistant add-on that trains and serves consumption forecast models in
a **glibc-based container**, so the Alpine-based HA core never needs a musl
wheel. The integration stays thin: it owns the data, reads the recorder,
assembles the hourly series and sends it here. The service never queries Home
Assistant.

The HTTP contract is `docs/openapi.yaml`, and `tests/smoke_test.py` validates
every response against it.

## Model-agnostic by design

The service is not "the LightGBM add-on". Backends are discovered at runtime:

```
GET /models
{
  "default": "lightgbm",
  "models": [
    {"id": "lightgbm", "name": "LightGBM", "available": true,
     "min_hours": 504, "autoregressive": true,  "lag_hours": 168},
    {"id": "profile",  "name": "Hour-of-week profile", "available": true,
     "min_hours": 336, "autoregressive": false, "lag_hours": 0}
  ]
}
```

The integration reads this to populate its model choice instead of hard-coding
names, so a service that later gains an XGBoost backend exposes it with no
integration release. `train` and `predict` take an optional `model`; omit it
and the service uses its configured default.

Two backends ship today:

| Backend | Needs | `history_tail` | Notes |
| --- | --- | --- | --- |
| `lightgbm` | 504 h | 168 h | Gradient-boosted trees; the accurate one |
| `profile` | 336 h | — | Hour-of-week mean + heating term; no dependencies |

`profile` exists for two reasons: it is the accuracy reference the others are
measured against, and it keeps the service useful on hardware where LightGBM
cannot run. `available: false` appears for any backend whose library is
missing, so callers can tell "not installed" from "not known".

### Adding a backend

Subclass `ForecastBackend` in `app/backends/`, implement `fit`, `predict`,
`save` and `load`, and add it to `_BACKENDS` in `app/backends/__init__.py`.
The class attributes (`min_hours`, `autoregressive`, `lag_hours`) are what
`GET /models` reports and what `/predict` enforces. Nothing in the HTTP layer
changes.

## Architecture

```
HA core (Alpine/musl)                    Add-on (Debian/glibc)
┌────────────────────────┐               ┌──────────────────────────┐
│ integration            │  POST /train  │ FastAPI + uvicorn        │
│  ├─ recorder query     │──────────────▶│  ├─ backend registry     │
│  ├─ HTTP client        │  POST /predict│  ├─ feature pipeline     │
│  └─ profile model ◀────┼───────────────│  └─ /data/instances/...  │
│      (fallback)        │ error/timeout └──────────────────────────┘
└────────────────────────┘
        http://local-cfmp:8099
```

`model` and `model_id` are different things and both appear in most requests:
`model` is *which backend* (`lightgbm`), `model_id` is *whose trained model*
(the integration's config entry id). One service can therefore hold several
Home Assistant configurations, each with several backends trained under the
same id.

## Model design

Two things decide whether a forecast like this works in practice.

**Only features known at forecast time.** A one-hour energy lag is a tempting
feature, but it does not exist when forecasting twelve hours ahead. A model
trained on it looks excellent in validation and fails in production. The
`lightgbm` backend therefore derives from energy only:

- `energy_lag168` — the same hour one week earlier, available out to 168 h
- `origin_*` — aggregates as of the moment the forecast is **issued**,
  identical for every hour in that forecast

**The training distribution must match the prediction situation.** Every
training row draws a horizon between 1 and 48 hours and gets the matching
origin. The model sees as many "one hour ahead" cases as "36 hours ahead" ones
and learns to lean on the calendar and the weather rather than the latest
reading. The horizon is itself a feature.

Forecasting is **direct, not recursive**: every hour is predicted from the same
origin rather than fed back into the model, so errors do not compound along the
horizon. The backend still reports `autoregressive: true`, because that flag
tells callers "this model needs a `history_tail`", which it does.

Other features: calendar in local time (sine/cosine-encoded daily and weekly
cycles), outdoor temperature with lags and 24 h rolling statistics, and heating
and cooling degree hours from each hour's `target` (or `base_temp` when
absent). Missing values stay NaN — LightGBM handles them natively.

The objective is `regression_l1` (MAE), which tolerates consumption spikes
better than squared error. Validation is a chronological holdout, because a
random split would leak the future through the lag features.

## API

Every route except `/health` requires `Authorization: Bearer <api_token>` when
a token is configured. Errors are always `{"error": "<code>", "message": "…"}`.

### `POST /train`

```json
{
  "model": "lightgbm",
  "model_id": "1a2b3c4d5e6f",
  "base_temp": 17.0,
  "series": [
    {"ts": "2025-01-01T00:00:00+02:00", "energy": 1.84, "out_temp": -7.2}
  ]
}
```

```json
{
  "trained": true,
  "model": "lightgbm",
  "model_id": "1a2b3c4d5e6f",
  "val_mae": 0.2056,
  "baseline_val_mae": 0.6188,
  "n_hours": 4728,
  "trained_at": "2026-09-26T01:02:09+03:00"
}
```

`val_mae` is the time-ordered validation MAE in kWh/h, directly comparable
with the integration's own profile model. `baseline_val_mae` is an extra: the
hour-of-week baseline on the same holdout, so the answer to "is this backend
worth it" is in the response rather than a separate experiment.

Too little data returns **422** with `trained: false` and
`reason: "insufficient_data"` — a valid request with a negative answer, which
the integration treats like its own "not enough data" case. A malformed body is
**400**; an unknown or unavailable `model` is **404**.

### `POST /predict`

```json
{
  "model": "lightgbm",
  "model_id": "1a2b3c4d5e6f",
  "history_tail": [{"ts": "2025-03-09T00:00:00+02:00", "energy": 2.1, "out_temp": -5.0}],
  "future": [{"ts": "2025-03-17T00:00:00+02:00", "out_temp": -3.1}]
}
```

```json
{"model": "lightgbm",
 "hourly": [{"ts": "2025-03-17T00:00:00+02:00", "kwh": 2.13}]}
```

Forecast timestamps echo the offset the caller used, so they line up with what
was sent — including across a DST boundary.

Omitting `model` resolves whichever backend holds a trained instance under that
`model_id`. A `history_tail` shorter than the backend's `lag_hours` returns
**409** (`history_tail_too_short`); a missing one for an autoregressive backend
returns **409** (`history_tail_required`). An untrained `model_id` is **404**.

### Other routes

- `GET /health` — watchdog probe, unauthenticated, reports `default_model`
- `GET /models`, `GET /models/{model}` — backend discovery
- `GET /models/{model}/instances/{model_id}` — `val_mae`, `trained_at`,
  `n_hours`, feature importance
- `DELETE /models/{model}/instances/{model_id}` — idempotent, always 204

## Options

| Option | Default | Notes |
| --- | --- | --- |
| `api_token` | *(empty)* | Leave empty only if you trust everything on the network |
| `default_model` | `lightgbm` | Backend used when a request omits `model` |
| `base_temp` | 17.0 | Heating threshold (°C) for hours with no `target` |
| `num_threads` | 2 | OpenMP threads; 2 is enough on a Raspberry Pi |
| `max_history_rows` | 70000 | Older data is dropped beyond this (~8 years hourly) |
| `keep_model_versions` | 3 | Older versions stay on disk for manual rollback |

Trained models live under `/data/instances/<model>/<model_id>/`, which
Supervisor preserves across updates.

## Installation

1. Push this directory to your own GitHub repository (`repository.yaml` and
   `.github/` at the root, the add-on in its own subdirectory).
2. **Fix the `image` field in `config.yaml`** to point at your repository
   (lowercase). The publish workflow checks this and fails with a clear message
   if it points elsewhere.
3. Publish the image:
   `git tag cfmp-v2026.9.0 && git push origin cfmp-v2026.9.0`.
4. Make the GHCR packages public once (see below).
5. HA → **Settings → Add-ons → Add-on Store → ⋮ → Repositories** → your repo URL.
6. Install **Consumption Forecast Model Provider**, set `api_token`, start it.
7. Check the log for `Starting Consumption Forecast Model Provider on port 8099`.

### Prebuilt image vs. building on the device

The `image` field in `config.yaml` decides which applies.

| | `image` set (default) | `image` commented out |
| --- | --- | --- |
| Install time | ~1 min, download only | 5–15 min of building |
| Load on a Pi | none | pip install grinds |
| Publishing | needs a git tag and an Actions run | nothing |
| Development loop | slow (a tag per change) | fast (edit and restart) |

The publish workflow builds both architectures on native runners, so no QEMU
emulation and a few minutes per run. Before publishing it **starts the image
and checks that `/health` answers**, so a broken image never reaches a device.

### Several add-ons in one repository

The workflow discovers add-ons instead of listing them: any top-level
directory with a `config.yaml` counts, so a second add-on needs no workflow
change. Because versions are per add-on, release tags carry the slug —
`cfmp-v2026.9.0` — and a bare `v2026.9.0` is accepted only
while the repository holds exactly one add-on. A push or pull request builds
just the add-ons whose files changed, without publishing.

Each add-on can add three optional CI files. `.ci/test.sh` runs the add-on's
own suite before anything is built, on a runner with Python available — for
this add-on it installs the dev dependencies and runs `tests/smoke_test.py`, so
a contract regression fails in seconds instead of after two image builds; a
failure there stops the build, and so stops the publish. `.ci/options.json`
replaces the `config.yaml` defaults that the container is started with (this
one sets an `api_token` so the auth check has something to test), and
`.ci/verify.sh` runs against the live container after `/health` answers, with
`BASE_URL` set.

The selection logic lives in `.github/scripts/discover.py` and has its own
tests:

```bash
python3 .github/scripts/discover_test.py
```

They build a throwaway two-add-on repository and cover every trigger plus the
misconfigurations worth failing on — a tag whose version disagrees with
`config.yaml`, an ambiguous bare tag, an `image` field pointing at someone
else's repository.

> **Make the packages public, once.** GitHub → Packages →
> `amd64-addon-cfmp` → Package settings → Change visibility →
> Public. Supervisor has no GitHub credentials, so it cannot pull a private
> package. Repeat for each architecture.

The finished image is roughly 450 MB, most of it LightGBM, numpy and pandas.

## Integration side

`examples/forecast_client.py` contains a ready aiohttp client and the fallback
logic. Four things are worth keeping in mind:

- **The service is an optional accelerator.** Every call is bounded by a
  timeout and every failure falls back to the profile model. A missing add-on
  must not break the integration.
- **Discover models, don't hard-code them.** Populate the config flow from
  `GET /models` and filter on `available`.
- **Publish the active model as a sensor attribute** so users can see where a
  number came from.
- **Train rarely.** Once a night is plenty; predict as often as you like.

The add-on does not belong in the integration's `manifest.json` at all — the
dependency is HTTP, not a Python package. That is the entire point of this
arrangement.

## Limitations

- **amd64 and aarch64 only** for the LightGBM backend; `profile` runs anywhere,
  but armv7 has no LightGBM wheel.
- **Memory use** during training is roughly 200–400 MB for a year of data. On a
  1 GB device, cap it with `max_history_rows`.
- **No automatic retraining.** Scheduling belongs to the integration, which
  owns the data.
- **Training is synchronous.** A call can take minutes on slow hardware, so use
  a generous client timeout. Concurrent training of the same instance returns
  409 rather than queueing.

## Testing

```bash
pip install lightgbm numpy pandas fastapi uvicorn pydantic httpx jsonschema pyyaml
python3 tests/smoke_test.py
```

Runs the whole chain on synthetic data and validates all 17 response bodies
against `docs/openapi.yaml` — including the error envelopes and the 400 / 404 /
409 / 422 cases — so any drift between the implementation and the contract
fails the test. Verified against LightGBM 4.6–4.7 and pandas 2.2–3.0.
