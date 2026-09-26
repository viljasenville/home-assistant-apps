#!/usr/bin/env bash
# The conformance suite, run by CI before any image is built. It exercises the
# app in-process with FastAPI's TestClient, so it needs no container and fails
# fast on contract drift between app/ and docs/openapi.yaml.
#
# The workflow runs this from the add-on directory with a Python available.
set -euo pipefail

cd "$(dirname "$0")/.."

# The runtime dependencies, plus the three the tests themselves need:
# httpx drives TestClient, jsonschema validates responses against the
# OpenAPI schemas, pyyaml reads the spec.
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r requirements.txt httpx jsonschema pyyaml

python3 tests/smoke_test.py
