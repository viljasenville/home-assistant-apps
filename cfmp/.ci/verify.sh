#!/usr/bin/env bash
# Add-on specific checks against the running container, beyond /health.
# The workflow runs this with BASE_URL set once /health answers.
set -euo pipefail

: "${BASE_URL:?BASE_URL is not set}"
TOKEN="smoketest"   # matches .ci/options.json

fail() { echo "::error::$1"; exit 1; }

code="$(curl -s -o /dev/null -w '%{http_code}' "${BASE_URL}/models")"
[ "${code}" = "401" ] || fail "Authentication is not in effect (GET /models returned ${code}, expected 401)."
echo "Authentication OK (401 without a token)."

models="$(curl -fsS -H "Authorization: Bearer ${TOKEN}" "${BASE_URL}/models")"
echo "${models}"

echo "${models}" | grep -q '"default"' || fail "GET /models has no default model."
echo "${models}" | grep -q '"id": *"lightgbm"' || fail "The lightgbm backend is not listed."
echo "${models}" | grep -q '"available": *true' || fail "No backend reports itself available."
echo "Backend discovery OK."
