#!/usr/bin/env bash
# Fleet-routing integration test (skypilot-fleet-provisioner Phase 2).
#
# Skipped when SKYPILOT_API_ENDPOINT is unset OR when the
# devai-skypilot-api-server container isn't running -- this test
# exercises the head's policy + provisioner integration end-to-end
# against a real (cheap) cloud worker, which costs money and needs
# real credentials.
#
# Usage:
#   SKYPILOT_API_ENDPOINT=http://devai-skypilot-api-server:46580 \
#     bash tests/test-fleet-routing.sh
#
# Exit codes:
#   0    success
#   77   skipped (per the canonical "skip" code)
#   non-zero otherwise

set -u

if [[ -z "${SKYPILOT_API_ENDPOINT:-}" ]]; then
    echo "SKIP: SKYPILOT_API_ENDPOINT not set; this test requires a running" >&2
    echo "      SkyPilot API server and real cloud credentials." >&2
    exit 77
fi

if ! curl -fsS --max-time 3 "${SKYPILOT_API_ENDPOINT}/api/v1/version" >/dev/null 2>&1; then
    echo "SKIP: SkyPilot API server at ${SKYPILOT_API_ENDPOINT} is not" >&2
    echo "      reachable. Run 'make skypilot-up' first." >&2
    exit 77
fi

echo ">>> Phase 2 fleet routing integration test (live cloud)"
echo "    endpoint: ${SKYPILOT_API_ENDPOINT}"

# 1. Verify head's local-fleet-only mode rejects an unfittable
#    request when SKYPILOT_API_ENDPOINT is unset on the head.
#    (Operator runs separately with the env var unset to confirm.)

# 2. With SKYPILOT_API_ENDPOINT set, send an unfittable request to
#    the head; expect a SkyPilot launch + worker registration +
#    successful response within the cold-start budget (5-15 min).
#
# This test surface is intentionally a skeleton -- the real test
# requires:
#   - operator-supplied cloud creds with non-zero quota
#   - acceptance of ~$1 spend per run
#   - DNS/network reachability from cloud worker -> head
#
# Track full implementation against the project board; until then
# this script documents the intended contract and exits 0 once the
# pre-flight passes (API server reachable + version returned).

VERSION_BODY=$(curl -fsS --max-time 5 "${SKYPILOT_API_ENDPOINT}/api/v1/version")
echo "    SkyPilot API server version: ${VERSION_BODY}"

echo ">>> test-fleet-routing pre-flight OK; full provisioning test deferred to E2E"
exit 0
