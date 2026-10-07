# shellcheck shell=bash
# A curl client on devai-net for the live router test scripts.
#
# The router publishes no host ports, so every request these scripts send
# has to come from a container on devai-net. They used to exec into
# devai-open-webui, which was removed on 2026-10-07 (platform-rebuild
# Phase 1, M9). Source this file, then call devai_net_client_start AFTER
# any `trap ... EXIT` of your own: it sets EXEC_HOST and chains its cleanup
# onto the existing EXIT trap. Every request then runs as
#   $RUNTIME exec "$EXEC_HOST" curl ...
#
# Environment:
#   CONTAINER_RUNTIME  podman (default) or docker
#   TEST_EXEC_HOST     an existing container on devai-net to use instead;
#                      nothing is started or removed then
#   TEST_CLIENT_IMAGE  image for the throwaway client. Default: the lab
#                      image, localhost/devai-lab-gpu:latest, falling back to
#                      localhost/devai-lab-cpu:latest (both carry curl and
#                      bake in no proxy)
#
# The throwaway client runs `sleep infinity` as PID 1, which ignores SIGTERM,
# so it is killed (SIGKILL) rather than stopped.

DEVAI_NET_CLIENT_NAME=""

devai_net_client_stop() {
    [ -n "$DEVAI_NET_CLIENT_NAME" ] || return 0
    "${RUNTIME:-${CONTAINER_RUNTIME:-podman}}" kill "$DEVAI_NET_CLIENT_NAME" >/dev/null 2>&1 || true
    "${RUNTIME:-${CONTAINER_RUNTIME:-podman}}" rm -f "$DEVAI_NET_CLIENT_NAME" >/dev/null 2>&1 || true
    DEVAI_NET_CLIENT_NAME=""
}

# The command of the EXIT trap currently set, or nothing.
_devai_net_client_prev_exit_trap() {
    local spec
    spec=$(trap -p EXIT)
    [ -n "$spec" ] || return 0
    # `trap -p` prints a re-usable `trap -- '<cmd>' EXIT`; let the shell
    # undo its quoting.
    eval "set -- $spec"
    printf '%s' "$3"
}

devai_net_client_start() {
    local runtime="${RUNTIME:-${CONTAINER_RUNTIME:-podman}}"
    if [ -n "${TEST_EXEC_HOST:-}" ]; then
        if "$runtime" container inspect "$TEST_EXEC_HOST" >/dev/null 2>&1; then
            EXEC_HOST="$TEST_EXEC_HOST"
            return 0
        fi
        echo "devai-net-client: TEST_EXEC_HOST=$TEST_EXEC_HOST is not a container; starting a client instead" >&2
    fi

    local image="${TEST_CLIENT_IMAGE:-}"
    if [ -z "$image" ]; then
        for image in localhost/devai-lab-gpu:latest localhost/devai-lab-cpu:latest ""; do
            [ -n "$image" ] || break
            "$runtime" image inspect "$image" >/dev/null 2>&1 && break
        done
    fi
    if [ -z "$image" ]; then
        echo "devai-net-client: no client image: neither localhost/devai-lab-gpu:latest nor localhost/devai-lab-cpu:latest exists (make build-gpu), and TEST_CLIENT_IMAGE is unset" >&2
        return 1
    fi

    local name="devai-test-client-$$"
    if ! "$runtime" run -d --rm --name "$name" --network devai-net \
            --entrypoint sleep "$image" infinity >/dev/null; then
        echo "devai-net-client: could not start $name from $image on devai-net (is the network up? make cache-up)" >&2
        return 1
    fi
    DEVAI_NET_CLIENT_NAME="$name"
    EXEC_HOST="$name"

    local prev
    prev=$(_devai_net_client_prev_exit_trap)
    # shellcheck disable=SC2064  # expand $prev now: it is the old trap's command
    trap "devai_net_client_stop${prev:+; $prev}" EXIT
}
