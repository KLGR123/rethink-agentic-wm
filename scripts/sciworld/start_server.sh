#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

SCIWORLD_ENV="/fs_mol/liujiarun/envs/sciworld"
SCIWORLD_PYTHON="${SCIWORLD_ENV}/bin/python"
SCIWORLD_PATH="/fs_mol/liujiarun/ScienceWorld"
SCIWORLD_HOST="127.0.0.1"
# 8002 by design: WebShop owns 8000 and ALFWorld 8001, so all three may run on
# one machine during development.
SCIWORLD_PORT="${SCIWORLD_PORT:-8002}"
SCIWORLD_ENV_URL="http://${SCIWORLD_HOST}:${SCIWORLD_PORT}"
SCIWORLD_MAX_STEPS="${SCIWORLD_MAX_STEPS:-100}"
# Empty = full task. "easy" enables teleport + open doors/containers + no
# electrical actions, which shrinks the action space a lot.
SCIWORLD_SIMPLIFICATION="${SCIWORLD_SIMPLIFICATION:-}"

# ScienceWorldEnv launches the bundled JVM with plain `java` from PATH.
export PATH="${SCIWORLD_ENV}/bin:${PATH}"
export JAVA_HOME="${SCIWORLD_ENV}/lib/jvm"

# Fail before spawning N JVMs if the checkout or the jar is missing.
if [[ ! -f "${SCIWORLD_PATH}/scienceworld/scienceworld.jar" ]]; then
    echo "[start_server] ERROR: no ScienceWorld jar at ${SCIWORLD_PATH}/scienceworld."
    echo "[start_server] Expected the checkout on /fs_mol; is /fs_mol mounted?"
    exit 1
fi

NUM_ENVS="${SCIWORLD_NUM_ENVS:-64}"
READY_TIMEOUT=300
LOG_FILE="/tmp/sciworld_env_server.log"

# Refuse to start if something already answers on the port. Otherwise the new server
# fails to bind only after its startup work, and the readiness loop below would be
# polling the stale server the whole time.
if curl -s -o /dev/null --max-time 2 "${SCIWORLD_ENV_URL}/health"; then
    echo "[start_server] ERROR: ${SCIWORLD_ENV_URL} is already in use."
    echo "[start_server] Current /health: $(curl -s --max-time 2 "${SCIWORLD_ENV_URL}/health")"
    echo "[start_server] Running env servers:"
    pgrep -af "scripts/sciworld/server.py" || echo "  (none found; port held by another program)"
    echo "[start_server] Kill the stale server (kill <PID>) and retry."
    exit 1
fi

echo "[start_server] Starting ScienceWorld env server (${NUM_ENVS} JVMs) on ${SCIWORLD_HOST}:${SCIWORLD_PORT}"
echo "[start_server] Checkout: ${SCIWORLD_PATH}"
echo "[start_server] Simplifications: '${SCIWORLD_SIMPLIFICATION}'"
echo "[start_server] Logs to ${LOG_FILE}"

"${SCIWORLD_PYTHON}" scripts/sciworld/server.py \
    --scienceworld-path "${SCIWORLD_PATH}" \
    --num-envs         "${NUM_ENVS}" \
    --max-steps        "${SCIWORLD_MAX_STEPS}" \
    --simplification   "${SCIWORLD_SIMPLIFICATION}" \
    --host             "${SCIWORLD_HOST}" \
    --port             "${SCIWORLD_PORT}" \
    > "${LOG_FILE}" 2>&1 &

SERVER_PID=$!
echo "[start_server] Server PID: ${SERVER_PID}"
echo "${SERVER_PID}" > /tmp/sciworld_env_server.pid

# Wait until /health returns idle > 0, i.e. at least one env is ready.
echo "[start_server] Waiting for server to be ready (timeout ${READY_TIMEOUT}s)…"
for i in $(seq 1 "${READY_TIMEOUT}"); do
    # Fail fast if the server died (import error, bind failure, OOM, …).
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[start_server] ERROR: server process ${SERVER_PID} exited after ${i}s. Last log lines:"
        tail -n 20 "${LOG_FILE}"
        exit 1
    fi
    if curl -sf "${SCIWORLD_ENV_URL}/health" | python3 -c \
        "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('idle',0)>0 else 1)" 2>/dev/null; then
        echo "[start_server] Server ready after ${i}s."
        curl -s "${SCIWORLD_ENV_URL}/health"
        echo
        exit 0
    fi
    sleep 1
done

# Kill the server on timeout; left running it becomes an orphan holding the port,
# which breaks every later launch.
echo "[start_server] ERROR: server did not become ready in ${READY_TIMEOUT}s. Killing PID ${SERVER_PID}. Check ${LOG_FILE}."
kill "${SERVER_PID}" 2>/dev/null || true
exit 1
