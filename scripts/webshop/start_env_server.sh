#!/usr/bin/env bash

set -euo pipefail

WEBSHOP_ENV="/fs_mol/liujiarun/envs/webshop"
WEBSHOP_PYTHON="${WEBSHOP_ENV}/bin/python"
WEBSHOP_PATH="/fs_mol/liujiarun/WebShop"
WEBSHOP_FILE_PATH="/fs_mol/liujiarun/WebShop/data/items_shuffle.json"
WEBSHOP_HOST="127.0.0.1"
WEBSHOP_PORT="8000"
WEBSHOP_ENV_URL="http://${WEBSHOP_HOST}:${WEBSHOP_PORT}"

export JAVA_HOME="${WEBSHOP_ENV}/lib/jvm"
export JVM_PATH="${JAVA_HOME}/lib/server/libjvm.so"
export LD_LIBRARY_PATH="${WEBSHOP_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${WEBSHOP_ENV}/bin:${JAVA_HOME}/bin:${PATH}"

NUM_ENVS=48
READY_TIMEOUT=300
LOG_FILE="/tmp/webshop_env_server.log"

# Refuse to start if something already answers on the port. Otherwise the new server
# fails to bind only after its ~2 min load, and the readiness loop below would be
# polling the stale server the whole time.
if env -u LD_LIBRARY_PATH curl -s -o /dev/null --max-time 2 "${WEBSHOP_ENV_URL}/health"; then
    echo "[start_env_server] ERROR: ${WEBSHOP_ENV_URL} is already in use."
    echo "[start_env_server] Current /health: $(env -u LD_LIBRARY_PATH curl -s --max-time 2 "${WEBSHOP_ENV_URL}/health")"
    echo "[start_env_server] Running env servers:"
    pgrep -af "webshop_env_server.py" || echo "  (none found; port held by another program)"
    echo "[start_env_server] Kill the stale server (kill <PID>) and retry."
    exit 1
fi

echo "[start_env_server] Starting WebShop env server (${NUM_ENVS} envs) on ${WEBSHOP_HOST}:${WEBSHOP_PORT}"
echo "[start_env_server] Logs to ${LOG_FILE}"

"${WEBSHOP_PYTHON}" scripts/webshop/webshop_env_server.py \
    --webshop-path "${WEBSHOP_PATH}" \
    --num-envs     "${NUM_ENVS}" \
    --host         "${WEBSHOP_HOST}" \
    --port         "${WEBSHOP_PORT}" \
    --file-path    "${WEBSHOP_FILE_PATH}" \
    > "${LOG_FILE}" 2>&1 &

SERVER_PID=$!
echo "[start_env_server] Server PID: ${SERVER_PID}"
echo "${SERVER_PID}" > /tmp/webshop_env_server.pid

# Wait until /health returns idle > 0, i.e. at least one env is ready.
echo "[start_env_server] Waiting for server to be ready (timeout ${READY_TIMEOUT}s)…"
for i in $(seq 1 "${READY_TIMEOUT}"); do
    # Fail fast if the server died (import error, bind failure, OOM, …).
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[start_env_server] ERROR: server process ${SERVER_PID} exited after ${i}s. Last log lines:"
        tail -n 20 "${LOG_FILE}"
        exit 1
    fi
    if env -u LD_LIBRARY_PATH curl -sf "${WEBSHOP_ENV_URL}/health" | python3 -c \
        "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('idle',0)>0 else 1)" 2>/dev/null; then
        echo "[start_env_server] Server ready after ${i}s."
        env -u LD_LIBRARY_PATH curl -s "${WEBSHOP_ENV_URL}/health"
        echo
        exit 0
    fi
    sleep 1
done

# Kill the server on timeout; left running it becomes an orphan holding the port,
# which breaks every later launch.
echo "[start_env_server] ERROR: server did not become ready in ${READY_TIMEOUT}s. Killing PID ${SERVER_PID}. Check ${LOG_FILE}."
kill "${SERVER_PID}" 2>/dev/null || true
exit 1
