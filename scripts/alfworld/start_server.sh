#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

ALFWORLD_ENV="/fs_mol/liujiarun/envs/alfworld"
ALFWORLD_PYTHON="${ALFWORLD_ENV}/bin/python"
ALFWORLD_PATH="/fs_mol/liujiarun/alfworld"
# Shared copy on /fs_mol: present on every node that mounts /fs_mol, unlike
# ~/.cache/alfworld which only exists where alfworld-download was run.
ALFWORLD_DATA="${ALFWORLD_DATA:-/fs_mol/liujiarun/alfworld/data}"
ALFWORLD_HOST="127.0.0.1"
# 8001 by design: the WebShop server owns 8000 and both may run on one machine.
ALFWORLD_PORT="${ALFWORLD_PORT:-8001}"
ALFWORLD_ENV_URL="http://${ALFWORLD_HOST}:${ALFWORLD_PORT}"
ALFWORLD_MAX_STEPS="${ALFWORLD_MAX_STEPS:-50}"

export PATH="${ALFWORLD_ENV}/bin:${PATH}"
export ALFWORLD_DATA

# Fail before spawning 96 envs if the data tree is missing: otherwise the
# server dies in startup with an empty split and only the log explains it.
if [[ ! -d "${ALFWORLD_DATA}/json_2.1.1/train" ]]; then
    echo "[start_server] ERROR: no ALFWorld data at ${ALFWORLD_DATA}/json_2.1.1."
    echo "[start_server] Expected the shared copy on /fs_mol; is /fs_mol mounted?"
    echo "[start_server] Or point ALFWORLD_DATA at an existing json_2.1.1 tree."
    exit 1
fi

NUM_ENVS="${ALFWORLD_NUM_ENVS:-128}"
READY_TIMEOUT=300
LOG_FILE="/tmp/alfworld_env_server.log"

# Refuse to start if something already answers on the port. Otherwise the new server
# fails to bind only after its startup work, and the readiness loop below would be
# polling the stale server the whole time.
if curl -s -o /dev/null --max-time 2 "${ALFWORLD_ENV_URL}/health"; then
    echo "[start_server] ERROR: ${ALFWORLD_ENV_URL} is already in use."
    echo "[start_server] Current /health: $(curl -s --max-time 2 "${ALFWORLD_ENV_URL}/health")"
    echo "[start_server] Running env servers:"
    pgrep -af "scripts/alfworld/server.py" || echo "  (none found; port held by another program)"
    echo "[start_server] Kill the stale server (kill <PID>) and retry."
    exit 1
fi

echo "[start_server] Starting ALFWorld env server (${NUM_ENVS} envs) on ${ALFWORLD_HOST}:${ALFWORLD_PORT}"
echo "[start_server] Data: ${ALFWORLD_DATA}"
echo "[start_server] Logs to ${LOG_FILE}"

"${ALFWORLD_PYTHON}" scripts/alfworld/server.py \
    --alfworld-path "${ALFWORLD_PATH}" \
    --alfworld-data "${ALFWORLD_DATA}" \
    --num-envs      "${NUM_ENVS}" \
    --max-steps     "${ALFWORLD_MAX_STEPS}" \
    --host          "${ALFWORLD_HOST}" \
    --port          "${ALFWORLD_PORT}" \
    > "${LOG_FILE}" 2>&1 &

SERVER_PID=$!
echo "[start_server] Server PID: ${SERVER_PID}"
echo "${SERVER_PID}" > /tmp/alfworld_env_server.pid

# Wait until /health returns idle > 0, i.e. at least one env is ready.
echo "[start_server] Waiting for server to be ready (timeout ${READY_TIMEOUT}s)…"
for i in $(seq 1 "${READY_TIMEOUT}"); do
    # Fail fast if the server died (import error, bind failure, OOM, …).
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[start_server] ERROR: server process ${SERVER_PID} exited after ${i}s. Last log lines:"
        tail -n 20 "${LOG_FILE}"
        exit 1
    fi
    if curl -sf "${ALFWORLD_ENV_URL}/health" | python3 -c \
        "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('idle',0)>0 else 1)" 2>/dev/null; then
        echo "[start_server] Server ready after ${i}s."
        curl -s "${ALFWORLD_ENV_URL}/health"
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
