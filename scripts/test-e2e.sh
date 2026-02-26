#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-5176}"
APP_URL="http://127.0.0.1:${PORT}"
SERVER_LOG="${SERVER_LOG:-/tmp/brenizer-vite.log}"

npx vite --host 127.0.0.1 --strictPort --port "${PORT}" >"${SERVER_LOG}" 2>&1 &
VITE_PID=$!

cleanup() {
  kill "${VITE_PID}" >/dev/null 2>&1 || true
  wait "${VITE_PID}" 2>/dev/null || true
}
trap cleanup EXIT

for i in $(seq 1 60); do
  if curl -fsS "${APP_URL}" >/dev/null; then
    break
  fi
  if ! kill -0 "${VITE_PID}" >/dev/null 2>&1; then
    echo "Vite server exited before becoming ready. Log:" >&2
    cat "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if [ "${i}" -eq 60 ]; then
    echo "Timed out waiting for Vite dev server at ${APP_URL}" >&2
    exit 1
  fi
  sleep 1
done

APP_URL="${APP_URL}" node test.js
