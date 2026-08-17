#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_FILE="${OSS_ENV_FILE:-$SCRIPT_DIR/.env.oss}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE. Run: bash install.sh" >&2
  exit 2
fi

set -a
. "$ENV_FILE"
set +a

detect_tailscale_ip() {
  if ! command -v tailscale >/dev/null 2>&1; then
    return 1
  fi
  tailscale ip -4 2>/dev/null | awk 'NF { print $1; exit }'
}

TAILSCALE_IP="$(detect_tailscale_ip || true)"
case "${BIND_MODE:-local}" in
  local)
    BIND_HOST="127.0.0.1"
    ;;
  all)
    BIND_HOST="0.0.0.0"
    ;;
  tailscale)
    if [[ -z "$TAILSCALE_IP" ]]; then
      echo "Could not detect a Tailscale IPv4 address. Re-run install.sh and choose local." >&2
      exit 2
    fi
    BIND_HOST="$TAILSCALE_IP"
    ;;
  *)
    BIND_HOST="${BIND_MODE}"
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python3}"
GRADER_PYTHON_BIN="${GRADER_PYTHON_BIN:-$PYTHON_BIN}"
RUNTIME_ROOT="${RUNTIME_ROOT:-temp/oss_endpoints}"
LOGS_DIR="${LOGS_DIR:-$RUNTIME_ROOT/logs}"
DATASET_ROOT="${DATASET_ROOT:-output}"
IMAGE_ROOT="${IMAGE_ROOT:-$DATASET_ROOT}"
LABELER_STATE_DIR="${LABELER_STATE_DIR:-$RUNTIME_ROOT/labeler_state}"
EXPORT_STATE_DIR="${EXPORT_STATE_DIR:-$RUNTIME_ROOT/export_viewer_state}"
GRADER_STATE_DIR="${GRADER_STATE_DIR:-$RUNTIME_ROOT/image_grader_state}"
ADAPTER_WORK_DIR="${ADAPTER_WORK_DIR:-$RUNTIME_ROOT/image_grader_adapter}"
GRADER_CONFIG="${GRADER_CONFIG:-image_grader/config.local.json}"
LABELER_PORT="${LABELER_PORT:-8787}"
EXPORT_VIEWER_PORT="${EXPORT_VIEWER_PORT:-8084}"
IMAGE_GRADER_PORT="${IMAGE_GRADER_PORT:-8790}"
ADAPTER_PORT="${ADAPTER_PORT:-8087}"
INVITE_TOKEN="${INVITE_TOKEN:?INVITE_TOKEN must be configured by install.sh}"
REVIEW_ROUND_SEED="${REVIEW_ROUND_SEED:-round-1}"

if [[ ! -x "$(command -v "$PYTHON_BIN" 2>/dev/null || true)" && ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ "${START_IMAGE_GRADER:-1}" == "1" || "${START_ADAPTER:-1}" == "1" ]]; then
  if [[ ! -x "$(command -v "$GRADER_PYTHON_BIN" 2>/dev/null || true)" && ! -x "$GRADER_PYTHON_BIN" ]]; then
    echo "Grader Python executable not found: $GRADER_PYTHON_BIN" >&2
    exit 2
  fi
  if [[ ! -f "$GRADER_CONFIG" ]]; then
    echo "Grader config not found: $GRADER_CONFIG. Run: bash install.sh" >&2
    exit 2
  fi
fi

mkdir -p \
  "$DATASET_ROOT" \
  "$IMAGE_ROOT" \
  "$LOGS_DIR" \
  "$LABELER_STATE_DIR" \
  "$EXPORT_STATE_DIR" \
  "$GRADER_STATE_DIR" \
  "$ADAPTER_WORK_DIR"

export PYTHONPATH="$SCRIPT_DIR/image_grader:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PIDS=()
NAMES=()
LOGS=()

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo
    echo "Stopping ${#PIDS[@]} service(s)..."
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
      fi
    done
    for _ in {1..25}; do
      local alive=0
      for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
          alive=1
          break
        fi
      done
      [[ "$alive" -eq 0 ]] && break
      sleep 0.2
    done
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
      fi
    done
    wait 2>/dev/null || true
  fi
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

start_service() {
  local name="$1"
  local log="$2"
  shift 2
  echo "Starting $name on $BIND_HOST -> $log"
  if command -v setsid >/dev/null 2>&1; then
    setsid "$@" >"$log" 2>&1 &
  else
    "$@" >"$log" 2>&1 &
  fi
  local pid=$!
  PIDS+=("$pid")
  NAMES+=("$name")
  LOGS+=("$log")
}

if [[ "${START_LABELER:-1}" == "1" ]]; then
  start_service "labeler" "$LOGS_DIR/labeler.log" \
    "$PYTHON_BIN" -m dpo_labeler.backend.server \
      --dataset-root "$DATASET_ROOT" \
      --state-dir "$LABELER_STATE_DIR" \
      --invite-token "$INVITE_TOKEN" \
      --review-round-seed "$REVIEW_ROUND_SEED" \
      --host "$BIND_HOST" \
      --port "$LABELER_PORT"
fi

if [[ "${START_EXPORT_VIEWER:-1}" == "1" ]]; then
  start_service "export-viewer" "$LOGS_DIR/export-viewer.log" \
    "$PYTHON_BIN" -m dpo_labeler.export_viewer.server \
      --state-dir "$EXPORT_STATE_DIR" \
      --image-root "$IMAGE_ROOT" \
      --host "$BIND_HOST" \
      --port "$EXPORT_VIEWER_PORT"
fi

if [[ "${START_IMAGE_GRADER:-1}" == "1" ]]; then
  start_service "image-grader" "$LOGS_DIR/image-grader.log" \
    "$GRADER_PYTHON_BIN" -m image_grader \
      --config "$GRADER_CONFIG" \
      serve \
      --state-dir "$GRADER_STATE_DIR" \
      --host "$BIND_HOST" \
      --port "$IMAGE_GRADER_PORT"
fi

if [[ "${START_ADAPTER:-1}" == "1" ]]; then
  start_service "image-grader-admin" "$LOGS_DIR/image-grader-admin.log" \
    "$GRADER_PYTHON_BIN" -m image_grader_adapter_ui \
      --work-dir "$ADAPTER_WORK_DIR" \
      --dataset-root "$DATASET_ROOT" \
      --grader-config "$GRADER_CONFIG" \
      --host "$BIND_HOST" \
      --port "$ADAPTER_PORT"
fi

if [[ ${#PIDS[@]} -eq 0 ]]; then
  echo "No services selected. Re-run install.sh and enable at least one endpoint." >&2
  exit 2
fi

sleep 1
for index in "${!PIDS[@]}"; do
  pid="${PIDS[$index]}"
  if ! kill -0 "$pid" 2>/dev/null; then
    status=0
    wait "$pid" || status=$?
    echo "${NAMES[$index]} exited during startup with status $status. Log: ${LOGS[$index]}" >&2
    exit "$status"
  fi
done

echo
echo "Services are running. Logs: $LOGS_DIR"
if [[ "$BIND_HOST" == "0.0.0.0" ]]; then
  PRINT_HOSTS=("127.0.0.1")
  [[ -n "$TAILSCALE_IP" ]] && PRINT_HOSTS+=("$TAILSCALE_IP")
else
  PRINT_HOSTS=("$BIND_HOST")
fi
for host in "${PRINT_HOSTS[@]}"; do
  [[ "${START_LABELER:-1}" == "1" ]] && echo "labeler:            http://$host:$LABELER_PORT/"
  [[ "${START_EXPORT_VIEWER:-1}" == "1" ]] && echo "export-viewer:      http://$host:$EXPORT_VIEWER_PORT/"
  [[ "${START_IMAGE_GRADER:-1}" == "1" ]] && echo "image-grader:       http://$host:$IMAGE_GRADER_PORT/"
  [[ "${START_ADAPTER:-1}" == "1" ]] && echo "image-grader-admin: http://$host:$ADAPTER_PORT/"
done
echo
echo "Press Ctrl-C to stop all services."

while true; do
  for index in "${!PIDS[@]}"; do
    pid="${PIDS[$index]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      status=0
      wait "$pid" || status=$?
      echo "${NAMES[$index]} exited with status $status. Log: ${LOGS[$index]}" >&2
      exit "$status"
    fi
  done
  sleep 2
done
