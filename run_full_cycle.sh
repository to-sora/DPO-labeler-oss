#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_full_cycle.sh --task-yaml TASK.yaml --global-seed SEED [--output-root DIR] [--python BIN]

Description:
  Compile a task YAML with an optional CLI global-seed override, split the compiled
  requests by shared ckpt + lora config, run every group with resume support, merge
  per-group run results, and collect final receipts/sessions.

Arguments:
  --task-yaml     Path to the task YAML to compile.
  --global-seed   Integer global seed override used at compile time.
  --output-root   Parent directory for generated artifacts.
                  The script creates a per-run subdirectory:
                  <output-root>/<task_scope>__<task_stem>__seed_<global_seed>
                  where <task_scope> is the parent folder name of the YAML when present
                  Default parent: output/batch
  --python        Python executable to use. Default: python
  -h, --help      Show this help message.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TASK_YAML=""
GLOBAL_SEED="2025"
OUTPUT_ROOT="output/batch"
PYTHON_BIN="${PYTHON_BIN:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-yaml)
      [[ $# -ge 2 ]] || { echo "Missing value for --task-yaml" >&2; exit 1; }
      TASK_YAML="$2"
      shift 2
      ;;
    --global-seed)
      [[ $# -ge 2 ]] || { echo "Missing value for --global-seed" >&2; exit 1; }
      GLOBAL_SEED="$2"
      shift 2
      ;;
    --output-root)
      [[ $# -ge 2 ]] || { echo "Missing value for --output-root" >&2; exit 1; }
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || { echo "Missing value for --python" >&2; exit 1; }
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

[[ -n "$TASK_YAML" ]] || { echo "--task-yaml is required" >&2; usage >&2; exit 1; }
[[ "$GLOBAL_SEED" =~ ^-?[0-9]+$ ]] || { echo "--global-seed must be an integer" >&2; exit 1; }
[[ -f "$TASK_YAML" ]] || { echo "Task YAML not found: $TASK_YAML" >&2; exit 1; }

task_scope_for() {
  local task_yaml="$1"
  local task_dir_base
  task_dir_base="$(basename "$(dirname "$task_yaml")")"
  case "$task_dir_base" in
    ""|"."|"/"|tasks|template|workflows)
      printf '\n'
      ;;
    *)
      printf '%s\n' "$task_dir_base"
      ;;
  esac
}

TASK_STEM="$(basename "${TASK_YAML%.*}")"
TASK_SCOPE="$(task_scope_for "$TASK_YAML")"
if [[ -z "$TASK_SCOPE" ]]; then
  RUN_LABEL="${TASK_STEM}__seed_${GLOBAL_SEED}"
else
  RUN_LABEL="${TASK_SCOPE}__${TASK_STEM}__seed_${GLOBAL_SEED}"
fi
OUTPUT_PARENT="$OUTPUT_ROOT"
if [[ "$(basename "$OUTPUT_PARENT")" == "$RUN_LABEL" ]]; then
  RUN_ROOT="$OUTPUT_PARENT"
else
  RUN_ROOT="${OUTPUT_PARENT}/${RUN_LABEL}"
fi

COMPILED_DIR="${RUN_ROOT}/compiled_run"
GROUPED_DIR="${RUN_ROOT}/grouped_requests"
GROUP_RUN_RESULTS_DIR="${RUN_ROOT}/group_run_results"
IMAGES_DIR="${RUN_ROOT}/images"
COLLECTED_DIR="${RUN_ROOT}/collected"
MERGED_RUN_RESULTS="${RUN_ROOT}/run_results.jsonl"
GROUP_MANIFEST="${GROUPED_DIR}/manifest.jsonl"

mkdir -p "$COMPILED_DIR" "$GROUPED_DIR" "$GROUP_RUN_RESULTS_DIR" "$IMAGES_DIR" "$COLLECTED_DIR"

echo "[1/5] Compiling ${TASK_YAML} with global seed ${GLOBAL_SEED}"
"$PYTHON_BIN" compile_yaml_to_requests_jsonl.py \
  --task-yaml "$TASK_YAML" \
  --output-dir "$COMPILED_DIR" \
  --global-seed "$GLOBAL_SEED"

echo "[2/5] Splitting compiled requests by shared config"
"$PYTHON_BIN" split_jsonl_by_shared_config.py \
  --input "${COMPILED_DIR}/requests.jsonl" \
  --output-dir "$GROUPED_DIR"

mapfile -t GROUP_JSONLS < <(
  "$PYTHON_BIN" - "$GROUP_MANIFEST" "$GROUPED_DIR" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
grouped_dir = Path(sys.argv[2])
with manifest_path.open("r", encoding="utf-8") as handle:
    for raw_line in handle:
        line = raw_line.strip()
        if not line:
            continue
        row = json.loads(line)
        print((grouped_dir / row["output_jsonl"]).as_posix())
PY
)

if [[ "${#GROUP_JSONLS[@]}" -eq 0 ]]; then
  echo "No grouped request files were generated." >&2
  exit 1
fi

echo "[3/5] Running ${#GROUP_JSONLS[@]} grouped batch(es)"
GROUP_RESULT_FILES=()
GENERATION_ERRORS=0
for GROUP_JSONL in "${GROUP_JSONLS[@]}"; do
  GROUP_NAME="$(basename "$GROUP_JSONL" .jsonl)"
  GROUP_RESULT_JSONL="${GROUP_RUN_RESULTS_DIR}/${GROUP_NAME}__run_results.jsonl"
  GROUP_IMAGE_DIR="${IMAGES_DIR}/${GROUP_NAME}"
  GROUP_RESULT_FILES+=("$GROUP_RESULT_JSONL")

  echo "  - Running group ${GROUP_NAME}"
  if ! "$PYTHON_BIN" run_workflows_jsonl.py \
      "$GROUP_JSONL" \
      --output "$GROUP_RESULT_JSONL" \
      --output-dir "$GROUP_IMAGE_DIR"; then
    GENERATION_ERRORS=1
  fi
done

echo "[4/5] Merging grouped run results into ${MERGED_RUN_RESULTS}"
: > "$MERGED_RUN_RESULTS"
for GROUP_RESULT_JSONL in "${GROUP_RESULT_FILES[@]}"; do
  [[ -f "$GROUP_RESULT_JSONL" ]] || { echo "Missing group run results: $GROUP_RESULT_JSONL" >&2; exit 1; }
  cat "$GROUP_RESULT_JSONL" >> "$MERGED_RUN_RESULTS"
done

echo "[5/5] Collecting receipts and sessions"
"$PYTHON_BIN" collect_receipts_and_sessions.py \
  --requests "${COMPILED_DIR}/requests.jsonl" \
  --run-results "$MERGED_RUN_RESULTS" \
  --output-dir "$COLLECTED_DIR"

echo
if [[ "$GENERATION_ERRORS" -eq 0 ]]; then
  echo "Cycle complete."
else
  echo "Cycle completed with generation errors." >&2
fi
echo "Run root: ${RUN_ROOT}"
echo "Compiled requests: ${COMPILED_DIR}/requests.jsonl"
echo "Merged run results: ${MERGED_RUN_RESULTS}"
echo "Receipts: ${COLLECTED_DIR}/receipts.jsonl"
echo "Sessions: ${COLLECTED_DIR}/sessions.jsonl"

if [[ "$GENERATION_ERRORS" -ne 0 ]]; then
  exit 1
fi
