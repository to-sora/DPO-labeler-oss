#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_multi_task_scheduled_cycle.sh --task-yaml TASK1.yaml --task-yaml TASK2.yaml [--task-yaml TASKN.yaml ...]
                                     [--global-seed SEED] [--output-root DIR] [--python BIN]

Description:
  Compile multiple task YAMLs with the same global seed, split each task's requests by
  shared ckpt + lora config, then run the grouped JSONLs in a global schedule that keeps
  matching model/config groups adjacent across tasks. This avoids unnecessary model switching
  on the backend without changing the existing single-task scripts.

Arguments:
  --task-yaml     Path to a task YAML. Repeat this flag for each task to schedule.
  --global-seed   Integer global seed override used for every task. Default: 2025
  --output-root   Parent directory for generated artifacts.
                  The script creates a scheduled-run subdirectory:
                  <output-root>/multi_schedule__<task_count>tasks__seed_<global_seed>__<hash>
                  Default parent: output/batch_schedule
  --python        Python executable to use. Default: python
  -h, --help      Show this help message.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TASK_YAMLS=()
GLOBAL_SEED="2025"
OUTPUT_ROOT="output/batch_schedule"
PYTHON_BIN="${PYTHON_BIN:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-yaml)
      [[ $# -ge 2 ]] || { echo "Missing value for --task-yaml" >&2; exit 1; }
      TASK_YAMLS+=("$2")
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

[[ "${#TASK_YAMLS[@]}" -gt 0 ]] || { echo "At least one --task-yaml is required" >&2; usage >&2; exit 1; }
[[ "$GLOBAL_SEED" =~ ^-?[0-9]+$ ]] || { echo "--global-seed must be an integer" >&2; exit 1; }

for TASK_YAML in "${TASK_YAMLS[@]}"; do
  [[ -f "$TASK_YAML" ]] || { echo "Task YAML not found: $TASK_YAML" >&2; exit 1; }
done

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

RUN_HASH="$("$PYTHON_BIN" - "$GLOBAL_SEED" "${TASK_YAMLS[@]}" <<'PY'
import hashlib
import sys
seed = sys.argv[1]
task_paths = [item.strip() for item in sys.argv[2:] if item.strip()]
payload = "\n".join([seed, *task_paths]).encode("utf-8")
print(hashlib.sha256(payload).hexdigest()[:12])
PY
)"

RUN_LABEL="multi_schedule__${#TASK_YAMLS[@]}tasks__seed_${GLOBAL_SEED}__${RUN_HASH}"
OUTPUT_PARENT="$OUTPUT_ROOT"
if [[ "$(basename "$OUTPUT_PARENT")" == "$RUN_LABEL" ]]; then
  RUN_ROOT="$OUTPUT_PARENT"
else
  RUN_ROOT="${OUTPUT_PARENT}/${RUN_LABEL}"
fi

TASKS_ROOT="${RUN_ROOT}/tasks"
SCHEDULE_DIR="${RUN_ROOT}/schedule"
SCHEDULE_MANIFEST="${SCHEDULE_DIR}/group_schedule.jsonl"

mkdir -p "$TASKS_ROOT" "$SCHEDULE_DIR"

task_label_for() {
  local task_yaml="$1"
  local task_stem task_scope
  task_stem="$(basename "${task_yaml%.*}")"
  task_scope="$(task_scope_for "$task_yaml")"
  if [[ -z "$task_scope" ]]; then
    printf '%s\n' "${task_stem}"
  else
    printf '%s\n' "${task_scope}__${task_stem}"
  fi
}

TASK_REQUESTS=()
TASK_GROUP_MANIFESTS=()
TASK_RUN_ROOTS=()

echo "[1/6] Compiling ${#TASK_YAMLS[@]} task(s) with global seed ${GLOBAL_SEED}"
for TASK_YAML in "${TASK_YAMLS[@]}"; do
  TASK_LABEL="$(task_label_for "$TASK_YAML")"
  TASK_RUN_ROOT="${TASKS_ROOT}/${TASK_LABEL}__seed_${GLOBAL_SEED}"
  TASK_COMPILED_DIR="${TASK_RUN_ROOT}/compiled_run"
  TASK_GROUPED_DIR="${TASK_RUN_ROOT}/grouped_requests"
  TASK_GROUP_RUN_RESULTS_DIR="${TASK_RUN_ROOT}/group_run_results"
  TASK_IMAGES_DIR="${TASK_RUN_ROOT}/images"
  TASK_COLLECTED_DIR="${TASK_RUN_ROOT}/collected"

  mkdir -p "$TASK_COMPILED_DIR" "$TASK_GROUPED_DIR" "$TASK_GROUP_RUN_RESULTS_DIR" "$TASK_IMAGES_DIR" "$TASK_COLLECTED_DIR"

  echo "  - Compiling ${TASK_YAML}"
  "$PYTHON_BIN" compile_yaml_to_requests_jsonl.py \
    --task-yaml "$TASK_YAML" \
    --output-dir "$TASK_COMPILED_DIR" \
    --global-seed "$GLOBAL_SEED"

  TASK_REQUESTS+=("${TASK_COMPILED_DIR}/requests.jsonl")
  TASK_GROUP_MANIFESTS+=("${TASK_GROUPED_DIR}/manifest.jsonl")
  TASK_RUN_ROOTS+=("$TASK_RUN_ROOT")
done

echo "[2/6] Splitting compiled requests per task"
for TASK_RUN_ROOT in "${TASK_RUN_ROOTS[@]}"; do
  TASK_COMPILED_DIR="${TASK_RUN_ROOT}/compiled_run"
  TASK_GROUPED_DIR="${TASK_RUN_ROOT}/grouped_requests"
  TASK_REQUESTS_JSONL="${TASK_COMPILED_DIR}/requests.jsonl"
  echo "  - Splitting $(basename "$TASK_RUN_ROOT")"
  "$PYTHON_BIN" split_jsonl_by_shared_config.py \
    --input "$TASK_REQUESTS_JSONL" \
    --output-dir "$TASK_GROUPED_DIR"
done

echo "[3/6] Building global group schedule"
"$PYTHON_BIN" - "$SCHEDULE_MANIFEST" "$TASKS_ROOT" "${TASK_YAMLS[@]}" <<'PY'
import json
import sys
from pathlib import Path


def task_label_for(task_yaml: str) -> str:
    task_path = Path(task_yaml)
    task_stem = task_path.stem
    task_dir_base = task_path.parent.name
    if task_dir_base in {"", ".", "/", "tasks", "template", "workflows"}:
        return task_stem
    return f"{task_dir_base}__{task_stem}"


schedule_manifest = Path(sys.argv[1])
tasks_root = Path(sys.argv[2])
task_yamls = [Path(item) for item in sys.argv[3:]]

schedule_manifest.parent.mkdir(parents=True, exist_ok=True)
if schedule_manifest.exists():
    schedule_manifest.unlink()

group_rows = []
first_seen_key_order = {}

for task_order, task_yaml in enumerate(task_yamls):
    task_label = task_label_for(task_yaml.as_posix())
    task_run_root_candidates = sorted(tasks_root.glob(f"{task_label}__seed_*"))
    if not task_run_root_candidates:
        raise SystemExit(f"Missing run root for task {task_yaml}")
    task_run_root = task_run_root_candidates[0]
    manifest_path = task_run_root / "grouped_requests" / "manifest.jsonl"
    grouped_dir = task_run_root / "grouped_requests"
    group_results_dir = task_run_root / "group_run_results"
    images_dir = task_run_root / "images"

    with manifest_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            schedule_key = json.dumps(
                {
                    "ckpt": row["ckpt"],
                    "lora_stack_config": row["lora_stack_config"],
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if schedule_key not in first_seen_key_order:
                first_seen_key_order[schedule_key] = len(first_seen_key_order)

            output_jsonl = row["output_jsonl"]
            group_name = Path(output_jsonl).stem
            group_rows.append(
                {
                    "schedule_key": schedule_key,
                    "schedule_order": first_seen_key_order[schedule_key],
                    "task_order": task_order,
                    "group_index": int(row["group_index"]),
                    "task_yaml": task_yaml.as_posix(),
                    "task_label": task_label,
                    "task_run_root": task_run_root.as_posix(),
                    "ckpt": row["ckpt"],
                    "lora_stack_config": row["lora_stack_config"],
                    "record_count": int(row["record_count"]),
                    "group_jsonl": (grouped_dir / output_jsonl).as_posix(),
                    "group_result_jsonl": (group_results_dir / f"{group_name}__run_results.jsonl").as_posix(),
                    "group_image_dir": (images_dir / group_name).as_posix(),
                }
            )

group_rows.sort(key=lambda row: (row["schedule_order"], row["task_order"], row["group_index"]))

with schedule_manifest.open("w", encoding="utf-8") as handle:
    for run_index, row in enumerate(group_rows, start=1):
        row = dict(row)
        row["run_index"] = run_index
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY

GROUP_COUNT="$("$PYTHON_BIN" - "$SCHEDULE_MANIFEST" <<'PY'
import sys
from pathlib import Path
manifest = Path(sys.argv[1])
count = 0
with manifest.open("r", encoding="utf-8") as handle:
    for raw_line in handle:
        if raw_line.strip():
            count += 1
print(count)
PY
)"

[[ "$GROUP_COUNT" -gt 0 ]] || { echo "No scheduled groups were generated." >&2; exit 1; }

echo "[4/6] Running ${GROUP_COUNT} scheduled grouped batch(es)"
GENERATION_STATUS_FILE="${SCHEDULE_DIR}/generation_status"
printf '0\n' > "$GENERATION_STATUS_FILE"
"$PYTHON_BIN" - "$SCHEDULE_MANIFEST" <<'PY' | while IFS=$'\t' read -r RUN_INDEX TASK_LABEL CKPT GROUP_JSONL GROUP_RESULT_JSONL GROUP_IMAGE_DIR; do
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
with manifest.open("r", encoding="utf-8") as handle:
    for raw_line in handle:
        line = raw_line.strip()
        if not line:
            continue
        row = json.loads(line)
        print(
            "\t".join(
                [
                    str(row["run_index"]),
                    row["task_label"],
                    row["ckpt"],
                    row["group_jsonl"],
                    row["group_result_jsonl"],
                    row["group_image_dir"],
                ]
            )
        )
PY
  mkdir -p "$(dirname "$GROUP_RESULT_JSONL")" "$GROUP_IMAGE_DIR"
  echo "  - [${RUN_INDEX}/${GROUP_COUNT}] ${TASK_LABEL} :: ${CKPT}"
  if ! "$PYTHON_BIN" run_workflows_jsonl.py \
      "$GROUP_JSONL" \
      --output "$GROUP_RESULT_JSONL" \
      --output-dir "$GROUP_IMAGE_DIR"; then
    printf '1\n' > "$GENERATION_STATUS_FILE"
  fi
done

GENERATION_ERRORS="$(<"$GENERATION_STATUS_FILE")"

echo "[5/6] Merging run results per task"
for TASK_RUN_ROOT in "${TASK_RUN_ROOTS[@]}"; do
  TASK_GROUP_RUN_RESULTS_DIR="${TASK_RUN_ROOT}/group_run_results"
  TASK_MERGED_RUN_RESULTS="${TASK_RUN_ROOT}/run_results.jsonl"
  : > "$TASK_MERGED_RUN_RESULTS"
  mapfile -t TASK_RESULT_FILES < <(find "$TASK_GROUP_RUN_RESULTS_DIR" -maxdepth 1 -type f -name '*__run_results.jsonl' | sort)
  [[ "${#TASK_RESULT_FILES[@]}" -gt 0 ]] || { echo "No group run results found in ${TASK_GROUP_RUN_RESULTS_DIR}" >&2; exit 1; }
  for RESULT_FILE in "${TASK_RESULT_FILES[@]}"; do
    cat "$RESULT_FILE" >> "$TASK_MERGED_RUN_RESULTS"
  done
done

echo "[6/6] Collecting receipts and sessions per task"
for TASK_RUN_ROOT in "${TASK_RUN_ROOTS[@]}"; do
  TASK_COMPILED_DIR="${TASK_RUN_ROOT}/compiled_run"
  TASK_COLLECTED_DIR="${TASK_RUN_ROOT}/collected"
  TASK_MERGED_RUN_RESULTS="${TASK_RUN_ROOT}/run_results.jsonl"
  echo "  - Collecting $(basename "$TASK_RUN_ROOT")"
  "$PYTHON_BIN" collect_receipts_and_sessions.py \
    --requests "${TASK_COMPILED_DIR}/requests.jsonl" \
    --run-results "$TASK_MERGED_RUN_RESULTS" \
    --output-dir "$TASK_COLLECTED_DIR"
done

echo
if [[ "$GENERATION_ERRORS" -eq 0 ]]; then
  echo "Scheduled cycle complete."
else
  echo "Scheduled cycle completed with generation errors." >&2
fi
echo "Run root: ${RUN_ROOT}"
echo "Schedule manifest: ${SCHEDULE_MANIFEST}"
echo "Per-task outputs:"
for TASK_RUN_ROOT in "${TASK_RUN_ROOTS[@]}"; do
  echo "  - ${TASK_RUN_ROOT}"
  echo "    Requests: ${TASK_RUN_ROOT}/compiled_run/requests.jsonl"
  echo "    Run results: ${TASK_RUN_ROOT}/run_results.jsonl"
  echo "    Receipts: ${TASK_RUN_ROOT}/collected/receipts.jsonl"
  echo "    Sessions: ${TASK_RUN_ROOT}/collected/sessions.jsonl"
done

if [[ "$GENERATION_ERRORS" -ne 0 ]]; then
  exit 1
fi
