# Operations

## Pipeline

The generation path has four explicit stages:

1. `compile_yaml_to_requests_jsonl.py` expands one task YAML into deterministic
   request rows and manifests.
2. `split_jsonl_by_shared_config.py` groups requests by checkpoint and LoRA
   configuration.
3. `run_workflows_jsonl.py` submits groups to ComfyUI with resume support.
4. `collect_receipts_and_sessions.py` joins successful results into receipt and
   paired-session JSONL files.

`run_full_cycle.sh` executes all four stages for one task and returns nonzero if
any generation request fails.

## Task Inventory

Two stable task entry points are stored in `template/tasks/`:

- `example_task.yaml`: one session and two image configurations for a quick
  smoke run.
- `quality_e2e_10x2.yaml`: ten sessions and two image configurations, producing
  20 images when all requests succeed.

The release also contains 126 generated experiment YAMLs under
`template/workflows/`. They cover batch V1.7/V1.8 and the DPO workflow families
through V8. These are generated assets; regenerate them with the matching
script under `generators/` rather than editing them manually.

Checkpoint files referenced by a task must exist in ComfyUI. The repository
contains identifiers and classification metadata, not model weights.

## Single Task

```bash
./run_full_cycle.sh \
  --task-yaml template/tasks/example_task.yaml \
  --global-seed 2025 \
  --output-root output/batch \
  --python "$VIRTUAL_ENV/bin/python"
```

The output path is derived from task scope, task name, and seed. Repeating the
same command resumes successful rows and retries only unfinished requests.

## Multiple Tasks

Use the cross-task scheduler instead of launching many full-cycle processes:

```bash
./run_multi_task_scheduled_cycle.sh \
  --task-yaml template/tasks/example_task.yaml \
  --task-yaml template/tasks/quality_e2e_10x2.yaml \
  --global-seed 2025 \
  --output-root output/batch_schedule \
  --python "$VIRTUAL_ENV/bin/python"
```

Repeat `--task-yaml` for every task to run. The scheduler compiles all tasks,
keeps groups with the same checkpoint/LoRA configuration adjacent, runs one
group at a time, and collects each task independently. This limits unnecessary
model switching while preserving task-level receipts and sessions.

The scheduler does not impose a time limit. ComfyUI availability, task size,
image dimensions, sampling steps, and model-switch cost determine runtime.

## Direct Stage Commands

Compile:

```bash
python compile_yaml_to_requests_jsonl.py \
  --task-yaml template/tasks/example_task.yaml \
  --output-dir compiled_run
```

Split:

```bash
python split_jsonl_by_shared_config.py \
  --input compiled_run/requests.jsonl \
  --output-dir grouped_requests
```

Run one group:

```bash
python run_workflows_jsonl.py \
  grouped_requests/0001__group.jsonl \
  --output output/run_results.jsonl \
  --output-dir output/images \
  --url 127.0.0.1 \
  --port 8188
```

Collect:

```bash
python collect_receipts_and_sessions.py \
  --requests compiled_run/requests.jsonl \
  --run-results output/run_results.jsonl \
  --output-dir collected
```

## Review Endpoints

`start_all_endpoints.sh` reads `.env.oss`, starts enabled services, monitors
their process groups, and writes logs below `RUNTIME_ROOT`.

Default URLs in local mode:

```text
http://127.0.0.1:8787/  DPO labeler
http://127.0.0.1:8084/  export viewer
http://127.0.0.1:8790/  image-grader API, disabled by default
http://127.0.0.1:8087/  grader admin/playground
```

Stop the launcher with Ctrl-C. It sends termination to every child process
group, waits briefly, then force-stops only services that did not exit.

For Tailscale access, select `BIND_MODE=tailscale` during setup and allow only
the required ports in the host firewall. Do not expose the services directly
to the public internet.

## Data And State

The following paths are runtime data and remain ignored:

```text
compiled_run/
grouped_requests/
output/
collected/
images/
real_outputs/
temp/
models/
```

Back up label events, exports, score databases, and generated images separately
when they matter. The launcher does not provide retention, replication, or
remote backup.

Review and grader pages can display task, dataset, session, seed, workflow,
checkpoint, and run metadata. Checkpoint aliases shorten display values but do
not remove underlying identifiers from API payloads. The current release is
therefore unsuitable for strict blind review without additional filtering.

## Checkpoint Changes

Do not rename a checkpoint only in a task file. Update the registry or an
ignored local overlay so prompt-family routing remains deterministic:

```text
checkpoint_aliases.yaml
checkpoint_aliases.local.yaml
```

Validate a publication-safe task:

```bash
python skills/edit-dpo-task/scripts/validate_task.py \
  template/tasks/quality_e2e_10x2.yaml \
  --expected-per-image 10 \
  --publication
```

Publication validation rejects unknown aliases, fallback-only classification,
and models not marked both public and publishable.
