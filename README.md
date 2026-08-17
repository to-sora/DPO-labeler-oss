# Comfy DPO Workflow Toolkit

A standalone toolkit for compiling prompt tasks, scheduling ComfyUI workflows,
collecting paired-image sessions, reviewing DPO preferences, and optionally
grading generated images.

This repository is a one-time source release. It is intended for local machines
and trusted private networks, not direct internet exposure. Model weights,
generated images, review state, score databases, virtual environments, and
local credentials are not included.

## Included

- JSONL compile, split, run, schedule, and receipt/session collection tools.
- SDXL EASE LoRA workflow adapters and ComfyUI API templates.
- Two directly runnable task examples and 126 generated experiment task YAMLs.
- Browser-based DPO labeler and export viewer.
- Optional image-grader API and grader admin/playground.
- Checkpoint alias, prompt-family, visibility, and publication registry.
- Tracked prompt, workflow, task, and research assets with documented
  provenance boundaries.
- Opt-in installer for seven pinned third-party wildcard packs.
- Deterministic unit, generator, frontend, review-tool, and grader tests.

## Prerequisites

- Linux or another environment with Bash and standard Unix process tools.
- Python 3.11 or newer. The release was verified with Python 3.13.
- A reachable ComfyUI API, normally `http://127.0.0.1:8188/`.
- ComfyUI checkpoints and custom nodes required by the selected task.
- Node.js only when running the frontend helper tests.
- Optional NVIDIA driver/runtime compatible with the CUDA grader profile.

ComfyUI, model weights, and LoRA files are separate installations governed by
their own licenses.

## Quick Start

```bash
git clone https://github.com/to-sora/DPO-labeler-oss.git
cd DPO-labeler-oss
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
bash install.sh
```

Interactive setup lists every optional wildcard source, requires the exact
response `AGREE` before downloading it, configures endpoint paths and ports,
and offers CPU, CUDA 13, or no grader dependency installation.

Endpoint-only setup without third-party wildcard downloads:

```bash
bash install.sh --non-interactive --skip-wildcards
```

Non-interactive setup with explicit wildcard acceptance and CPU grading:

```bash
bash install.sh \
  --non-interactive \
  --accept-third-party-terms \
  --install-grader-deps cpu
```

The setup writes ignored local files: `.env`, `.env.oss`, and
`image_grader/config.local.json`. It never installs model weights.

See [Installation](docs/installation.md) for all setup modes, grader profiles,
model paths, proxy handling, and troubleshooting.

## Generate Images

Start ComfyUI first. The lightweight task uses one session with two image
configurations:

```bash
./run_full_cycle.sh \
  --task-yaml template/tasks/example_task.yaml \
  --global-seed 2025 \
  --python "$VIRTUAL_ENV/bin/python"
```

The validated quality task generates 10 images from each of two checkpoint
configurations:

```bash
./run_full_cycle.sh \
  --task-yaml template/tasks/quality_e2e_10x2.yaml \
  --global-seed 20260422 \
  --python "$VIRTUAL_ENV/bin/python"
```

Run several task files through the global model-aware scheduler:

```bash
./run_multi_task_scheduled_cycle.sh \
  --task-yaml template/tasks/example_task.yaml \
  --task-yaml template/tasks/quality_e2e_10x2.yaml \
  --global-seed 2025 \
  --python "$VIRTUAL_ENV/bin/python"
```

The scheduler groups matching checkpoint and LoRA configurations across tasks
to reduce backend model switching. It resumes existing successful requests and
returns a nonzero status if generation fails.

See [Operations](docs/operations.md) and
[Quality End-to-End Task](docs/quality_e2e_task.md).

## Start Review Services

After setup:

```bash
bash start_all_endpoints.sh
```

Default endpoints:

| Service | Default | Port |
| --- | --- | ---: |
| DPO labeler | enabled | `8787` |
| Export viewer | enabled | `8084` |
| Image-grader API | disabled | `8790` |
| Grader admin/playground | enabled | `8087` |

The launcher binds to localhost by default. Interactive setup can instead bind
to a detected Tailscale address or all interfaces. It supervises every enabled
service and stops all child process groups on Ctrl-C or when one service fails.

## Image Grading

The grader dependency profiles are:

- `none`: do not install the optional grader stack.
- `cpu`: install base grader dependencies plus CPU-only Torch and TorchVision.
- `cuda13`: install base grader dependencies plus CUDA 13 Torch and TorchVision.

Only enable model IDs whose files exist below the configured model root. The
grader does not download weights. The admin playground supports direct keys of
the form:

```text
score["native"]["waifu_scorer_v3"]
```

It also supports restricted arithmetic, averages, minima/maxima, and
percentiles for top/bottom ranking. It does not execute arbitrary Python.

See [Image Grader](docs/image_grader.md) for model layouts, CPU/GPU setup,
formula syntax, examples, and model limitations.

## Checkpoint Registry

`checkpoint_aliases.yaml` maps filenames to stable model IDs and prompt
families. It also records whether an alias is public and publishable. Historical
private identifiers remain marked `visibility: private` and `publish: false` so
bundled experiment tasks can still be classified; no private weights are
included.

Put machine-specific or new private aliases in the ignored
`checkpoint_aliases.local.yaml` overlay. See
[Checkpoint Registry](docs/checkpoint_registry.md).

## Wildcard Downloads

Seven third-party wildcard packs are deliberately absent from Git. Their source
versions, expected sizes, SHA-256 hashes, transformations, and provenance status
are recorded in `assets/wildcard_sources.json`.

List sources without downloading:

```bash
bash install_wildcard.sh --list-sources
```

Install only after reviewing and accepting their terms:

```bash
bash install_wildcard.sh --accept-third-party-terms
```

Set `CIVITAI_API_TOKEN` in the process environment if authentication is
required. No token or shared proxy is embedded in this repository. Read
[Asset Provenance](docs/asset_provenance_audit.md) and
[Third-Party Notices](THIRD_PARTY_NOTICES.md) before redistribution.

## Test

Install the accepted wildcard packs before running the complete generator
suite. From the repository root:

```bash
python -m unittest discover -s tests/unit -t . -p 'test_*.py'
python -m unittest discover -s tests/generators -t . -p 'test_*.py'
python -m unittest \
  dpo_labeler/backend/test_labeler_app.py \
  dpo_labeler/backend/test_server.py \
  dpo_labeler/export_viewer/test_app.py \
  dpo_labeler/export_viewer/test_server.py
PYTHONPATH="$PWD/image_grader:$PWD" \
  python -m unittest discover -s image_grader/tests -p 'test_*.py'
node --test tests/frontend/test_dpo_labeler_frontend_helpers.mjs
```

The publication tree passed 174 Python tests and seven JavaScript tests. See
[Release Verification](docs/release_verification.md) for live ComfyUI, wildcard
installer, browser UI, and CPU grader evidence.

## Security And Known Limits

- Keep all HTTP services on localhost or a trusted VPN. They are not hardened
  public web services and do not provide TLS termination.
- Review and grader interfaces can expose dataset, task, session, checkpoint,
  seed, and run metadata. Do not treat the current UI as strict blind review.
- Long checkpoint metadata can still cause horizontal overflow near a 913 px
  viewport despite alias display.
- A browser cancelling an image request can produce a `BrokenPipeError` log;
  this affects only the abandoned HTTP response, not stored data.
- Aesthetic graders are not reliable detectors for malformed hands, fingers,
  anatomy, or every generation defect.
- Optional third-party wildcard and model licenses are not covered by the
  repository MIT license.
- This is a one-time release with no promise of long-term maintenance or hosted
  support.

Read [Security](SECURITY.md) before exposing any endpoint.
