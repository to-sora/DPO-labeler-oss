# Installation

## Fresh Checkout

Create an isolated Python environment from the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.13 was used for release verification. Python 3.11 or newer is the
supported target for this source release.

## Setup Modes

`install.sh` delegates to `scripts/oss_setup.py`. The interactive path is the
recommended first installation:

```bash
bash install.sh
```

It performs these operations in order:

1. Lists the seven optional third-party wildcard sources and requires `AGREE`
   before any archive request.
2. Configures the ComfyUI URL, bind mode, runtime paths, endpoint enablement,
   ports, invite token, and grader paths.
3. Optionally installs CPU or CUDA 13 grader dependencies into the selected
   grader Python environment.
4. Writes `.env`, `.env.oss`, and `image_grader/config.local.json`.

All three generated configuration files are ignored by Git.

Configure endpoints without optional wildcard data:

```bash
bash install.sh --non-interactive --skip-wildcards
```

Install wildcard data and CPU grader dependencies non-interactively:

```bash
bash install.sh \
  --non-interactive \
  --accept-third-party-terms \
  --install-grader-deps cpu
```

Non-interactive mode requires `--accept-third-party-terms` whenever wildcard
installation is enabled. Passing the flag means the person running the command
reviewed the source inventory and chose to accept the upstream terms.

Available grader profiles:

| Profile | Installed requirements | Intended device |
| --- | --- | --- |
| `none` | none | Grader disabled or managed separately |
| `cpu` | base grader stack plus CPU Torch/TorchVision | `cpu` |
| `cuda13` | base grader stack plus CUDA 13 Torch/TorchVision | `cuda` |

The dependency profile selects packages; the `device` field in
`image_grader/config.local.json` selects inference placement. Confirm that a
CUDA installation and driver are compatible before changing the device from
`cpu` to `cuda`.

## Generated Configuration

`.env` controls the ComfyUI runner. Its important values are:

```dotenv
RUN_WORKFLOWS_URL=http://127.0.0.1:8188/
RUN_WORKFLOWS_ALLOW_INSECURE=true
RUN_WORKFLOWS_TIMEOUT_SECONDS=300
```

Setup sets `RUN_WORKFLOWS_ALLOW_INSECURE=true` for any `http://` ComfyUI URL.
Use plain HTTP only on loopback or an explicitly trusted private network; use
HTTPS for other remote endpoints.

`.env.oss` controls service startup. The generated defaults use:

```text
BIND_MODE=local
DATASET_ROOT=output
IMAGE_ROOT=output
RUNTIME_ROOT=temp/oss_endpoints
LABELER_PORT=8787
EXPORT_VIEWER_PORT=8084
IMAGE_GRADER_PORT=8790
ADAPTER_PORT=8087
```

Bind modes are:

- `local`: `127.0.0.1`, recommended.
- `tailscale`: the first detected Tailscale IPv4 address.
- `all`: `0.0.0.0`; protect it with host firewall rules.
- An explicit address entered during setup.

## Grader Models

Setup creates a local config but never downloads model weights. With the
default `models_root` of `models/image_eval`, the waifu scorer uses:

```text
models/image_eval/
|-- Eugeoter/waifu-scorer-v3/model.safetensors
`-- openai/clip-vit-large-patch14/
    |-- config.json
    |-- preprocessor_config.json
    `-- model.safetensors
```

The scorer head is small, but the CLIP backbone makes this bundle approximately
1.72 GB. The other configured graders require their complete Hugging Face-style
directories:

```text
models/image_eval/shadowlilac/aesthetic-shadow/
models/image_eval/minizhu/aesthetic-anime-v2/
```

Only select enabled model IDs whose files are present. Model references and
license findings are documented in `asset_provenance_audit.md`.

Validate a CPU runtime without running inference:

```bash
python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())'
```

Expected CPU output ends with `False` for CUDA availability.

## Wildcard Sources And Proxies

Inspect the pinned source inventory first:

```bash
bash install_wildcard.sh --list-sources
```

The installer enforces expected archive sizes and SHA-256 hashes, rejects path
traversal and symlink entries, limits extraction expansion, stages changes, and
rolls back a failed replacement. Existing differing files are preserved unless
`--force` is supplied.

Supported environment variables:

```text
CIVITAI_API_TOKEN
CIVITAI_PROXY
CIVITAI_CA_BUNDLE
CIVITAI_WAIT_SECONDS
```

The proxy is scoped to Civitai requests. Credentials are sent on the first
request and removed before redirects. For a controlled TLS-inspecting proxy:

```bash
CIVITAI_PROXY=http://proxy.example:8080 \
CIVITAI_CA_BUNDLE=/path/to/proxy-ca.crt \
bash install_wildcard.sh \
  --accept-third-party-terms \
  --allow-legacy-proxy-ca \
  --civitai-token-query \
  --civitai-direct-redirects \
  --civitai-wait-seconds 2
```

`--civitai-token-query` can expose the token to trusted proxy logs on the first
URL. `--allow-legacy-proxy-ca` weakens certificate validation for that explicit
CA. Use neither option with an untrusted proxy.

Verify an installed tree without downloading:

```bash
bash install_wildcard.sh --verify-only
```

## First Start

Start every endpoint enabled in `.env.oss`:

```bash
bash start_all_endpoints.sh
```

The script prints accessible URLs and log locations. Press Ctrl-C once to stop
all child services. If one service exits during startup or operation, the
launcher exits nonzero and stops the remaining services.

## Common Failures

`No module named 'torch'`:

```bash
bash install.sh --non-interactive --skip-wildcards --install-grader-deps cpu
```

Run that command with the same Python environment configured as
`GRADER_PYTHON_BIN`.

`Model path does not exist` means dependencies are installed but the selected
weights are absent. Place the model under the configured root or disable that
model ID.

`Could not detect a Tailscale IPv4 address` means `BIND_MODE=tailscale` was
selected without a running Tailscale client. Re-run setup and select `local`, or
start Tailscale first.

A service log containing `BrokenPipeError` immediately after cancelling an
image load is normally an abandoned browser response. It does not alter image,
label, or score data.
