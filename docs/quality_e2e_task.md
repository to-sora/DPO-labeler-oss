# Quality End-to-End Task

## Purpose

`template/tasks/quality_e2e_10x2.yaml` is the tracked, real-generation task for
checking a standalone installation against ComfyUI. It creates 10 sessions with
two images per session: 10 images from each configuration and 20 images total.

This is a realistic randomized generation task. It is not a controlled
checkpoint-only benchmark because generation seeds and sample settings are
selected independently for each image.

## Required Local Assets

ComfyUI must provide the standard nodes used by the workflow plus the
`easy loraStack` and `easy loraStackApply` nodes. The following checkpoints must
be visible to ComfyUI under these exact names:

| Image configuration | Checkpoint filename |
| --- | --- |
| `image1` | `sdxl_novaAnimeXL_ilV150.safetensors` |
| `image2` | `sdxl_bluePencilXL_v700.safetensors` |

Checkpoint weights are not bundled and are not downloaded by `install.sh` or
`install_wildcard.sh`. Obtain model weights under their own terms, then either
use these filenames or update both `ckpt` fields in the task. The two names are
routed to the `sdxl_anime_base` prompt family by the compiler.

Both filenames are public, publishable aliases in `checkpoint_aliases.yaml`.
Use `skills/edit-dpo-task/` when changing model aliases, prompt families, seed
controls, or per-configuration image counts.

The task uses tracked `research_v8` prompt templates and SFW runtime wildcards.
The standalone setup still requires explicit acceptance before it downloads
the separate pinned third-party wildcard packs:

```bash
bash install.sh --non-interactive --accept-third-party-terms
```

That acceptance applies only to the wildcard sources listed by
`bash install_wildcard.sh --list-sources`; it is not acceptance of model terms.

## Configuration

Both image configurations use `SdxlEaseLoraWorkflow` with LoRA disabled. At
runtime, all ten inherited LoRA filename slots are normalized to `None`, so the
task does not depend on LoRA files named inside the bundled workflow JSON.

The task retains the verified V8 quality and prompt behavior:

- SFW family-routed wildcard template with a session-seeded base prompt.
- Random segment dropout with pair chance `0.3`, segment probability `0.1`, and
  `image_index_seed` control.
- Generation seed controlled by `image_index_seed`.
- Steps selected uniformly from `30`, `35`, and `40`.
- CFG selected uniformly from `7.0` and `7.5`.
- Width and height selected independently from `768`, `1024`, `1280`, and
  `1536`, with weights `3`, `3`, `3`, and `2`.
- Fixed global seed `20260422` and `session_count: 10`.

Because dropout and sampling use `image_index_seed`, paired images can have
different prompt segments, seeds, dimensions, steps, and CFG. For a controlled
model comparison, change those seed controls to `session_seed` in a separate
task.

## Run

With ComfyUI running on the configured endpoint and the project virtual
environment active:

```bash
./run_full_cycle.sh \
  --task-yaml template/tasks/quality_e2e_10x2.yaml \
  --global-seed 20260422 \
  --python "$VIRTUAL_ENV/bin/python"
```

A successful run contains:

- 20 compiled requests and 20 successful run-result rows;
- 20 receipts, split as 10 `image1` and 10 `image2` rows;
- 10 session rows, each containing exactly two images;
- 20 PNG files whose dimensions match their compiled requests.

Generated images, receipts, sessions, and local configuration remain ignored
and are not part of the publication.

## Verified Run

On 2026-08-15, a clean standalone checkout was installed with the real prompt
and wildcard setup and run against local ComfyUI on an NVIDIA GeForce RTX 4090.
All 20 requests succeeded. The 20 PNGs decoded successfully, had unique SHA-256
hashes, matched requested dimensions, and passed a full contact-sheet visual
inspection with no blank or corrupted output. Every compiled request carried
the expected public, publishable registry alias and keyword-derived
`sdxl_anime_base` family; no default-family fallback was used.

The same clean clone also ran a 10-session verification copy of
`template/tasks/example_task.yaml`, producing another 20 successful images.
The combined release verification therefore covered 40 of 40 successful real
generations, with 10 images from every image configuration. See
`docs/release_verification.md` for the cross-task evidence.
