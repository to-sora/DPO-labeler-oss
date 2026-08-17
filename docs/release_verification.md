# Release Verification

Verification date: 2026-08-17

## Publication Scope

This directory is prepared as an independent GitHub repository, not as a
subdirectory export of its development repository. Its public `main` branch is
initialized from the verified working tree as one root commit. Earlier
development branches, reflogs, generated runtime data, downloaded wildcard
packs, model files, local configuration, and credentials are not reachable from
the publication history.

Only the new repository's `main` branch should be pushed. Do not copy or push
the separately retained local Git-metadata backup.

## Automated Tests

All Python commands used the project virtual environment. The final source tree
passed:

| Suite | Result |
| --- | ---: |
| Core unit tests | 80 passed |
| Workflow generator tests | 41 passed |
| Labeler and export-viewer tests | 32 passed |
| Image-grader and adapter tests | 21 passed |
| Python total | 174 passed |
| Labeler frontend helper tests | 7 passed |

Additional gates passed:

- Python compilation for changed runtime and test modules.
- JavaScript syntax checks for frontend files.
- Shell syntax checks for executable scripts.
- Git whitespace validation, excluding intentional whitespace in data assets.
- Checkpoint registry coverage and publication validation.
- Installer consent, archive integrity, extraction safety, credential redaction,
  proxy scoping, idempotence, force behavior, and rollback tests.

The complete generator suite requires the explicitly accepted third-party
wildcard packs. Those installed files were present for testing and removed from
the final repository directory afterward.

## Wildcard Installer

`assets/wildcard_sources.json` pins seven upstream archives by version, size,
and SHA-256. The installer was tested with locally reconstructed source
archives and with a real authenticated Civitai route. It reproduced the
expected runtime layout, including deterministic aliases and curated outputs,
then passed byte-for-byte parity verification.

The installer enforces explicit acceptance, HTTPS, bounded retries, expected
sizes and hashes, ZIP expansion limits, path/symlink rejection, staging, final
tree hashes, and rollback. No shared proxy address or API token is stored in
the repository.

Direct Civitai access and a tested public proxy returned HTTP 403 in the
verification environment. A separately controlled local proxy succeeded when
its CA, query-token fallback, and direct-redirect behavior were explicitly
enabled. These environment-specific workarounds are optional and are not
configured by default.

## Real ComfyUI Generation

A clean standalone installation was tested against a local ComfyUI instance on
an NVIDIA GeForce RTX 4090:

- `template/tasks/quality_e2e_10x2.yaml`: 20 of 20 successful requests, ten
  images from each checkpoint configuration.
- A ten-session verification copy of `template/tasks/example_task.yaml`: 20 of
  20 successful requests, ten base and ten latent-upscaled images.
- Combined: 40 successful requests, 40 receipts, 20 paired sessions, and 40
  unique decodable PNG hashes.
- Requested dimensions matched every output.
- Every request resolved through a public, publishable checkpoint alias.
- Contact-sheet inspection found no blank or corrupted result.

The tracked example task remains one session; only the ignored verification
copy used ten sessions.

The multi-task scheduler was also exercised against a LAN ComfyUI host across
the bundled task inventory. One malformed local checkpoint produced a
repeatable tensor-shape error in both local and LAN contexts; this was a model
file incompatibility rather than an orchestration or OSS packaging defect.

## Browser Review Validation

The labeler, export viewer, and grader admin were exercised with headless Chrome
at desktop and constrained widths. Verified behavior includes:

- checkpoint aliases are displayed while raw checkpoint values remain
  available as titles and API data;
- selecting all matching sessions updates the visible count;
- all-score failure states show a terminal error and individual model errors;
- score cards do not render null dereference text, `[object Object]`, or
  `undefined` placeholders;
- top and bottom percentile buckets render numeric scores correctly.

Two accepted UI limitations remain documented rather than hidden:

- long session metadata can overflow horizontally around a 913 px viewport;
- review/admin payloads expose identity metadata and are not strict blind
  review.

## CPU Grader Validation

The optional CPU profile was installed once through the real setup helper on a
LAN client:

```text
Python:      3.13
Torch:       2.13.0+cpu
TorchVision: 0.28.0+cpu
CUDA:        unavailable, as expected
```

One browser-triggered inference run scored two images with
`waifu_scorer_v3` and `native` preprocessing:

```text
Scores:            6.570, 4.801
Structured errors: 0
Elapsed request:   5.295 seconds
Peak RSS:          approximately 1.62 GiB
Process swaps:     0
```

The model files matched their source SHA-256 hashes before inference. The test
created one run directory and two successful score rows; no inference retry was
performed. All adapter, browser, and tunnel processes were stopped afterward.

## Content And Size Gates

- Real `.env` files tracked or reachable: none.
- Private keys or strong credential patterns found: none.
- Host-specific home paths or LAN/public proxy addresses found: none.
- Model weights, images, databases, and generated output tracked: none.
- Downloaded third-party wildcard directories tracked: none.
- Tracked symlinks: none.
- Largest file and Git blob: 5,260,185 bytes.
- Files at or above 50 MiB: none.
- Git LFS required: no.
- Public branch root commits: one.
- Public branch commit count at preparation: one.

## Known Release Boundaries

- The HTTP applications are trusted-network tools, not production internet
  services.
- Grader model weights and their licenses remain the user's responsibility.
- Third-party wildcard packs are downloaded from original hosts and are not
  relicensed or redistributed here.
- Provenance for some tracked prompt/research text still requires repository
  owner confirmation; see `asset_provenance_audit.md`.
- This is a one-time source publication without a long-term maintenance
  commitment.

## Push Checklist

After creating an empty GitHub repository:

```bash
git status --short --branch
git rev-list --max-parents=0 HEAD
git rev-list --count HEAD
git remote add origin https://github.com/to-sora/DPO-labeler-oss.git
git push -u origin main
```

Before the push, confirm that status is clean, the root command prints one
commit, and the count command prints `1`. Push `main` only; do not use
`git push --all`.
