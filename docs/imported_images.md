# Imported image tasks

Imported tasks compare existing images across named dimensions. A server operator
loads YAML manifests at startup; reviewers sign in, select a task, and choose A or
B for each open dimension. There is no browser upload/paste control or HTTP import
API. `POST /api/v1/imported/tasks` returns 405, including for signed-in users.

## Start the server

Use Python 3.11 or newer with the `venv`/`ensurepip` modules available. Linux
distributions may package these separately as `python3-venv`.
From the repository root, use an existing environment with `requirements.txt`
installed, or run `bash dpo_labeler/install-local-app-build.sh`. The installer uses
`python3`; set `LABELER_PYTHON` to another Python executable if needed.

```bash
dpo_labeler/start.sh \
  --dataset-root /srv/labeler/datasets \
  --image-root /srv/labeler/images \
  --image-root /srv/labeler/prompts \
  --state-dir /srv/labeler/state \
  --import-yaml /srv/labeler/tasks/morgan.yaml
```

Repeat `--import-yaml PATH` for additional manifests. Paths name files on the
server, and relative manifest paths are resolved from the server's working
directory. All selected files are processed before the server accepts requests.
Invalid manifests stop startup with an error. No manifest directory is watched,
and there is no hot reload.

The convenience launcher defaults to HTTP on `0.0.0.0:8083` with invite token
`change-me`. Override these with `--host`, `--port`, and `--invite-token`.
The same import flag is available on `python -m dpo_labeler.backend.server`;
that entry point requires explicit dataset, state, and invite-token arguments.
`--stop` stops launcher-owned services; `--force` frees the chosen port.

An unchanged manifest at the same resolved path reuses its saved task, including
review progress, on subsequent starts. Changing its YAML contents or using a new
manifest path creates a separate task. Existing saved tasks remain available
when an import argument is removed. Previously imported tasks also remain
available without any new import arguments.

## Manifest schema and example

[Download the example YAML](../dpo_labeler/examples/imported.yaml).
Each task must contain **1–1,000 images**. A larger list is rejected before image
files are opened or ranking state is allocated.

```yaml
task-name: Morgan portrait preferences
character-name: Morgan
image_dir: /srv/labeler/images/morgan
prompt_dir: /srv/labeler/prompts/morgan  # Optional; defaults to image_dir.
dim-name:
  - Overall quality
  - sexy
  - art
  - similarity
image:
  - image-path: portrait-a.png
    prompt-path: portrait-a.txt
    prompt: Portrait beside a window
  - image-path: portraits/portrait-b.jpg
    prompt: Portrait in a garden
```

| Field | Required | Meaning |
| --- | --- | --- |
| `task-name` | Yes | Nonempty task label. |
| `character-name` | Yes | Nonempty character or collection label. |
| `image_dir` | Yes | Existing absolute image directory on the server. |
| `prompt_dir` | No | Existing absolute prompt directory; defaults to `image_dir`. |
| `dim-name` | Yes | Nonempty list of unique, nonempty dimension names. |
| `image` | Yes | List of 1–1,000 image mappings. |
| `image[].image-path` | Yes | Relative path to a readable image inside `image_dir`. Resolved image paths must be unique. |
| `image[].prompt-path` | No | Relative to `prompt_dir`, or absolute inside it. Existing files are read as UTF-8. |
| `image[].prompt` | No | Inline prompt string; defaults to an empty string. |

Both directories must resolve inside `--dataset-root` or one of the repeatable
`--image-root` directories. Image paths and prompt paths are checked after
resolving symlinks; neither may escape its declared directory. An existing
prompt file overrides `prompt`. A missing prompt file uses the inline fallback.

Prompt text is saved when the task is first imported. Images are referenced,
not copied. Keep source images and symlinks stable: imported source paths are
validated at import, and are not periodically revalidated. Changing a source
file alone does not create a new task or refresh its saved prompts. To load
revised source files, change the manifest or its path and restart the server.

## Review progress and exports

Reviewer identity comes from the authenticated session. Use a distinct username
for each reviewer; sessions using the same username share that reviewer's
progress. Each reviewer has independent pending pairs, comparisons, exposure
counts, rankings, and retry records. Two reviewers may save opposite choices on
the same image pair without invalidating one another's work. Browser retry
storage is also scoped by username and task.

Tasks saved by the older shared-progress implementation keep their previous
comparisons and ranking state as the starting point for every reviewer. That
shared history is preserved. Subsequent choices and rankings are independent.

Choose A or B for every open dimension. Preferences implied by that reviewer's
saved ranking are filled in automatically. Prompts open with each pair. Download
one dimension or all dimensions at any time; incomplete rankings are estimates.
See [the export schema](imported_exports.md) for label events, DPO pairs, and
rankings.

Each task's cache is `--state-dir/imported/<task-id>/task.sqlite3`. Reviewer state
and comparison events are saved transactionally in that database. Exports are
stored under `exports/<reviewer-hash>/` within the task directory. Removing the
task directory removes its cached progress; an enabled startup manifest will
recreate the task on the next start.

## Verification

```bash
.venv/bin/python -m unittest discover -s dpo_labeler -t .
node --test tests/frontend/test_dpo_labeler_frontend_helpers.mjs
```

Optional browser checks require Firefox and geckodriver on `PATH`. Alternatively,
set `FIREFOX_BINARY` and `GECKODRIVER` to their executable paths. Install the Python
browser dependencies and run:

```bash
bash install-local-build.sh
.local-tool/uat/bin/python -m dpo_labeler.uat.run \
  --output-dir output/imported-uat
```

The runner creates temporary images, a temporary database, and a loopback server
on an automatically assigned port. It checks two reviewer sessions, response-loss
recovery, ranking/export behavior, startup-only imports, mobile layout, standard
review, and restart persistence. It does not connect to ComfyUI or use existing
review state.

## 中文摘要

匯入由伺服器啟動時的 `--import-yaml` 執行，可重複指定；網頁不提供匯入，也不會熱載入。
每份 YAML 限 1–1,000 張圖片。`prompt_dir` 預設為 `image_dir`，兩者均須位於設定的允許目錄內。
相同路徑及內容會沿用已儲存的任務。評分者名稱取自登入工作階段，各名稱的待比較圖片及排名分開保存；
舊版共同評分會保留作為起點。上方表格及 YAML 範例列出完整匯入格式。
