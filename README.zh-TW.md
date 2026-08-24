# Comfy DPO Workflow Toolkit

[English](README.md) | **繁體中文** | [简体中文](README.zh-CN.md)

這是一套獨立的 DPO 工作流程工具，可編譯 prompt 任務、排程 ComfyUI 工作流程、收集成對影像 session、進行 DPO 偏好標註，以及選擇性地使用 image grader 評分生成影像。

本 repository 是一次性的 source release，預期用於本機或受信任的私人網路，不適合直接暴露於公開 Internet。Repository 不包含模型權重、生成圖片、review state、score database、virtual environment 或本機憑證。

## 主要內容

- JSONL compile、split、run、schedule 與 receipt/session collection 工具。
- SDXL EASE LoRA workflow adapter 與 ComfyUI API template。
- 可直接執行的 task 範例與實驗 YAML。
- 瀏覽器 DPO labeler 與 export viewer。
- 選用的 image-grader API 與 grader admin/playground。
- Checkpoint alias、prompt family、visibility 與 publication registry。
- 有 provenance 邊界說明的 prompt / workflow / task / research assets。
- 七組第三方 wildcard pack 的 opt-in installer。
- Unit、generator、frontend、review-tool 與 grader tests。

## 環境需求

- Linux 或具備 Bash 與標準 Unix process tools 的環境。
- Python 3.11 以上；此 release 曾以 Python 3.13 驗證。
- 可連線的 ComfyUI API，通常為 `http://127.0.0.1:8188/`。
- 所選 task 所需的 ComfyUI checkpoints 與 custom nodes。
- 只有執行 frontend helper tests 時才需要 Node.js。
- 若使用 CUDA grader，需要相容的 NVIDIA driver/runtime。

## 快速開始

```bash
git clone https://github.com/to-sora/DPO-labeler-oss.git
cd DPO-labeler-oss
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
bash install.sh
```

只設定 endpoints、不下載第三方 wildcard：

```bash
bash install.sh --non-interactive --skip-wildcards
```

明確接受 wildcard 條款並安裝 CPU grader：

```bash
bash install.sh \
  --non-interactive \
  --accept-third-party-terms \
  --install-grader-deps cpu
```

Setup 會產生被 Git 忽略的 `.env`、`.env.oss` 與 `image_grader/config.local.json`，不會自動安裝模型權重。

完整安裝方式請看 [Installation](docs/installation.md)。

## 生成圖片

先啟動 ComfyUI，再執行：

```bash
./run_full_cycle.sh \
  --task-yaml template/tasks/example_task.yaml \
  --global-seed 2025 \
  --python "$VIRTUAL_ENV/bin/python"
```

品質驗證 task：

```bash
./run_full_cycle.sh \
  --task-yaml template/tasks/quality_e2e_10x2.yaml \
  --global-seed 20260422 \
  --python "$VIRTUAL_ENV/bin/python"
```

多 task model-aware scheduler：

```bash
./run_multi_task_scheduled_cycle.sh \
  --task-yaml template/tasks/example_task.yaml \
  --task-yaml template/tasks/quality_e2e_10x2.yaml \
  --global-seed 2025 \
  --python "$VIRTUAL_ENV/bin/python"
```

更多操作方式請看 [Operations](docs/operations.md)。

## 啟動 Review Services

```bash
bash start_all_endpoints.sh
```

預設服務：

| 服務 | 預設 | Port |
| --- | --- | ---: |
| DPO labeler | 啟用 | `8787` |
| Export viewer | 啟用 | `8084` |
| Image-grader API | 停用 | `8790` |
| Grader admin/playground | 啟用 | `8087` |

預設只 bind localhost；也可以在互動安裝時選擇 Tailscale address 或其他介面。請只在本機或受信任的私人網路使用。

## Image Grader

Grader profile：

- `none`：不安裝 grader stack。
- `cpu`：安裝 CPU-only Torch / TorchVision 與 grader dependencies。
- `cuda13`：安裝 CUDA 13 Torch / TorchVision 與 grader dependencies。

### `allowed_roots` 檔案系統安全設定

啟用 image-grader HTTP API 時，務必設定 `server.allowed_roots`，明確指定 grader 可以讀取的資料目錄。

**不要把 `allowed_roots` 留成空陣列。** 目前 release 中，`allowed_roots: []` 是寬鬆 / fail-open 設定，因此不能把空陣列當成安全邊界。

執行 `bash install.sh` 時，setup 會把你設定的 dataset root 寫入 `image_grader/config.local.json` 的 `server.allowed_roots`。如果你手動建立或修改 grader config，請使用明確且盡可能狹窄的絕對路徑，例如：

```json
{
  "server": {
    "allowed_roots": ["/absolute/path/to/your/dataset"]
  }
}
```

不要為了方便把 `/` 或整個 home directory 設為 allowed root。

更多 grader 設定請看 [Image Grader](docs/image_grader.md)。

## Labeler 路徑安全

Labeler 只接受解析後仍位於設定 `dataset_root` 內的 `saved_path`。Absolute path、`..` 路徑或 symlink 若最終解析到 dataset root 外，該 session 會被 catalog 拒絕，不會拿來讀取圖片。

## Label event idempotency

重送完全相同的 `event_id` 與相同 label event 內容仍是 idempotent：系統會回傳原事件，不會重複寫入。

若同一 `event_id` 被拿來提交不同 pair、decision、defects、note、reviewer/client context 或其他 label event 內容，系統會明確拒絕，而不會靜默把它當成成功重送。

## Wildcard Downloads

七組第三方 wildcard pack 不直接存放在 Git。來源版本、預期大小、SHA-256、轉換與 provenance 記錄在 `assets/wildcard_sources.json`。

```bash
bash install_wildcard.sh --list-sources
bash install_wildcard.sh --accept-third-party-terms
```

如需 Civitai authentication，請在 process environment 設定 `CIVITAI_API_TOKEN`。請先閱讀 [Asset Provenance](docs/asset_provenance_audit.md) 與 [Third-Party Notices](THIRD_PARTY_NOTICES.md)。

## 測試

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

完整 release verification 請看 [Release Verification](docs/release_verification.md)。

## 安全限制

- 所有 HTTP services 應保持在 localhost 或受信任 VPN；這些服務不是 hardened public web services，也不提供 TLS termination。
- Review / grader UI 可能顯示 dataset、task、session、checkpoint、seed 與 run metadata。
- Aesthetic grader 不能可靠偵測所有手指、手部、解剖或其他生成缺陷。
- 第三方 wildcard 與 model license 不受本 repository 的 MIT license 覆蓋。
- 這是一次性 release，不承諾長期維護或 hosted support。

在暴露任何 endpoint 前，請閱讀 [Security](SECURITY.md)。
