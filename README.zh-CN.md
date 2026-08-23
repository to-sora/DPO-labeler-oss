# Comfy DPO Workflow Toolkit

[English](README.md) | [繁體中文](README.zh-TW.md) | **简体中文**

这是一套独立的 DPO 工作流工具，可编译 prompt 任务、调度 ComfyUI 工作流、收集成对图像 session、进行 DPO 偏好标注，并可选择使用 image grader 对生成图像评分。

本 repository 是一次性的 source release，预期用于本机或可信的私有网络，不适合直接暴露到公开 Internet。Repository 不包含模型权重、生成图片、review state、score database、virtual environment 或本地凭证。

## 主要内容

- JSONL compile、split、run、schedule 与 receipt/session collection 工具。
- SDXL EASE LoRA workflow adapter 与 ComfyUI API template。
- 可直接运行的 task 示例与实验 YAML。
- 浏览器 DPO labeler 与 export viewer。
- 可选的 image-grader API 与 grader admin/playground。
- Checkpoint alias、prompt family、visibility 与 publication registry。
- 带 provenance 边界说明的 prompt / workflow / task / research assets。
- 七组第三方 wildcard pack 的 opt-in installer。
- Unit、generator、frontend、review-tool 与 grader tests。

## 环境要求

- Linux 或具备 Bash 与标准 Unix process tools 的环境。
- Python 3.11 或更高版本；此 release 曾使用 Python 3.13 验证。
- 可访问的 ComfyUI API，通常为 `http://127.0.0.1:8188/`。
- 所选 task 所需的 ComfyUI checkpoints 与 custom nodes。
- 仅在运行 frontend helper tests 时需要 Node.js。
- 如果使用 CUDA grader，需要兼容的 NVIDIA driver/runtime。

## 快速开始

```bash
git clone https://github.com/to-sora/DPO-labeler-oss.git
cd DPO-labeler-oss
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
bash install.sh
```

只配置 endpoints、不下载第三方 wildcard：

```bash
bash install.sh --non-interactive --skip-wildcards
```

明确接受 wildcard 条款并安装 CPU grader：

```bash
bash install.sh \
  --non-interactive \
  --accept-third-party-terms \
  --install-grader-deps cpu
```

Setup 会生成被 Git 忽略的 `.env`、`.env.oss` 和 `image_grader/config.local.json`，不会自动安装模型权重。

完整安装方式请查看 [Installation](docs/installation.md)。

## 生成图片

先启动 ComfyUI，再运行：

```bash
./run_full_cycle.sh \
  --task-yaml template/tasks/example_task.yaml \
  --global-seed 2025 \
  --python "$VIRTUAL_ENV/bin/python"
```

质量验证 task：

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

更多操作方式请查看 [Operations](docs/operations.md)。

## 启动 Review Services

```bash
bash start_all_endpoints.sh
```

默认服务：

| 服务 | 默认 | Port |
| --- | --- | ---: |
| DPO labeler | 启用 | `8787` |
| Export viewer | 启用 | `8084` |
| Image-grader API | 禁用 | `8790` |
| Grader admin/playground | 启用 | `8087` |

默认只 bind localhost；也可以在交互安装时选择 Tailscale address 或其他接口。请只在本机或可信私有网络使用。

## Image Grader

Grader profile：

- `none`：不安装 grader stack。
- `cpu`：安装 CPU-only Torch / TorchVision 与 grader dependencies。
- `cuda13`：安装 CUDA 13 Torch / TorchVision 与 grader dependencies。

### `allowed_roots` 文件系统安全设置

启用 image-grader HTTP API 时，务必设置 `server.allowed_roots`，明确指定 grader 可以读取的数据目录。

**不要把 `allowed_roots` 留为空数组。** 当前 release 中，`allowed_roots: []` 是宽松 / fail-open 配置，因此不能把空数组当作安全边界。

执行 `bash install.sh` 时，setup 会把配置的 dataset root 写入 `image_grader/config.local.json` 的 `server.allowed_roots`。如果你手动创建或修改 grader config，请使用明确且尽可能窄的绝对路径，例如：

```json
{
  "server": {
    "allowed_roots": ["/absolute/path/to/your/dataset"]
  }
}
```

不要为了方便把 `/` 或整个 home directory 设为 allowed root。

更多 grader 设置请查看 [Image Grader](docs/image_grader.md)。

## Labeler 路径安全

Labeler 只接受解析后仍位于配置 `dataset_root` 内的 `saved_path`。Absolute path、`..` 路径或 symlink 如果最终解析到 dataset root 之外，该 session 会被 catalog 拒绝，不会被用于读取图片。

## Label event idempotency

重复提交完全相同的 `event_id` 和相同 label event 内容仍然是 idempotent：系统会返回原事件，不会重复写入。

如果同一个 `event_id` 被用于提交不同 pair、decision、defects、note、reviewer/client context 或其他 label event 内容，系统会明确拒绝，而不会静默当作成功重试。

## Wildcard Downloads

七组第三方 wildcard pack 不直接存放在 Git 中。来源版本、预期大小、SHA-256、转换与 provenance 记录在 `assets/wildcard_sources.json`。

```bash
bash install_wildcard.sh --list-sources
bash install_wildcard.sh --accept-third-party-terms
```

如需 Civitai authentication，请在 process environment 设置 `CIVITAI_API_TOKEN`。请先阅读 [Asset Provenance](docs/asset_provenance_audit.md) 与 [Third-Party Notices](THIRD_PARTY_NOTICES.md)。

## 测试

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

完整 release verification 请查看 [Release Verification](docs/release_verification.md)。

## 安全限制

- 所有 HTTP services 应保持在 localhost 或可信 VPN；这些服务不是 hardened public web services，也不提供 TLS termination。
- Review / grader UI 可能显示 dataset、task、session、checkpoint、seed 与 run metadata。
- Aesthetic grader 不能可靠检测所有手指、手部、解剖或其他生成缺陷。
- 第三方 wildcard 与 model license 不受本 repository 的 MIT license 覆盖。
- 这是一次性 release，不承诺长期维护或 hosted support。

在暴露任何 endpoint 前，请阅读 [Security](SECURITY.md)。
