from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import GraderConfig, ModelConfig


ScorePayload = dict[str, Any]


class PreparedImageLike(Protocol):
    image_path: Path
    image: Any


class ModelBackend:
    def __init__(self, grader_config: GraderConfig, model_config: ModelConfig) -> None:
        self.grader_config = grader_config
        self.model_config = model_config
        self.model_id = model_config.model_id

    def score_batch(self, images: list[PreparedImageLike]) -> list[ScorePayload]:
        raise NotImplementedError

    def close(self) -> None:
        return None


@dataclass
class ModelRegistry:
    config: GraderConfig
    _loaded: dict[str, ModelBackend]

    def get(self, model_id: str) -> ModelBackend:
        if model_id not in self._loaded:
            self._loaded[model_id] = create_backend(self.config, self.config.models[model_id])
        return self._loaded[model_id]

    def close(self) -> None:
        for backend in self._loaded.values():
            backend.close()
        self._loaded.clear()


def create_registry(config: GraderConfig) -> ModelRegistry:
    return ModelRegistry(config=config, _loaded={})


def create_backend(grader_config: GraderConfig, model_config: ModelConfig) -> ModelBackend:
    if model_config.kind == "transformers_vit_hq":
        return TransformersVitHqBackend(grader_config, model_config)
    if model_config.kind == "clip_mlp_waifu_v3":
        return WaifuScorerV3Backend(grader_config, model_config)
    if model_config.kind == "torch_pickle_regressor":
        return TorchPickleRegressorBackend(grader_config, model_config)
    raise ValueError(f"unsupported model kind for {model_config.model_id}: {model_config.kind}")


class TransformersVitHqBackend(ModelBackend):
    def __init__(self, grader_config: GraderConfig, model_config: ModelConfig) -> None:
        super().__init__(grader_config, model_config)
        self._torch = None
        self._processor = None
        self._model = None

    def score_batch(self, images: list[PreparedImageLike]) -> list[ScorePayload]:
        self._ensure_loaded()
        torch = self._torch
        assert torch is not None
        assert self._processor is not None
        assert self._model is not None
        batch = [item.image for item in images]
        inputs = self._processor(images=batch, return_tensors="pt")
        inputs = {key: value.to(self.grader_config.device) for key, value in inputs.items()}
        hq_index = self._hq_index()
        with torch.inference_mode(), _autocast_context(torch, self.grader_config.device, self.grader_config.dtype):
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits.float(), dim=-1)
        payloads: list[ScorePayload] = []
        for row in probs.detach().cpu():
            hq_probability = float(row[hq_index].item())
            raw = {"hq_probability": hq_probability}
            if row.numel() > 1:
                raw["probabilities"] = [float(value) for value in row.tolist()]
            payloads.append({"ok": True, "score": hq_probability * 10.0, "scale": "0_10", "raw": raw, "error": None})
        return payloads

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import ViTForImageClassification, ViTImageProcessor

        model_path = self.grader_config.resolve_model_path(self.model_config)
        dtype = _resolve_torch_dtype(torch, self.grader_config.device, self.grader_config.dtype)
        self._processor = ViTImageProcessor.from_pretrained(str(model_path), local_files_only=True)
        self._model = ViTForImageClassification.from_pretrained(
            str(model_path),
            local_files_only=True,
            torch_dtype=dtype,
        )
        self._model.to(self.grader_config.device)
        self._model.eval()
        self._torch = torch

    def _hq_index(self) -> int:
        assert self._model is not None
        label2id = getattr(self._model.config, "label2id", {}) or {}
        if self.model_config.score_label in label2id:
            return int(label2id[self.model_config.score_label])
        return int(self.model_config.options.get("score_index", 0))


class WaifuScorerV3Backend(ModelBackend):
    def __init__(self, grader_config: GraderConfig, model_config: ModelConfig) -> None:
        super().__init__(grader_config, model_config)
        self._torch = None
        self._processor = None
        self._clip_model = None
        self._head = None

    def score_batch(self, images: list[PreparedImageLike]) -> list[ScorePayload]:
        self._ensure_loaded()
        torch = self._torch
        assert torch is not None
        assert self._processor is not None
        assert self._clip_model is not None
        assert self._head is not None
        inputs = self._processor(images=[item.image for item in images], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.grader_config.device)
        with torch.inference_mode(), _autocast_context(torch, self.grader_config.device, self.grader_config.dtype):
            features = self._clip_model.get_image_features(pixel_values=pixel_values)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            elif isinstance(features, (list, tuple)):
                features = features[0]
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            scores = self._head(features.float()).squeeze(-1)
        payloads: list[ScorePayload] = []
        for value in scores.detach().cpu().tolist():
            score = float(value)
            payloads.append({"ok": True, "score": score, "scale": "0_10", "raw": {"aesthetic": score}, "error": None})
        return payloads

    def _ensure_loaded(self) -> None:
        if self._head is not None:
            return
        import torch
        from safetensors.torch import load_file
        from transformers import CLIPImageProcessor, CLIPModel

        model_path = self.grader_config.resolve_model_path(self.model_config)
        clip_model = self.model_config.clip_model or "openai/clip-vit-large-patch14"
        dtype = _resolve_torch_dtype(torch, self.grader_config.device, self.grader_config.dtype)
        self._processor = CLIPImageProcessor.from_pretrained(clip_model)
        self._clip_model = CLIPModel.from_pretrained(clip_model, torch_dtype=dtype)
        self._clip_model.to(self.grader_config.device)
        self._clip_model.eval()
        self._head = _build_waifu_v3_head(torch).to(self.grader_config.device)
        state_dict = _remap_waifu_v3_state_dict(load_file(str(model_path), device="cpu"))
        self._head.load_state_dict(state_dict)
        self._head.eval()
        self._torch = torch


class TorchPickleRegressorBackend(ModelBackend):
    def __init__(self, grader_config: GraderConfig, model_config: ModelConfig) -> None:
        super().__init__(grader_config, model_config)
        self._torch = None
        self._transform = None
        self._model = None

    def score_batch(self, images: list[PreparedImageLike]) -> list[ScorePayload]:
        self._ensure_loaded()
        torch = self._torch
        assert torch is not None
        assert self._transform is not None
        assert self._model is not None
        tensors = [self._transform(item.image) for item in images]
        batch = torch.stack(tensors).to(self.grader_config.device)
        if self.grader_config.dtype == "float16" or (
            self.grader_config.dtype == "auto" and self.grader_config.device.startswith("cuda")
        ):
            batch = batch.half()
        with torch.inference_mode(), _autocast_context(torch, self.grader_config.device, self.grader_config.dtype):
            output = self._model(batch)
        if isinstance(output, (list, tuple)):
            output = output[0]
        scores = output.float().detach().cpu()
        if scores.ndim > 1 and scores.shape[-1] == 1:
            scores = scores.squeeze(-1)
        payloads: list[ScorePayload] = []
        for row in scores:
            if getattr(row, "ndim", 0) > 0 and row.numel() > 1:
                values = [float(value) for value in row.tolist()]
                score = values[-1]
                raw: dict[str, Any] = {"values": values}
            else:
                score = float(row.item())
                raw = {"aesthetic": score}
            payloads.append({"ok": True, "score": score, "scale": "0_10", "raw": raw, "error": None})
        return payloads

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.model_config.trusted_pickle:
            raise ValueError(
                f"{self.model_id} uses a pickle .pth file; set trusted_pickle=true in config after verifying the file"
            )
        import torch
        from torchvision import transforms

        model_path = self.grader_config.resolve_model_path(self.model_config)
        self._model = torch.load(str(model_path), map_location=self.grader_config.device, weights_only=False)
        if not hasattr(self._model, "eval") or not callable(self._model):
            raise ValueError(f"{self.model_id} did not load as a callable torch module from {model_path}")
        self._model.to(self.grader_config.device)
        if self.grader_config.dtype == "float16" or (
            self.grader_config.dtype == "auto" and self.grader_config.device.startswith("cuda")
        ):
            self._model.half()
        self._model.eval()
        input_size = int(self.model_config.input_size or self.model_config.options.get("input_size", 224))
        mean = tuple(float(value) for value in self.model_config.options.get("mean", (0.485, 0.456, 0.406)))
        std = tuple(float(value) for value in self.model_config.options.get("std", (0.229, 0.224, 0.225)))
        self._transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self._torch = torch


def _build_waifu_v3_head(torch: Any) -> Any:
    nn = torch.nn

    class WaifuV3Mlp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(768, 2048),
                nn.BatchNorm1d(2048),
                nn.Tanh(),
                nn.Linear(2048, 512),
                nn.BatchNorm1d(512),
                nn.Tanh(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.Tanh(),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.Tanh(),
                nn.Linear(128, 32),
                nn.Linear(32, 1),
            )

        def forward(self, x: Any) -> Any:
            return self.layers(x)

    return WaifuV3Mlp()


def _remap_waifu_v3_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    index_map = {
        "layers.0": "layers.0",
        "layers.2": "layers.1",
        "layers.4": "layers.3",
        "layers.6": "layers.4",
        "layers.8": "layers.6",
        "layers.10": "layers.7",
        "layers.12": "layers.9",
        "layers.14": "layers.10",
        "layers.16": "layers.12",
        "layers.18": "layers.13",
    }
    remapped: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in index_map.items():
            if key == old_prefix or key.startswith(old_prefix + "."):
                new_key = new_prefix + key[len(old_prefix):]
                break
        remapped[new_key] = value
    return remapped


def _resolve_torch_dtype(torch: Any, device: str, dtype_name: str) -> Any:
    normalized = dtype_name.lower()
    if normalized == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype_name}")


def _autocast_context(torch: Any, device: str, dtype_name: str) -> Any:
    if not device.startswith("cuda"):
        return nullcontext()
    dtype = _resolve_torch_dtype(torch, device, dtype_name)
    if dtype is torch.float32:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)
