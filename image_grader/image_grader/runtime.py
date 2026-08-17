from __future__ import annotations

from .config import GraderConfig


class RuntimeValidationError(ValueError):
    pass


def validate_runtime(config: GraderConfig) -> None:
    if not config.device.startswith("cuda"):
        return
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeValidationError(
            "config.device requests CUDA but torch is not installed. "
            "Install requirements-cuda13.txt or set device to cpu."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeValidationError(
            "config.device requests CUDA but torch cannot access a CUDA device. "
            "Check the NVIDIA driver with nvidia-smi, or set device to cpu."
        )
    device_index = _cuda_index(config.device)
    if device_index is not None and device_index >= torch.cuda.device_count():
        raise RuntimeValidationError(
            f"config.device={config.device!r} is out of range; "
            f"torch sees {torch.cuda.device_count()} CUDA device(s)."
        )


def _cuda_index(device: str) -> int | None:
    if device == "cuda":
        return None
    if not device.startswith("cuda:"):
        return None
    try:
        return int(device.split(":", 1)[1])
    except ValueError as exc:
        raise RuntimeValidationError(f"invalid CUDA device string: {device!r}") from exc
