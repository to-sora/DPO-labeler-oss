from __future__ import annotations

import abc
import copy
import hashlib
import io
import json
import mimetypes
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlparse

import requests
from PIL import Image


@dataclass(frozen=True)
class GenerationReceipt:
    workflow_name: str
    prompt_id: str
    positive_prompt: str
    negative_prompt: str
    seed: int
    steps: int
    cfg: float
    width: int
    height: int
    ckpt: str
    lora_stack_config: Dict[str, Any]
    original_filename: str
    saved_filename: str
    saved_path: str
    image_sha256: str
    image_size_bytes: int
    image_width: int
    image_height: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WorkflowBase(abc.ABC):
    """Base class for ComfyUI API-format workflows.

    Important contract:
    - This base class does not rely on stable node IDs.
    - Common nodes are resolved by `_meta.title` and, for the checkpoint loader,
      by `class_type`.
    - The title strings below are still a hardcoded compatibility contract.
      Any workflow that wants to use this base class must expose the same
      semantic input set with the same titles, or override TITLE_MAP.

    That means this base class is reusable outside SDXL as long as another
    workflow keeps the same prompt/seed/step/cfg/size/LoRA title set.
    """

    TEMPLATE_JSON_FILE: str = ""

    TITLE_MAP: Dict[str, str] = {
        "positive_prompt": "POS_pormpt",
        "negative_prompt": "NEG_pormpt",
        "lora_stack": "简易Lora堆",
        "width": "Width_S",
        "height": "Height-S",
        "cfg": "浮点数",
        "steps": "Step_S",
        "seed": "seed",
        "save_image": "保存图像",
    }

    CLASS_TYPE_MAP: Dict[str, str] = {
        "checkpoint": "CheckpointLoaderSimple",
    }

    def __init__(
        self,
        ckpt: str,
        lora_stack_config: Optional[Mapping[str, Any]],
        output_dir: str | Path,
        url: str,
        port: int,
        allow_insecure: bool = True,
    ) -> None:
        self.ckpt = ckpt
        self.lora_stack_config = dict(lora_stack_config or {})
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.allow_insecure = allow_insecure
        self.base_url = self._build_base_url(url=url, port=port, allow_insecure=allow_insecure)
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self.client_id = str(uuid.uuid4())

    @classmethod
    def template_path(cls) -> Path:
        if not cls.TEMPLATE_JSON_FILE:
            raise ValueError(f"{cls.__name__} must define TEMPLATE_JSON_FILE")
        return Path(__file__).with_name(cls.TEMPLATE_JSON_FILE)

    @classmethod
    def load_template(cls) -> Dict[str, Any]:
        with cls.template_path().open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @classmethod
    def resolve_node_map(cls, workflow: Mapping[str, Any]) -> Dict[str, str]:
        resolved: Dict[str, str] = {}
        for semantic_name, title in cls.TITLE_MAP.items():
            resolved[semantic_name] = cls._find_single_node_id_by_title(workflow, title)
        for semantic_name, class_type in cls.CLASS_TYPE_MAP.items():
            resolved[semantic_name] = cls._find_single_node_id_by_class_type(workflow, class_type)
        return resolved

    def build_prompt(
        self,
        positive_prompt: str,
        negative_prompt: str,
        seed: int,
        steps: int,
        cfg: float,
        width: int,
        height: int,
        **workflow_kwargs: Any,
    ) -> Dict[str, Any]:
        workflow = copy.deepcopy(self.load_template())
        node_map = self.resolve_node_map(workflow)
        self._apply_common_inputs(
            workflow=workflow,
            node_map=node_map,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            cfg=cfg,
            width=width,
            height=height,
        )
        self._apply_workflow_specific(workflow=workflow, node_map=node_map, **workflow_kwargs)
        return workflow

    def generate(
        self,
        positive_prompt: str,
        negative_prompt: str,
        seed: int,
        steps: int,
        cfg: float,
        width: int,
        height: int,
        timeout_seconds: float = 300.0,
        poll_interval_seconds: float = 1.0,
        **workflow_kwargs: Any,
    ) -> GenerationReceipt:
        prompt = self.build_prompt(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            cfg=cfg,
            width=width,
            height=height,
            **workflow_kwargs,
        )
        prompt_id = self._queue_prompt(prompt)
        history_item = self._wait_for_history(
            prompt_id=prompt_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        image_record = self._select_image_record(history_item)
        image_bytes = self._download_image(image_record)
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        image_suffix = self._resolve_image_suffix(image_record.get("filename", ""), image_bytes)
        saved_filename = f"{image_sha256}{image_suffix}"
        saved_path = self.output_dir / saved_filename
        saved_path.write_bytes(image_bytes)

        image_width, image_height = self._read_image_size(image_bytes)
        return GenerationReceipt(
            workflow_name=self.__class__.__name__,
            prompt_id=prompt_id,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            seed=int(seed),
            steps=int(steps),
            cfg=float(cfg),
            width=int(width),
            height=int(height),
            ckpt=self.ckpt,
            lora_stack_config=self._normalized_lora_stack_inputs(),
            original_filename=str(image_record.get("filename", "")),
            saved_filename=saved_filename,
            saved_path=str(saved_path),
            image_sha256=image_sha256,
            image_size_bytes=len(image_bytes),
            image_width=image_width,
            image_height=image_height,
        )

    def _apply_common_inputs(
        self,
        workflow: Dict[str, Any],
        node_map: Mapping[str, str],
        positive_prompt: str,
        negative_prompt: str,
        seed: int,
        steps: int,
        cfg: float,
        width: int,
        height: int,
    ) -> None:
        workflow[node_map["checkpoint"]]["inputs"]["ckpt_name"] = self.ckpt
        workflow[node_map["positive_prompt"]]["inputs"]["value"] = positive_prompt
        workflow[node_map["negative_prompt"]]["inputs"]["value"] = negative_prompt
        workflow[node_map["width"]]["inputs"]["value"] = int(width)
        workflow[node_map["height"]]["inputs"]["value"] = int(height)
        workflow[node_map["cfg"]]["inputs"]["value"] = float(cfg)
        workflow[node_map["steps"]]["inputs"]["value"] = int(steps)
        workflow[node_map["seed"]]["inputs"]["value"] = int(seed)
        workflow[node_map["lora_stack"]]["inputs"] = self._normalized_lora_stack_inputs(
            base_inputs=workflow[node_map["lora_stack"]]["inputs"]
        )

    def _normalized_lora_stack_inputs(
        self,
        base_inputs: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        inputs: Dict[str, Any] = dict(base_inputs or {})
        inputs.update(self.lora_stack_config)

        requested_toggle = bool(inputs.get("toggle", True))
        active_lora_count = 0
        for index in range(1, 11):
            name_key = f"lora_{index}_name"
            strength_key = f"lora_{index}_strength"
            model_strength_key = f"lora_{index}_model_strength"
            clip_strength_key = f"lora_{index}_clip_strength"

            name_value = inputs.get(name_key, "None")
            if not requested_toggle or not name_value or str(name_value).strip().lower() == "none":
                inputs[name_key] = "None"
            else:
                active_lora_count = index

            inputs.setdefault(strength_key, 1.0)
            inputs.setdefault(model_strength_key, 1.0)
            inputs.setdefault(clip_strength_key, 1.0)

        inputs["toggle"] = requested_toggle and active_lora_count > 0
        inputs["mode"] = inputs.get("mode", "simple")
        inputs["num_loras"] = max(int(inputs.get("num_loras", max(active_lora_count, 1))), 1)
        return inputs

    def _queue_prompt(self, prompt: Dict[str, Any]) -> str:
        payload = {"prompt": prompt, "client_id": self.client_id}
        response = self.session.post(
            f"{self.base_url}/prompt",
            json=payload,
            timeout=60,
            verify=self._verify_ssl,
        )
        response.raise_for_status()
        data = response.json()
        if "prompt_id" not in data:
            raise RuntimeError(f"Unexpected /prompt response: {data}")
        return str(data["prompt_id"])

    def _wait_for_history(
        self,
        prompt_id: str,
        timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> Dict[str, Any]:
        deadline = time.time() + timeout_seconds
        last_payload: Dict[str, Any] | None = None

        while time.time() < deadline:
            response = self.session.get(
                f"{self.base_url}/history/{prompt_id}",
                timeout=60,
                verify=self._verify_ssl,
            )
            response.raise_for_status()
            payload = response.json()
            last_payload = payload

            history_item = payload.get(prompt_id)
            if history_item:
                outputs = history_item.get("outputs", {})
                if any("images" in node_output for node_output in outputs.values()):
                    return history_item

                status = history_item.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI execution failed for prompt_id={prompt_id}: {history_item}")

            time.sleep(poll_interval_seconds)

        raise TimeoutError(
            f"Timed out waiting for prompt_id={prompt_id}. Last /history payload: {last_payload}"
        )

    def _select_image_record(self, history_item: Mapping[str, Any]) -> Dict[str, Any]:
        outputs = history_item.get("outputs", {})
        for node_output in outputs.values():
            for image in node_output.get("images", []):
                if image.get("type") == "output":
                    return dict(image)
        for node_output in outputs.values():
            images = node_output.get("images", [])
            if images:
                return dict(images[0])
        raise RuntimeError(f"No image output found in history item: {history_item}")

    def _download_image(self, image_record: Mapping[str, Any]) -> bytes:
        params = {
            "filename": image_record["filename"],
            "subfolder": image_record.get("subfolder", ""),
            "type": image_record.get("type", "output"),
        }
        response = self.session.get(
            f"{self.base_url}/view",
            params=params,
            timeout=120,
            verify=self._verify_ssl,
        )
        response.raise_for_status()
        return response.content

    @staticmethod
    def _read_image_size(image_bytes: bytes) -> tuple[int, int]:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return int(image.width), int(image.height)

    @staticmethod
    def _resolve_image_suffix(original_filename: str, image_bytes: bytes) -> str:
        suffix = Path(original_filename).suffix.lower()
        if suffix:
            return suffix
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format:
                guessed = f".{image.format.lower()}"
                if guessed == ".jpeg":
                    return ".jpg"
                return guessed
        mime_guess, _ = mimetypes.guess_type(original_filename)
        if mime_guess == "image/png":
            return ".png"
        if mime_guess == "image/jpeg":
            return ".jpg"
        return ".bin"

    @property
    def _verify_ssl(self) -> bool:
        return not self.allow_insecure

    @staticmethod
    def _build_base_url(url: str, port: int, allow_insecure: bool) -> str:
        parsed = urlparse(url)
        if parsed.scheme:
            base = url.rstrip("/")
            if parsed.port is None:
                return f"{base}:{port}"
            return base
        scheme = "http" if allow_insecure else "https"
        return f"{scheme}://{url}:{port}"

    @classmethod
    def _find_single_node_id_by_title(cls, workflow: Mapping[str, Any], title: str) -> str:
        matches = [node_id for node_id, node in workflow.items() if cls._node_title(node) == title]
        if len(matches) != 1:
            raise KeyError(f"Expected exactly one node with title={title!r}, found {matches}")
        return matches[0]

    @classmethod
    def _find_single_node_id_by_class_type(cls, workflow: Mapping[str, Any], class_type: str) -> str:
        matches = [node_id for node_id, node in workflow.items() if node.get("class_type") == class_type]
        if len(matches) != 1:
            raise KeyError(f"Expected exactly one node with class_type={class_type!r}, found {matches}")
        return matches[0]

    @classmethod
    def find_node_ids_by_title(cls, workflow: Mapping[str, Any], title: str) -> list[str]:
        return [node_id for node_id, node in workflow.items() if cls._node_title(node) == title]

    @staticmethod
    def _node_title(node: Mapping[str, Any]) -> Optional[str]:
        meta = node.get("_meta") or {}
        return meta.get("title")

    @abc.abstractmethod
    def _apply_workflow_specific(
        self,
        workflow: Dict[str, Any],
        node_map: Mapping[str, str],
        **workflow_kwargs: Any,
    ) -> None:
        raise NotImplementedError
