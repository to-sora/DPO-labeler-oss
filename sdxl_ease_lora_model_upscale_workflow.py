from __future__ import annotations

from typing import Any, Dict, Mapping

from workflow_base import WorkflowBase


class SdxlEaseLoraModelUpscaleWorkflow(WorkflowBase):
    TEMPLATE_JSON_FILE = "DPO_SDXL_EASE_LORA_UPSCALE.json"
    UPSCALE_MODEL_TITLE = "加载放大模型"

    def _apply_workflow_specific(
        self,
        workflow: Dict[str, Any],
        node_map: Mapping[str, str],
        upscale_model_name: str | None = None,
        **workflow_kwargs: Any,
    ) -> None:
        if not upscale_model_name:
            return None

        model_loader_node_ids = self.find_node_ids_by_title(workflow, self.UPSCALE_MODEL_TITLE)
        if len(model_loader_node_ids) != 1:
            raise KeyError(
                f"Expected exactly one upscale model loader titled {self.UPSCALE_MODEL_TITLE!r}, found {model_loader_node_ids}"
            )
        workflow[model_loader_node_ids[0]]["inputs"]["model_name"] = upscale_model_name
