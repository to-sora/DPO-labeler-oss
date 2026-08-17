from __future__ import annotations

from typing import Any, Dict, Mapping

from workflow_base import WorkflowBase


class SdxlEaseLoraLatentUpscaleWorkflow(WorkflowBase):
    TEMPLATE_JSON_FILE = "DPO_SDXL_EASE_LORA_UP_SCALE.json"
    LATENT_UPSCALE_TITLE = "缩放Latent（比例）"

    def _apply_workflow_specific(
        self,
        workflow: Dict[str, Any],
        node_map: Mapping[str, str],
        upscale_by: float = 1.5,
        **workflow_kwargs: Any,
    ) -> None:
        for node_id in self.find_node_ids_by_title(workflow, self.LATENT_UPSCALE_TITLE):
            workflow[node_id]["inputs"]["scale_by"] = float(upscale_by)
