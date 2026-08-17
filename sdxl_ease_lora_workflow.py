from __future__ import annotations

from typing import Any, Dict, Mapping

from workflow_base import WorkflowBase


class SdxlEaseLoraWorkflow(WorkflowBase):
    TEMPLATE_JSON_FILE = "DPO_SDXL_EASE_LORA.json"

    def _apply_workflow_specific(
        self,
        workflow: Dict[str, Any],
        node_map: Mapping[str, str],
        **workflow_kwargs: Any,
    ) -> None:
        return None
