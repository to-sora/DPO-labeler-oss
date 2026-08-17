from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from sdxl_ease_lora_latent_upscale_workflow import SdxlEaseLoraLatentUpscaleWorkflow
from sdxl_ease_lora_model_upscale_workflow import SdxlEaseLoraModelUpscaleWorkflow
from sdxl_ease_lora_workflow import SdxlEaseLoraWorkflow
from workflow_base import WorkflowBase


class WorkflowGenerateTests(unittest.TestCase):
    def _make_png_bytes(self, width: int = 64, height: int = 48) -> bytes:
        image = Image.new("RGB", (width, height), color=(12, 34, 56))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _fake_history(self, filename: str = "server_name.png") -> dict:
        return {
            "outputs": {
                "19": {
                    "images": [
                        {
                            "filename": filename,
                            "subfolder": "",
                            "type": "output",
                        }
                    ]
                }
            }
        }

    def _assert_saved_image(self, workflow_cls) -> None:
        image_bytes = self._make_png_bytes()
        expected_sha = hashlib.sha256(image_bytes).hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            workflow = workflow_cls(
                ckpt="model.safetensors",
                lora_stack_config=None,
                output_dir=tmpdir,
                url="127.0.0.1",
                port=8188,
            )

            with patch.object(workflow, "_queue_prompt", return_value="prompt-123"), patch.object(
                workflow, "_wait_for_history", return_value=self._fake_history()
            ), patch.object(workflow, "_download_image", return_value=image_bytes):
                receipt = workflow.generate(
                    positive_prompt="good prompt",
                    negative_prompt="bad prompt",
                    seed=123,
                    steps=30,
                    cfg=7.5,
                    width=1024,
                    height=1024,
                )

            saved_path = Path(receipt.saved_path)
            self.assertTrue(saved_path.exists())
            self.assertEqual(saved_path.name, f"{expected_sha}.png")
            self.assertEqual(saved_path.read_bytes(), image_bytes)
            self.assertEqual(receipt.image_size_bytes, len(image_bytes))
            self.assertEqual((receipt.image_width, receipt.image_height), (64, 48))
            self.assertFalse(receipt.lora_stack_config["toggle"])

    def test_base_workflow_saves_sha256_png_and_returns_size(self) -> None:
        self._assert_saved_image(SdxlEaseLoraWorkflow)

    def test_latent_upscale_workflow_saves_sha256_png_and_returns_size(self) -> None:
        self._assert_saved_image(SdxlEaseLoraLatentUpscaleWorkflow)

    def test_model_upscale_workflow_saves_sha256_png_and_returns_size(self) -> None:
        self._assert_saved_image(SdxlEaseLoraModelUpscaleWorkflow)

    def test_build_prompt_disables_lora_when_no_lora_is_given(self) -> None:
        workflow = SdxlEaseLoraWorkflow(
            ckpt="model.safetensors",
            lora_stack_config={},
            output_dir=".",
            url="127.0.0.1",
            port=8188,
        )
        prompt = workflow.build_prompt(
            positive_prompt="p",
            negative_prompt="n",
            seed=1,
            steps=2,
            cfg=3.0,
            width=4,
            height=5,
        )
        node_map = workflow.resolve_node_map(prompt)
        lora_inputs = prompt[node_map["lora_stack"]]["inputs"]
        self.assertFalse(lora_inputs["toggle"])
        for index in range(1, 11):
            self.assertEqual(lora_inputs[f"lora_{index}_name"], "None")
        self.assertEqual(prompt[node_map["checkpoint"]]["inputs"]["ckpt_name"], "model.safetensors")
        self.assertEqual(prompt[node_map["width"]]["inputs"]["value"], 4)
        self.assertEqual(prompt[node_map["height"]]["inputs"]["value"], 5)

    def test_resolve_node_map_uses_meta_titles_not_node_ids(self) -> None:
        workflow = SdxlEaseLoraWorkflow.load_template()
        remapped = {}
        for offset, old_node_id in enumerate(sorted(workflow.keys(), key=int), start=1000):
            remapped[str(offset)] = workflow[old_node_id]

        node_map = WorkflowBase.resolve_node_map(remapped)
        self.assertEqual(remapped[node_map["positive_prompt"]]["_meta"]["title"], "POS_pormpt")
        self.assertEqual(remapped[node_map["negative_prompt"]]["_meta"]["title"], "NEG_pormpt")
        self.assertEqual(remapped[node_map["width"]]["_meta"]["title"], "Width_S")
        self.assertEqual(remapped[node_map["seed"]]["_meta"]["title"], "seed")


if __name__ == "__main__":
    unittest.main()
