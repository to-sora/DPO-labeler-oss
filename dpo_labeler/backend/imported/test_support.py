import json
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

import yaml
from PIL import Image

from .service import ImportedTasks


class ImportCase(unittest.TestCase):
    def setUp(self) -> None:
        Path("output").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir="output")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.images = self.root / "external"
        self.images.mkdir()
        self.service = ImportedTasks(self.root / "state", [self.images])

    def manifest(self, n: int = 10, dims: int = 5) -> dict:
        for i in range(n):
            Image.new("RGB", (32, 40), (i % 255, 100, 200)).save(self.images / f"{i}.png")
        return {"task-name": "External task", "character-name": "Character 貓",
                "image_dir": str(self.images), "dim-name": [f"dim-{i}" for i in range(dims)],
                "image": [{"image-path": f"{i}.png", "prompt": f"prompt {i}"} for i in range(n)]}

    def create(self, n: int = 10, dims: int = 5) -> dict:
        return self.service.import_yaml(yaml.safe_dump(self.manifest(n, dims)))

    def vote(self, task: dict, rankings: list[list[int]] | None = None) -> dict:
        a, b = task["pair"]["image_ids"]
        ranks = rankings or [list(range(len(task["images"])))] * len(task["dimensions"])
        return {"comparison_id": task["pair"]["comparison_id"],
                "client_instance_id": "browser-instance", "choices": {
                    d: "a_good" if order.index(a) < order.index(b) else "b_good"
                    for d, order in zip(task["dimensions"], ranks)}}

    def archive(self, task_id: str, dim: int | None = None) -> dict[str, str]:
        with zipfile.ZipFile(BytesIO(self.service.export(task_id, dim))) as archive:
            return {p: archive.read(p).decode("utf-8") for p in archive.namelist()}


def rows(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines()]
