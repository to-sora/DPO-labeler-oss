import json
import math
from pathlib import Path

import yaml
from PIL import Image, ImageDraw


def prepare(root: Path, external: Path) -> Path:
    external.mkdir(parents=True, exist_ok=True)
    images = []
    for i in range(10):
        width, height = [(480, 640), (800, 450), (512, 512)][i % 3]
        image = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(image)
        for y in range(height):
            t = y / height
            draw.line((0, y, width, y), fill=(int(70 + 100*t), 110 + i*8, int(210 - 80*t)))
        draw.ellipse((width*.57, height*.10, width*.85, height*.32), fill="#ffe8ae")
        for layer in range(3):
            points = [(0, height)] + [(x, int(height*(.45 + .12*layer) + math.sin(x/90+i)*height*.12))
                                     for x in range(0, width+1, 8)] + [(width, height)]
            draw.polygon(points, fill=["#637b83", "#3d676e", "#264c57"][layer])
        draw.text((20, 20), f"DEMO {i + 1:02d}", fill="white", font_size=24)
        image.save(external / f"demo-{i}.png")
        images.append({"image-path": f"demo-{i}.png", "prompt-path": "missing.txt",
                       "prompt": f"Landscape variation {i + 1}; mountains, warm sun, detailed light."})
    (external / "prompt.txt").write_text("Prompt loaded from an external file.", encoding="utf-8")
    images[0]["prompt-path"] = "prompt.txt"
    manifest = {"task-name": "Ten landscape studies", "character-name": "Landscape 山景",
                "image_dir": str(external), "image": images,
                "dim-name": ["Composition", "Lighting", "Detail", "Color", "Character"]}
    root.mkdir(parents=True, exist_ok=True)
    path = root / "import.yaml"
    path.write_text(yaml.safe_dump(manifest, allow_unicode=True), encoding="utf-8")
    return path


def legacy(dataset: Path, external: Path) -> None:
    from dpo_labeler.backend.test_labeler_app import _session_row
    dataset.mkdir(parents=True, exist_ok=True)
    row = _session_row("legacy-demo", (external / "demo-0.png", external / "demo-1.png"),
                       session_index=0)
    (dataset / "sessions.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
