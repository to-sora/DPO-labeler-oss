from pathlib import Path

import yaml


def configuration() -> dict:
    path = Path(__file__).parent / "config" / "port-config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))
