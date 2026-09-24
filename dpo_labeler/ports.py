import ipaddress
import json
import subprocess
from pathlib import Path

import yaml


def configuration() -> dict:
    path = Path(__file__).parent / "config" / "port-config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def addresses(policy: str) -> list[str]:
    if policy == "public":
        return ["0.0.0.0"]
    if policy == "local":
        return ["127.0.0.1"]
    if policy != "standard":
        raise ValueError("policy must be local, standard or public")
    links = json.loads(subprocess.check_output(["ip", "-4", "-j", "addr"], text=True))
    hosts = [a["local"] for link in links if link["ifname"].startswith(("wg", "tails"))
             for a in link["addr_info"] if a["family"] == "inet"]
    if not hosts:
        raise ValueError("No IPv4 address on a wg* or tails* interface")
    return hosts


def whitelist(config: dict) -> set[str] | None:
    if not config.get("enable_whitelist", False):
        return None
    return {str(ipaddress.IPv4Address(value.strip()))
            for value in config.get("whitelist", "").split(",") if value.strip()}
