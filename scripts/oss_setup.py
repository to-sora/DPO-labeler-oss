#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import secrets
import shlex
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__:
    from . import install_wildcards
else:
    import install_wildcards


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env.oss"
RUNNER_ENV_PATH = ROOT / ".env"
GRADER_EXAMPLE_CONFIG = ROOT / "image_grader" / "config.example.json"
GRADER_LOCAL_CONFIG = ROOT / "image_grader" / "config.local.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install wildcard assets and configure local OSS endpoints.")
    parser.add_argument("--non-interactive", action="store_true", help="Use defaults and write config without prompts")
    parser.add_argument(
        "--accept-third-party-terms",
        action="store_true",
        help="Confirm that you chose the wildcard downloads and accept the source terms",
    )
    parser.add_argument("--skip-wildcards", action="store_true", help="Configure endpoints without wildcard packs")
    parser.add_argument("--force-wildcards", action="store_true", help="Replace wildcard files that differ")
    parser.add_argument("--force-unzip", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--wildcard-archive-dir", type=Path, help="Use verified wildcard ZIPs from this directory")
    parser.add_argument("--wildcard-cache-dir", type=Path, default=install_wildcards.DEFAULT_CACHE)
    parser.add_argument(
        "--install-grader-deps",
        choices=("none", "cpu", "cuda13"),
        default=None,
        help="Install optional image-grader dependencies into the configured grader Python environment",
    )
    parser.add_argument(
        "--civitai-proxy",
        default=os.environ.get("CIVITAI_PROXY", ""),
        help="Proxy used only for Civitai hosts; CIVITAI_PROXY is also supported",
    )
    parser.add_argument(
        "--civitai-ca-bundle",
        type=Path,
        default=os.environ.get("CIVITAI_CA_BUNDLE") or None,
        help="Additional CA certificate used only with the Civitai proxy",
    )
    parser.add_argument(
        "--allow-legacy-proxy-ca",
        action="store_true",
        help="Allow an explicitly trusted proxy CA that lacks strict X.509 extensions",
    )
    parser.add_argument(
        "--civitai-wait-seconds",
        type=float,
        default=os.environ.get("CIVITAI_WAIT_SECONDS", "0"),
        help="Delay between Civitai archive requests",
    )
    parser.add_argument(
        "--civitai-token-query",
        action="store_true",
        help="Send the API token in the first Civitai URL when a proxy strips Authorization",
    )
    parser.add_argument(
        "--civitai-direct-redirects",
        action="store_true",
        help="Bypass the Civitai proxy after the first HTTPS redirect",
    )
    args = parser.parse_args(argv)

    if not args.skip_wildcards:
        manifest = install_wildcards.load_manifest(install_wildcards.DEFAULT_MANIFEST)
        accepted = bool(args.accept_third_party_terms)
        if not accepted and not args.non_interactive:
            accepted = prompt_wildcard_terms(manifest)
        try:
            install_wildcards.install(
                manifest,
                destination=install_wildcards.DEFAULT_DESTINATION,
                accept_terms=accepted,
                force=bool(args.force_wildcards or args.force_unzip),
                archive_dir=args.wildcard_archive_dir.resolve() if args.wildcard_archive_dir else None,
                cache_dir=args.wildcard_cache_dir.resolve(),
                civitai_proxy=str(args.civitai_proxy).strip() or None,
                civitai_ca_bundle=args.civitai_ca_bundle.expanduser().resolve()
                if args.civitai_ca_bundle
                else None,
                allow_legacy_proxy_ca=bool(args.allow_legacy_proxy_ca),
                civitai_wait_seconds=float(args.civitai_wait_seconds),
                civitai_token_query=bool(args.civitai_token_query),
                civitai_direct_redirects=bool(args.civitai_direct_redirects),
                token=os.environ.get("CIVITAI_API_TOKEN", "").strip() or None,
                timeout=120.0,
                retries=3,
            )
        except install_wildcards.InstallerError as exc:
            raise SystemExit(f"Wildcard installation failed: {exc}") from exc

    config = default_config()
    if not args.non_interactive:
        config = tui_config(config)
    grader_dependency_profile = args.install_grader_deps
    if grader_dependency_profile is None:
        grader_enabled = config["START_IMAGE_GRADER"] == "1" or config["START_ADAPTER"] == "1"
        grader_dependency_profile = (
            prompt_grader_dependency_profile(config["GRADER_DEVICE"])
            if grader_enabled and not args.non_interactive
            else "none"
        )
    install_grader_dependencies(config["GRADER_PYTHON_BIN"], grader_dependency_profile)
    write_grader_config(config)
    write_env(config)
    write_runner_env(config)
    chmod_executable(ROOT / "install.sh")
    chmod_executable(ROOT / "install_wildcard.sh")
    chmod_executable(ROOT / "start_all_endpoints.sh")

    print()
    print(f"Wrote {ENV_PATH.relative_to(ROOT)}")
    print(f"Wrote {RUNNER_ENV_PATH.relative_to(ROOT)}")
    print(f"Wrote {GRADER_LOCAL_CONFIG.relative_to(ROOT)}")
    print("Start services with: bash start_all_endpoints.sh")
    return 0


def prompt_wildcard_terms(manifest: dict[str, Any]) -> bool:
    print()
    install_wildcards.print_sources(manifest)
    print()
    response = input("Type AGREE to download these third-party files: ").strip()
    if response != "AGREE":
        raise SystemExit("Wildcard terms were not accepted; use --skip-wildcards for endpoint-only setup.")
    return True


def default_config() -> dict[str, str]:
    token = secrets.token_urlsafe(18)
    active_python = str(Path(sys.executable).absolute())
    return {
        "PYTHON_BIN": active_python,
        "GRADER_PYTHON_BIN": active_python,
        "COMFYUI_URL": "http://127.0.0.1:8188/",
        "COMFYUI_TIMEOUT_SECONDS": "300",
        "BIND_MODE": "local",
        "DATASET_ROOT": "output",
        "IMAGE_ROOT": "output",
        "RUNTIME_ROOT": "temp/oss_endpoints",
        "INVITE_TOKEN": token,
        "REVIEW_ROUND_SEED": "round-1",
        "LABELER_PORT": "8787",
        "EXPORT_VIEWER_PORT": "8084",
        "IMAGE_GRADER_PORT": "8790",
        "ADAPTER_PORT": "8087",
        "START_LABELER": "1",
        "START_EXPORT_VIEWER": "1",
        "START_IMAGE_GRADER": "0",
        "START_ADAPTER": "1",
        "GRADER_DEVICE": "cpu",
        "GRADER_MODELS_ROOT": "models/image_eval",
    }


def tui_config(config: dict[str, str]) -> dict[str, str]:
    print()
    print("Comfy DPO OSS setup")
    print("===================")
    print("Press Enter to accept defaults.")
    print()

    config["PYTHON_BIN"] = prompt("Python executable", config["PYTHON_BIN"])
    config["GRADER_PYTHON_BIN"] = prompt("Grader Python executable", config["GRADER_PYTHON_BIN"])
    config["COMFYUI_URL"] = prompt("ComfyUI URL", config["COMFYUI_URL"])
    config["COMFYUI_TIMEOUT_SECONDS"] = prompt_positive_number(
        "ComfyUI request timeout seconds", config["COMFYUI_TIMEOUT_SECONDS"]
    )
    config["BIND_MODE"] = prompt_bind_mode(config["BIND_MODE"])
    config["DATASET_ROOT"] = prompt("Dataset root scanned by labeler/admin", config["DATASET_ROOT"])
    config["IMAGE_ROOT"] = prompt("Image root searched by export viewer", config["IMAGE_ROOT"])
    config["RUNTIME_ROOT"] = prompt("Endpoint runtime/state root", config["RUNTIME_ROOT"])
    config["INVITE_TOKEN"] = prompt("Labeler invite token", config["INVITE_TOKEN"])
    config["REVIEW_ROUND_SEED"] = prompt("Review round seed", config["REVIEW_ROUND_SEED"])

    print()
    print("Endpoint enablement")
    config["START_LABELER"] = yes_no("Enable labeler", config["START_LABELER"] == "1")
    config["START_EXPORT_VIEWER"] = yes_no("Enable export viewer", config["START_EXPORT_VIEWER"] == "1")
    config["START_IMAGE_GRADER"] = yes_no("Enable image grader API", config["START_IMAGE_GRADER"] == "1")
    config["START_ADAPTER"] = yes_no("Enable image grader admin/playground", config["START_ADAPTER"] == "1")

    print()
    print("Ports")
    config["LABELER_PORT"] = prompt_port("Labeler port", config["LABELER_PORT"])
    config["EXPORT_VIEWER_PORT"] = prompt_port("Export viewer port", config["EXPORT_VIEWER_PORT"])
    config["IMAGE_GRADER_PORT"] = prompt_port("Image grader API port", config["IMAGE_GRADER_PORT"])
    config["ADAPTER_PORT"] = prompt_port("Image grader admin port", config["ADAPTER_PORT"])

    print()
    print("Image grader")
    config["GRADER_DEVICE"] = prompt("Grader device", config["GRADER_DEVICE"])
    config["GRADER_MODELS_ROOT"] = prompt("Grader model root", config["GRADER_MODELS_ROOT"])
    return config


def prompt_grader_dependency_profile(device: str) -> str:
    choices = {
        "1": ("cpu", "CPU-only Torch and grader dependencies"),
        "2": ("cuda13", "CUDA 13 Torch and grader dependencies"),
        "3": ("none", "skip dependency installation"),
    }
    default_index = "2" if str(device).startswith("cuda") else "1"
    print()
    print("Image grader dependencies:")
    for key, (value, description) in choices.items():
        marker = " default" if key == default_index else ""
        print(f"  {key}. {value} - {description}{marker}")
    selected = input(f"Choice [{default_index}]: ").strip() or default_index
    return choices.get(selected, choices[default_index])[0]


def install_grader_dependencies(python_bin: str, profile: str) -> None:
    if profile == "none":
        return
    runtime_requirements = {
        "cpu": ROOT / "image_grader" / "requirements-cpu.txt",
        "cuda13": ROOT / "image_grader" / "requirements-cuda13.txt",
    }
    requirement_paths = [ROOT / "image_grader" / "requirements.txt", runtime_requirements[profile]]
    for requirement_path in requirement_paths:
        if not requirement_path.is_file():
            raise SystemExit(f"Missing grader requirements file: {requirement_path}")
        print(f"Installing grader dependencies from {requirement_path.relative_to(ROOT)}")
        try:
            subprocess.run(
                [python_bin, "-m", "pip", "install", "--requirement", str(requirement_path)],
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SystemExit(f"Image grader dependency installation failed for profile {profile}: {exc}") from exc


def prompt(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def prompt_port(label: str, default: str) -> str:
    while True:
        value = prompt(label, default)
        try:
            port = int(value)
        except ValueError:
            print("Port must be a number.")
            continue
        if 1 <= port <= 65535:
            return str(port)
        print("Port must be between 1 and 65535.")


def prompt_positive_number(label: str, default: str) -> str:
    while True:
        value = prompt(label, default)
        try:
            number = float(value)
        except ValueError:
            print("Value must be a number.")
            continue
        if number > 0:
            return value
        print("Value must be greater than zero.")


def yes_no(label: str, default: bool) -> str:
    suffix = "Y/n" if default else "y/N"
    while True:
        value = input(f"{label} [{suffix}]: ").strip().lower()
        if not value:
            return "1" if default else "0"
        if value in {"y", "yes"}:
            return "1"
        if value in {"n", "no"}:
            return "0"
        print("Answer y or n.")


def prompt_bind_mode(default: str) -> str:
    choices = {
        "1": ("local", "127.0.0.1 only"),
        "2": ("tailscale", "detected Tailscale IPv4"),
        "3": ("all", "0.0.0.0"),
        "4": ("custom", "custom host/IP"),
    }
    default_index = next((key for key, item in choices.items() if item[0] == default), "1")
    print("Bind host:")
    for key, (value, desc) in choices.items():
        marker = " default" if key == default_index else ""
        print(f"  {key}. {value} - {desc}{marker}")
    selected = input(f"Choice [{default_index}]: ").strip() or default_index
    if selected == "4":
        return prompt("Custom bind host/IP", "127.0.0.1")
    return choices.get(selected, choices[default_index])[0]


def write_grader_config(config: dict[str, str]) -> None:
    if not GRADER_EXAMPLE_CONFIG.is_file():
        raise SystemExit(f"Missing grader example config: {GRADER_EXAMPLE_CONFIG}")
    payload = json.loads(GRADER_EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    payload["device"] = config["GRADER_DEVICE"]
    payload["models_root"] = config["GRADER_MODELS_ROOT"]
    payload.setdefault("server", {})
    payload["server"]["allowed_roots"] = [str(resolve_user_path(config["DATASET_ROOT"]))]
    models = payload.get("models", {})
    waifu = models.get("waifu_scorer_v3")
    if isinstance(waifu, dict):
        waifu["clip_model"] = str(Path(config["GRADER_MODELS_ROOT"]) / "openai" / "clip-vit-large-patch14")
    GRADER_LOCAL_CONFIG.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_env(config: dict[str, str]) -> None:
    derived = {
        **config,
        "LABELER_STATE_DIR": str(Path(config["RUNTIME_ROOT"]) / "labeler_state"),
        "EXPORT_STATE_DIR": str(Path(config["RUNTIME_ROOT"]) / "export_viewer_state"),
        "GRADER_STATE_DIR": str(Path(config["RUNTIME_ROOT"]) / "image_grader_state"),
        "ADAPTER_WORK_DIR": str(Path(config["RUNTIME_ROOT"]) / "image_grader_adapter"),
        "GRADER_CONFIG": "image_grader/config.local.json",
    }
    lines = [
        "# Generated by scripts/oss_setup.py",
        "# Re-run bash install.sh to regenerate.",
    ]
    for key in sorted(derived):
        lines.append(f"{key}={shlex.quote(str(derived[key]))}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_runner_env(config: dict[str, str]) -> None:
    comfyui_url = str(config["COMFYUI_URL"]).strip()
    if not comfyui_url:
        raise SystemExit("ComfyUI URL cannot be empty")
    timeout = float(config["COMFYUI_TIMEOUT_SECONDS"])
    if timeout <= 0:
        raise SystemExit("ComfyUI request timeout must be greater than zero")
    values = {
        "RUN_WORKFLOWS_URL": comfyui_url,
        "RUN_WORKFLOWS_ALLOW_INSECURE": "true" if comfyui_url.lower().startswith("http://") else "false",
        "RUN_WORKFLOWS_OUTPUT": "output/run_results.jsonl",
        "RUN_WORKFLOWS_OUTPUT_DIR": "output/images",
        "RUN_WORKFLOWS_TIMEOUT_SECONDS": str(config["COMFYUI_TIMEOUT_SECONDS"]),
    }
    lines = ["# Generated by scripts/oss_setup.py"]
    for key, value in values.items():
        lines.append(f"{key}={shlex.quote(str(value))}")
    RUNNER_ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_user_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def chmod_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


if __name__ == "__main__":
    raise SystemExit(main())
