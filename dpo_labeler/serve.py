import argparse
from pathlib import Path

from .backend.app import DpoLabelerApp
from .backend.imported.handler import ImportedRequestHandler
from .backend.imported.service import ImportedTasks
from .tls import context
from .ports import addresses, configuration, whitelist
from .runtime import run


def main() -> None:
    config = configuration()
    parser = argparse.ArgumentParser(description="DPO labeler with imported-image review (HTTPS)")
    parser.add_argument("--dataset-root", default="output")
    parser.add_argument("--state-dir", default="output/labeler")
    parser.add_argument("--image-root", action="append", default=[])
    parser.add_argument("--exclude-dir", action="append", default=[])
    parser.add_argument("--policy", choices=["public", "local", "standard"], default=config["policy"])
    parser.add_argument("--port", type=int, default=config["port"])
    args = parser.parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    app = DpoLabelerApp(dataset_root=dataset_root, state_dir=args.state_dir,
                        invite_token="reachable-device", cookie_secure=True,
                        image_roots=args.image_root, exclude_dirs=args.exclude_dir)
    frontend = Path(__file__).parent / "frontend"
    handler = type("ConfiguredImportedHandler", (ImportedRequestHandler,), {
        "app": app, "frontend_dir": frontend,
        "imported": ImportedTasks(Path(args.state_dir)),
    })
    tls = context(Path(__file__).parent / "data" / "tls")
    try:
        run(addresses(args.policy), args.port, handler, tls, whitelist(config))
    finally:
        app.close()


if __name__ == "__main__":
    main()
