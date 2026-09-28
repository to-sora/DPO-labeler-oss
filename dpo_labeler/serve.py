import sys
from pathlib import Path

from .backend.server import build_arg_parser, main as serve
from .ports import configuration


def main() -> None:
    defaults = [
        "--dataset-root", "output",
        "--state-dir", "output/labeler",
        "--host", "0.0.0.0",
        "--port", str(configuration()["port"]),
        "--invite-token", "change-me",
    ]
    arguments = [*defaults, *sys.argv[1:]]
    args = build_arg_parser().parse_args(arguments)
    Path(args.dataset_root).mkdir(parents=True, exist_ok=True)
    serve(arguments)


if __name__ == "__main__":
    main()
