import argparse
import signal
import subprocess
import sys
from pathlib import Path

from .ports import configuration
from .processes import listeners, stop_owned, terminate


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--port", type=int, default=configuration()["port"])
    args, remaining = parser.parse_known_args()
    directory = Path(__file__).parent / "data" / "run"
    directory.mkdir(parents=True, exist_ok=True)
    if args.stop:
        stop_owned(directory)
        return
    if "--help" not in remaining:
        occupied = listeners(args.port)
        for pid, name in occupied.items():
            print(f"Port {args.port}: PID {pid} ({name}) / 連接埠已佔用", flush=True)
        if occupied and not args.force:
            raise SystemExit(1)
        for pid in occupied:
            terminate(pid)
    process = subprocess.Popen([sys.executable, "-m", "dpo_labeler.serve",
                                "--port", str(args.port), *remaining], start_new_session=True)
    file = directory / f"{process.pid}.pid"
    file.write_text(str(process.pid), encoding="utf-8")
    def stop(*_) -> None:
        terminate(process.pid, group=True)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        raise SystemExit(process.wait())
    finally:
        if process.poll() is None:
            stop()
        file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
