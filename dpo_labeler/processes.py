import os
import re
import signal
import subprocess
import time
from pathlib import Path


def listeners(port: int) -> dict[int, str]:
    listing = subprocess.check_output(
        ["ss", "-4", "-H", "-ltnp", f"sport = :{port}"], text=True)
    found = {int(pid): name for name, pid in re.findall(r'\("([^"]+)",pid=(\d+)', listing)}
    if listing.strip() and not found:
        raise RuntimeError(f"Port {port} occupied; process details unavailable to this account")
    return found


def terminate(pid: int, group: bool = False) -> None:
    def send(sig: int) -> None:
        try:
            os.killpg(pid, sig) if group else os.kill(pid, sig)
        except ProcessLookupError:
            pass
    send(signal.SIGTERM)
    for _ in range(50):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            if stat.rsplit(")", 1)[1].split()[0] == "Z":
                return
        except FileNotFoundError:
            return
        time.sleep(0.1)
    send(signal.SIGKILL)


def stop_owned(directory: Path) -> None:
    for file in directory.glob("*.pid"):
        try:
            pid = int(file.read_text())
            command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if b"dpo_labeler.serve" in command:
                terminate(pid, group=True)
        except (FileNotFoundError, ValueError):
            pass
        file.unlink(missing_ok=True)
