import re
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def server(root: Path, manifest: Path, images: Path):
    log_path = root / "server.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen([
            sys.executable, "-u", "-m", "dpo_labeler.backend.server",
            "--dataset-root", str(root / "datasets"), "--state-dir", str(root / "state"),
            "--image-root", str(images), "--import-yaml", str(manifest),
            "--host", "127.0.0.1", "--port", "0", "--invite-token", "change-me",
        ], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 20
            while True:
                output = log_path.read_text(encoding="utf-8")
                match = re.search(r"listening on (http://127\.0\.0\.1:\d+/)", output)
                if match:
                    yield match[1]
                    break
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError(f"Fixture server did not start:\n{output}")
                time.sleep(0.1)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
