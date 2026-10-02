import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def main() -> None:
    root = Path("fan-out/imported-image-uat")
    code = "import socket,threading; s=socket.socket(); s.bind(('0.0.0.0',0)); s.listen(); print(s.getsockname()[1],flush=True); threading.Event().wait()"
    occupant = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    service = None
    try:
        port = int(occupant.stdout.readline())
        args = ["dpo_labeler/start.sh", "--port", str(port),
                "--state-dir", "output/imported-launcher/state", "--dataset-root", "output/imported-launcher/datasets"]
        blocked = subprocess.run(args, capture_output=True, text=True, timeout=10)
        assert blocked.returncode == 1 and f"PID {occupant.pid}" in blocked.stdout
        service = subprocess.Popen(args + ["--force"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if service.poll() is not None:
                raise AssertionError(f"Launcher exited during startup: {service.stderr.read()}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/config", timeout=1) as response:
                    assert json.load(response)["ok"]
                    break
            except (OSError, TimeoutError):
                time.sleep(.1)
        else:
            raise AssertionError("Forced launcher did not start")
        assert occupant.wait(timeout=5) != 0
        subprocess.run(["dpo_labeler/start.sh", "--stop"], check=True, timeout=15)
        assert service.wait(timeout=5) == 0
        with socket.socket() as probe:
            assert probe.connect_ex(("127.0.0.1", port)) != 0
        result = {"occupied_port_reports_pid":True, "force_replaces_owned_fixture":True,
                  "stop_exits_cleanly":True}
        (root / "launcher-result.json").write_text(json.dumps(result))
        print(json.dumps(result))
    finally:
        for process in (service, occupant):
            if process and process.poll() is None:
                process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
