import json
import socket
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote

BASE = "http://127.0.0.1:18789/api/v1/imported/tasks"


def request(path: str = "", body: dict | None = None, cookie: str = "") -> dict:
    req = urllib.request.Request(BASE + quote(path), data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", "Cookie": cookie})
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.load(response)["data"]


def main() -> None:
    root = Path("fan-out/imported-image-uat")
    source = (root / "import.yaml").read_text()
    login = urllib.request.Request(BASE.replace("/imported/tasks", "/session/start"),
        data=json.dumps({"invite_token": "change-me", "reviewer_username": "http-reviewer",
                         "client_instance_id": "http-client"}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(login, timeout=15) as response:
        cookie = response.headers["Set-Cookie"].split(";", 1)[0]
    task = request(body={"yaml": source}, cookie=cookie)
    task_id = task["task_id"]
    payload = {"comparison_id": task["pair"]["comparison_id"],
               "reviewer_username": "http-reviewer", "client_instance_id": "http-client",
               "choices": dict.fromkeys(task["dimensions"], "a_good")}
    with socket.create_connection(("127.0.0.1", 18789), timeout=5) as speculative:
        # A browser can open a connection before sending an HTTP request.
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: request(f"/{task_id}/comparisons", payload, cookie), range(8)))
    assert sum(not result["replayed"] for result in results) == 1
    current = request(f"/{task_id}", cookie=cookie)
    assert current["comparisons"] == 1 and sum(current["exposure"]) == 2
    changed = {**payload, "choices": dict.fromkeys(task["dimensions"], "b_good")}
    try:
        request(f"/{task_id}/comparisons", changed, cookie)
    except HTTPError as exc:
        assert exc.code == 409
    else:
        raise AssertionError("Conflicting retry accepted")
    result = {"concurrent_submissions": 8, "counted_comparisons": 1,
              "events": len(results[0]["events"]), "exposure": sum(current["exposure"]),
              "idle_http_connection_did_not_block": True, "task_id": task_id, "payload": payload}
    (root / "http-result.json").write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("payload", "task_id")}))


if __name__ == "__main__":
    main()
