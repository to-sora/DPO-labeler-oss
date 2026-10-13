import argparse
import json
import random
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

from .browser import api, find, firefox, screenshot, wait
from .fixtures import legacy, prepare
from .review import finish, open_task, vote
from .server import server


def check_reviews(base: str, output: Path) -> dict:
    first, second = firefox(output), None
    try:
        second = firefox(output)
        task_id = open_task(first, base + "review/example", "alice")
        assert open_task(second, base, "bob") == task_id
        route = f"/api/v1/imported/tasks/{task_id}"
        tasks = [api(driver, route)["data"] for driver in (first, second)]
        assert tasks[0]["pair"]["comparison_id"] != tasks[1]["pair"]["comparison_id"]
        assert tasks[0]["pair"]["image_ids"] == tasks[1]["pair"]["image_ids"]
        assert not first.find_elements(By.CSS_SELECTOR, '#import-create, #import-yaml, #import-file, #import-reviewer')
        status = first.execute_async_script("""
          const done = arguments[arguments.length - 1];
          fetch('/api/v1/imported/tasks', {method:'POST',
            headers:{'Content-Type':'application/json'}, body:JSON.stringify({yaml:'task-name: forbidden'})})
            .then(response => done(response.status));
        """)
        assert status == 405
        layouts = {name: screenshot(first, output / f"{name}.png", width, height)
                   for name, width, height in [("desktop", 1440, 1000), ("mobile", 390, 844)]}
        screenshot(first, output / "desktop.png", 1440, 1000)
        first.execute_script("""
          const original = window.fetch; let drop = true;
          window.fetch = async (...args) => {
            const response = await original(...args);
            if (drop && args[0].endsWith('/comparisons') && response.ok) {
              drop = false; throw new TypeError('Simulated lost response');
            }
            return response;
          };
        """)
        orders = [random.Random(d).sample(range(10), 10) for d in range(5)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(vote, first, tasks[0], orders),
                       pool.submit(vote, second, tasks[1], [list(reversed(order)) for order in orders])]
            for future in futures:
                future.result()
        wait(first, lambda d: find(d, "import-save").text == "Retry saved comparison")
        assert find(first, "import-message").is_displayed()
        open_task(first, base, "alice")
        assert api(first, route)["data"]["comparisons"] == 1
        assert api(second, route)["data"]["comparisons"] == 1
        assert first.execute_script("return localStorage.getItem('imported-pending-' + JSON.stringify(['alice', arguments[0]]))", task_id) is None
        result = finish(first, task_id, output)
        assert result["task"]["complete"]
        other = api(second, route)["data"]
        assert other["comparisons"] == 1 and not other["complete"]
        first.find_element(By.CSS_SELECTOR, '.import-results summary').click()
        for selected in ("", "0"):
            filename = "all-dimensions.zip" if selected == "" else "dimension-0.zip"
            Select(find(first, "import-dimension")).select_by_value(selected)
            find(first, "import-export").click()
            wait(first, lambda _: (output / filename).is_file())
            wait(first, lambda d: find(d, "import-refresh").is_enabled())
        with zipfile.ZipFile(output / "all-dimensions.zip") as archive:
            assert json.loads(archive.read("task.json"))["reviewer_username"] == "alice"
            for d in range(5):
                ranking = json.loads(archive.read(f"dimension-{d}/ranking.json"))
                assert ranking["certified"]
                for j in range(1, 5):
                    assert set(ranking["cutoffs"][f"{j / 5:.1f}"]["image_ids"]) == set(orders[d][:j * 2])
                events = [json.loads(line) for line in archive.read(f"dimension-{d}/label_events.jsonl").splitlines()]
                assert {event["reviewer_username"] for event in events} == {"alice"}
        find(first, "nav-imported").click()
        wait(first, lambda d: d.find_elements(By.CSS_SELECTOR, '#task-groups input[type="checkbox"]'))
        first.find_element(By.CSS_SELECTOR, '#task-groups input[type="checkbox"]').click()
        find(first, "start-review").click()
        wait(first, lambda d: len(d.find_elements(By.CSS_SELECTOR, '#pair-images img')) == 2)
        first.find_element(By.CSS_SELECTOR, '[data-decision="a_good"]').click()
        wait(first, lambda d: find(d, "pending-count").text == "0")
        find(first, "nav-export").click()
        find(first, "export-download").click()
        wait(first, lambda _: (output / "dpo_pairs.jsonl").is_file())
        assert json.loads((output / "dpo_pairs.jsonl").read_text(encoding="utf-8"))["strict_dpo"]
        return {"task_id": task_id, "alice_comparisons": result["task"]["comparisons"],
                "bob_comparisons": 1, "independent_rankings": True, "lost_response_retry": True,
                "startup_only_imports": True, "exports": True, "standard_review": True, "layouts": layouts}
    finally:
        first.quit()
        if second is not None:
            second.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated Firefox checks with local image fixtures")
    parser.add_argument("--output-dir", type=Path, help="Keep screenshots and exports in this directory")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="dpo-imported-uat-") as directory:
        root = Path(directory)
        output = args.output_dir.resolve() if args.output_dir else root / "artifacts"
        output.mkdir(parents=True, exist_ok=True)
        images = root / "images"
        manifest = prepare(root, images)
        legacy(root / "datasets", images)
        with server(root, manifest, images) as base:
            result = check_reviews(base, output)
        with server(root, manifest, images) as base:
            driver = firefox(output)
            try:
                assert open_task(driver, base, "bob") == result["task_id"]
                restored = api(driver, f"/api/v1/imported/tasks/{result['task_id']}")["data"]
                assert restored["comparisons"] == 1
                result["restart_preserved_progress"] = True
            finally:
                driver.quit()
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
