import random
from pathlib import Path

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

from .browser import api, find, login, screenshot, wait


def open_import(driver, base: str, yaml_path: Path) -> str:
    driver.get(base)
    login(driver)
    wait(driver, lambda d: find(d, "import-file").is_enabled())
    driver.find_element(By.CSS_SELECTOR, ".import-upload summary").click()
    find(driver, "import-file").send_keys(str(yaml_path.resolve()))
    wait(driver, lambda d: find(d, "import-yaml").get_attribute("value").startswith("character"))
    find(driver, "import-create").click()
    wait(driver, lambda d: d.find_elements(By.CSS_SELECTOR, ".import-choice"))
    assert all(p.is_displayed() for p in driver.find_elements(By.CSS_SELECTOR, "#import-images details p"))
    return find(driver, "import-task").get_attribute("value")


def vote(driver, task: dict, orders: list[list[int]]) -> None:
    a, b = task["pair"]["image_ids"]
    for d, (dim, order) in enumerate(zip(task["dimensions"], orders)):
        group = driver.find_elements(By.CSS_SELECTOR, ".import-choice")[d]
        if dim in task["pair"]["locked"]:
            assert not group.is_enabled()
            continue
        choice = "a_good" if order.index(a) < order.index(b) else "b_good"
        group.find_element(By.CSS_SELECTOR, f'input[value="{choice}"]').click()
    wait(driver, lambda d: find(d, "import-save").is_enabled())
    find(driver, "import-save").click()
    wait(driver, lambda d: not find(d, "import-refresh").get_attribute("disabled"))
    assert all(p.is_displayed() for p in driver.find_elements(By.CSS_SELECTOR, "#import-images details p"))


def finish(driver, task_id: str, output: Path) -> dict:
    route = f"/api/v1/imported/tasks/{task_id}"
    orders = [random.Random(d).sample(range(10), 10) for d in range(5)]
    locked_count = 0
    while True:
        task = api(driver, route)["data"]
        if task["complete"]:
            return {"task": task, "locked_dimensions_seen": locked_count}
        assert task["comparisons"] < 45
        if task["pair"]["locked"]:
            locked_count += len(task["pair"]["locked"])
            if not (output / "inferred.png").exists():
                screenshot(driver, output / "inferred.png", 1440, 1000)
        vote(driver, task, orders)
        if task["comparisons"] == 0:
            driver.find_element(By.CSS_SELECTOR, '.import-results summary').click()
            find(driver, "import-export").click()
            wait(driver, lambda d: not find(d, "import-refresh").get_attribute("disabled"))
            driver.refresh()
            wait(driver, lambda d: find(d, "imported-view").is_displayed())
            wait(driver, lambda d: find(d, "import-task").is_enabled())
            Select(find(driver, "import-task")).select_by_value(task_id)
            wait(driver, lambda d: not find(d, "import-work").get_attribute("hidden"))
