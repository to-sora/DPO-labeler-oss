import json
from pathlib import Path

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

from .browser import api, find, firefox, screenshot, wait
from .review import finish, open_import


def main() -> None:
    output = Path("fan-out/imported-image-uat")
    driver = firefox(output)
    try:
        task_id = open_import(driver, "https://127.0.0.1:18789/", output / "import.yaml")
        (output / "task-id.txt").write_text(task_id)
        driver.find_element(By.CSS_SELECTOR, '#import-manage > summary').click()
        find(driver, "import-filter").send_keys("no-such-character")
        assert len(Select(find(driver, "import-task")).options) == 1
        find(driver, "import-filter").clear()
        find(driver, "import-filter").send_keys("/tmp/dpo-import-uat")
        assert any(o.get_attribute('value') == task_id for o in Select(find(driver, "import-task")).options)
        Select(find(driver, "import-task")).select_by_value(task_id)
        wait(driver, lambda d: len(d.find_elements(By.CSS_SELECTOR, ".import-choice")) == 5)
        wait(driver, lambda d: all(i.get_attribute("complete") for i in d.find_elements(By.CSS_SELECTOR, "#import-images img")))
        assert not find(driver, "import-save").is_enabled()
        sizes = {}
        for name, w, h in [("desktop",1440,1000), ("mobile",390,844),
                           ("landscape",844,390), ("large",1920,1080)]:
            sizes[name] = screenshot(driver, output / f"{name}.png", w, h)
        sizes["zoom"] = screenshot(driver, output / "zoom.png", 1440, 1000, zoom=2)
        screenshot(driver, output / "desktop.png", 1440,1000)
        driver.find_element(By.CSS_SELECTOR, '.import-results summary').click()
        find(driver, "import-export").click()
        wait(driver, lambda d: find(d, "import-refresh").is_enabled())
        driver.find_element(By.CSS_SELECTOR, '.import-results summary').click()
        result = finish(driver, task_id, output)
        screenshot(driver, output / "complete.png", 1440,1000)
        task = result['task']
        summary = {'task_id': task_id, 'comparisons': task['comparisons'],
                   'dimensions': len(task['dimensions']), 'images': len(task['images']),
                   'complete': task['complete'], 'max_exposure': max(task['exposure']),
                   'inferred': result['locked_dimensions_seen'], 'layouts': sizes}
        (output / "result.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"comparisons":result["task"]["comparisons"], "layouts":sizes}))
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
