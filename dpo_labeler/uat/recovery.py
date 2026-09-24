import json
from pathlib import Path

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

from .browser import api, firefox, find, screenshot, wait


def main() -> None:
    root = Path("fan-out/imported-image-uat")
    task_id = json.loads((root / "http-result.json").read_text())["task_id"]
    driver = firefox(root)
    try:
        def open_task() -> None:
            driver.get("https://127.0.0.1:18789/")
            wait(driver, lambda d: d.find_elements(By.ID, "nav-imported"))
            find(driver, "nav-imported").click()
            wait(driver, lambda d: find(d, "import-task").is_enabled())
            find(driver, "import-reviewer").clear()
            find(driver, "import-reviewer").send_keys("recovery-reviewer")
            Select(find(driver, "import-task")).select_by_value(task_id)
            wait(driver, lambda d: d.find_elements(By.CSS_SELECTOR, ".import-choice"))
        open_task()
        before = api(driver, f"/api/v1/imported/tasks/{task_id}")["data"]["comparisons"]
        driver.execute_script("""
          const original=window.fetch; let drop=true;
          window.fetch=async (...args) => {
            const result=await original(...args);
            if(drop && args[0].endsWith('/comparisons') && result.ok) {
              drop=false; throw new TypeError('Simulated lost response');
            }
            return result;
          };
        """)
        for radio in driver.find_elements(By.CSS_SELECTOR, '.import-choice:not(:disabled) input[value="a_good"]'):
            radio.click()
        wait(driver, lambda d: find(d, "import-save").is_enabled())
        find(driver, "import-save").click()
        wait(driver, lambda d: find(d, "import-save").text == "Retry saved comparison")
        assert find(driver, "import-message").is_displayed()
        screenshot(driver, root / "retry.png", 390,844)
        open_task()
        after = api(driver, f"/api/v1/imported/tasks/{task_id}")["data"]
        assert after["comparisons"] == before + 1
        assert driver.execute_script('return localStorage.getItem(arguments[0])', f'imported-pending-{task_id}') is None
        driver.find_element(By.CSS_SELECTOR, '#import-manage > summary').click()
        driver.find_element(By.CSS_SELECTOR, '.import-upload summary').click()
        find(driver, "import-yaml").send_keys('task-name: missing fields')
        find(driver, "import-create").click()
        wait(driver, lambda d: 'must be' in find(d, "import-message").text)
        assert find(driver, "import-message").is_displayed()
        print('Lost-response reload recovery and visible import errors passed.')
        (root / "recovery-result.json").write_text(json.dumps({"lost_response_reload":True,"visible_errors":True}))
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
