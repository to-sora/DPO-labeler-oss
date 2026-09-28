import json
from pathlib import Path

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

from .browser import firefox, find, login, screenshot, wait


def main() -> None:
    root = Path("fan-out/imported-image-uat")
    partial = json.loads((root / "http-result.json").read_text())["task_id"]
    driver = firefox(root)
    try:
        driver.get("http://127.0.0.1:18789/")
        login(driver)
        wait(driver, lambda d: find(d, "import-task").is_enabled())
        Select(find(driver, "import-task")).select_by_value(partial)
        wait(driver, lambda d: len(d.find_elements(By.CSS_SELECTOR, "#import-images img")) == 2)
        driver.find_element(By.CSS_SELECTOR, ".import-image-button").click()
        assert find(driver, "import-zoom").get_attribute("open")
        find(driver, "import-close").click()
        find(driver, "logout-button").click()
        wait(driver, lambda d: find(d, "invite-token").is_displayed())
        login(driver)
        find(driver, "nav-imported").click()
        wait(driver, lambda d: d.find_elements(By.CSS_SELECTOR, '#task-groups input[type="checkbox"]'))
        driver.find_element(By.CSS_SELECTOR, '#task-groups input[type="checkbox"]').click()
        find(driver, "start-review").click()
        wait(driver, lambda d: len(d.find_elements(By.CSS_SELECTOR, '#pair-images img')) == 2)
        screenshot(driver, root / "legacy-review.png", 1440, 1000)
        driver.find_element(By.CSS_SELECTOR, '[data-decision="a_good"]').click()
        wait(driver, lambda d: find(d, "pending-count").text == '0')
        find(driver, "nav-export").click()
        find(driver, "export-download").click()
        wait(driver, lambda _: (root / "dpo_pairs.jsonl").exists())
        pair = json.loads((root / "dpo_pairs.jsonl").read_text().strip())
        assert pair["strict_dpo"] and pair["decision"] == "a_good"
        (root / "legacy-result.json").write_text(json.dumps({"navigation":True, "invite_token_login":True,
             "image_zoom":True, "strict_dpo":True, "legacy_rows":1}))
        print("Legacy navigation, invite-token login, image zoom and strict DPO export passed.")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
