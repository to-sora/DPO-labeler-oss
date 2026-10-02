import json
import random
import zipfile
from pathlib import Path

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

from .browser import firefox, find, login, screenshot, wait


def main() -> None:
    root = Path("fan-out/imported-image-uat")
    exports = root / "exports"
    exports.mkdir(exist_ok=True)
    task_id = (root / "task-id.txt").read_text()
    driver = firefox(exports)
    try:
        driver.get("http://127.0.0.1:18789/")
        login(driver)
        wait(driver, lambda d: find(d, "import-task").is_enabled())
        Select(find(driver, "import-task")).select_by_value(task_id)
        wait(driver, lambda d: find(d, "import-complete").is_displayed())
        driver.find_element(By.CSS_SELECTOR, '.import-results summary').click()
        for value in ['', '0', '1', '2', '3', '4']:
            name = 'all-dimensions.zip' if value == '' else f'dimension-{value}.zip'
            (exports / name).unlink(missing_ok=True)
            Select(find(driver, "import-dimension")).select_by_value(value)
            find(driver, "import-export").click()
            wait(driver, lambda d: find(d, "import-refresh").is_enabled())
            wait(driver, lambda _: (exports / name).exists())
        screenshot(driver, root / "complete.png",1440,1000)
    finally:
        driver.quit()
    event_count = human_count = 0
    with zipfile.ZipFile(exports / "all-dimensions.zip") as archive:
        assert json.loads(archive.read('task.json'))['task_id'] == task_id
        for d in range(5):
            ranking = json.loads(archive.read(f'dimension-{d}/ranking.json'))
            assert ranking['certified']
            expected = random.Random(d).sample(range(10),10)
            for j in range(1,5):
                cutoff = ranking['cutoffs'][f'{j/5:.1f}']
                assert cutoff['certified'] and set(cutoff['image_ids']) == set(expected[:j*2])
            events = archive.read(f'dimension-{d}/label_events.jsonl').splitlines()
            pairs = archive.read(f'dimension-{d}/dpo_pairs.jsonl').splitlines()
            event_count += len(events)
            human_count += len(pairs)
    result = {'events':event_count,'human_dpo_pairs':human_count,'certified_cutoffs':20,
              'individual_dimension_downloads':5}
    (root / 'export-result.json').write_text(json.dumps(result))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
