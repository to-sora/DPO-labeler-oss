from pathlib import Path
import subprocess

from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait


def firefox(downloads: Path):
    options = webdriver.FirefoxOptions()
    options.binary_location = str(Path.home() / "UAT-firefox/firefox")
    options.add_argument("-headless")
    options.add_argument("--remote-allow-system-access")
    options.accept_insecure_certs = True
    options.enable_bidi = True
    options.set_preference("network.dns.disableIPv6", True)
    options.set_preference("network.proxy.type", 0)
    options.set_preference("browser.download.folderList", 2)
    options.set_preference("browser.download.dir", str(downloads.resolve()))
    options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/zip,application/x-ndjson")
    service = Service(str(Path(".local-tool/geckodriver").resolve()),
                      log_output=subprocess.DEVNULL)
    return webdriver.Firefox(options=options, service=service)


def find(driver, id: str):
    return driver.find_element(By.ID, id)


def wait(driver, predicate):
    return WebDriverWait(driver, 25).until(predicate)


def login(driver, reviewer: str = "Firefox-reviewer", token: str = "change-me") -> None:
    wait(driver, lambda d: d.find_elements(By.ID, "nav-imported"))
    if not find(driver, "imported-view").is_displayed():
        wait(driver, lambda d: find(d, "invite-token").is_displayed())
        find(driver, "invite-token").clear()
        find(driver, "invite-token").send_keys(token)
        find(driver, "username").clear()
        find(driver, "username").send_keys(reviewer)
        driver.find_element(By.CSS_SELECTOR, '#auth-form button[type="submit"]').click()
    wait(driver, lambda d: find(d, "imported-view").is_displayed())
    wait(driver, lambda d: find(d, "import-task").is_enabled())


def api(driver, path: str) -> dict:
    return driver.execute_async_script("""
      const done = arguments[arguments.length - 1];
      fetch(arguments[0], {cache:'no-store'}).then(r => r.json()).then(done);
    """, path)


def screenshot(driver, path: Path, width: int, height: int, zoom: float = 1) -> dict:
    driver.browsing_context.set_viewport(driver.current_window_handle,
                                        {"width": width, "height": height})
    driver.set_context("chrome")
    driver.execute_script("gBrowser.selectedBrowser.browsingContext.fullZoom=arguments[0]", zoom)
    driver.set_context("content")
    driver.execute_script("window.scrollTo(0,0)")
    driver.get_full_page_screenshot_as_file(str(path))
    dimensions = driver.execute_script("""return {
      viewport:innerWidth, height:innerHeight, ratio:devicePixelRatio,
      scroll:document.documentElement.scrollWidth,
      broken:[...document.querySelectorAll('#import-images img')].some(i => !i.complete || !i.naturalWidth)
    }""")
    assert dimensions["scroll"] <= dimensions["viewport"], dimensions
    assert not dimensions["broken"], dimensions
    assert dimensions["ratio"] == zoom, dimensions
    if zoom == 1:
        assert dimensions["viewport"] == width, dimensions
    return dimensions
