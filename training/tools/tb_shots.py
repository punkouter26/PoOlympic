"""Screenshot TensorBoard scalar cards (Time Series tab, http://localhost:6006) with the installed Chrome, for the
training review pages (AGENTS.md: 3 charts, 3 explanation levels). Playwright is not a project dependency:

    uv run --with playwright python tools/tb_shots.py <out_dir> <run_filter regex> <tag>[=file_name] ...
"""
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

out, run_filter, tags = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    page = b.new_page(viewport={"width": 1500, "height": 1100}, device_scale_factor=2)
    for spec in tags:
        tag, _, name = spec.partition("=")
        name = name or re.sub(r"\W+", "_", tag)
        url = f"http://localhost:6006/?darkMode=false&runFilter={run_filter}&tagFilter=^{re.escape(tag)}$#timeseries"
        page.goto(url)
        card = page.locator("scalar-card-component, card-view").first
        card.wait_for(timeout=30000)
        page.wait_for_timeout(1500)
        # widen the card (fullsize toggle) when the button exists, so the line is readable
        full = card.locator("button[aria-label*='ull size'], button[aria-label*='ull Size']")
        if full.count():
            full.first.click()
            page.wait_for_timeout(1500)
        card.locator("canvas, svg").first.wait_for(timeout=30000)
        page.wait_for_timeout(2500)
        card.screenshot(path=str(out / f"{name}.png"))
        print("saved", out / f"{name}.png")
    b.close()
