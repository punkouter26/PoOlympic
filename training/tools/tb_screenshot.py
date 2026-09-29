"""AGENTS.md rule (30+ min runs): screenshot TensorBoard charts for review. Drives the installed Edge (Playwright,
headless) against the running TensorBoard, filters the runs and the tags, and saves the chart area.

Usage: uv run --with playwright python tools/tb_screenshot.py <run regex> <tag regex> <out.png> [--url http://localhost:6006]
e.g.   uv run --with playwright python tools/tb_screenshot.py "crawl_v3" "Episode_Reward/progress" ../parity/tensorboard/progress.png
"""

from __future__ import annotations

import argparse

from playwright.sync_api import sync_playwright


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("tags")
    ap.add_argument("out")
    ap.add_argument("--url", default="http://localhost:6006/")
    ap.add_argument("--smoothing", default="0.6")
    a = ap.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 950})
        page.goto(a.url + "#timeseries")
        runs = page.locator("input[placeholder*='Filter runs']")
        runs.wait_for(timeout=30000)
        page.wait_for_timeout(3000)
        runs.fill(a.runs)
        runs.press("Enter")
        tags = page.locator("input[placeholder*='Filter tags']")
        tags.fill(a.tags)
        tags.press("Escape")
        page.mouse.click(5, 900)                              # drop the tag-suggestion overlay
        page.wait_for_timeout(4000)
        card = page.locator("scalar-card, card-view").first
        (card if card.count() else page).screenshot(path=a.out)
        browser.close()
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
