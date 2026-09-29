"""Annotated before/after UI screenshots (AGENTS.md: every UI change gets one, changes marked).

Panels side by side (optionally cropped to the 9:16 camera frame), numbered red boxes on the panels, a legend below.
Specs live in this file (one per comparison); coordinates are pixels of the source screenshots.

Usage: uv run python tools/annotate_ui.py [name ...]      (default: all)
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

UI = Path(__file__).resolve().parents[2] / "parity" / "ui"
D3 = UI.parent / "d3"
FRAME = (0, 476, 960, 2181)          # the letterboxed 9:16 camera rect in the 960 x 2658 Game view captures
FONT = "C:/Windows/Fonts/segoeui.ttf"
FONT_B = "C:/Windows/Fonts/segoeuib.ttf"

SPECS = {
    "d3_hud_before_after": {
        "title": "D3 broadcast HUD — before (IMGUI placeholder) vs after (UI Toolkit BroadcastHud + BroadcastDirector)",
        "panels": [
            ("BEFORE · Terminal Velocity (IMGUI)", UI / "before" / "race_hud.png", FRAME, []),
            ("AFTER · live", UI / "after" / "race_hud_live.png", FRAME,
             [(1, (24, 499, 938, 605)), (2, (27, 649, 649, 1004)), (3, (566, 649, 649, 1004)), (4, (24, 1995, 938, 2099)),
              (5, (273, 2112, 560, 2158)), (6, (0, 1050, 960, 1950))]),
            ("AFTER · betting slip", D3 / "steeple_1_slip.png", FRAME, [(7, (44, 850, 916, 1662))]),
            ("AFTER · result card", D3 / "steeple_3_live.png", FRAME, [(8, (44, 850, 916, 1380))]),
        ],
        "legend": [
            "1  Top bar: title + event rules line, clock + leader distance (styled panels, Inter font)",
            "2  Standings: place, body chip (M = MATT blue, Z = zombie green), live result",
            "3  ODDS column: win odds from athlete traits (Plackett-Luce fit on 600 CPU heats, tools/fit_odds.py)",
            "4  Play-by-play ticker: start, lead changes (4 s cooldown), athletes out, winner, records, bets",
            "5  Wallet + your bet (virtual coins)",
            "6  Tracking camera (BroadcastDirector): follows the leader; trackside / head-on / high-wide cuts",
            "7  NEW betting slip: each heat waits 12 s for a 10-coin bet (traits + odds per athlete)",
            "8  NEW result card: podium, bet payout, event record, gauntlet points + Next event",
        ],
    },
    "menu_gauntlet_before_after": {
        "title": "Main menu — before vs after (gauntlet builder, wallet, Event 13)",
        "panels": [
            ("BEFORE", UI / "before" / "menu.png", None, []),
            ("AFTER · gauntlet mode", UI / "after" / "menu_gauntlet.png", None,
             [(1, (645, 1176, 912, 1246)), (2, (785, 1365, 875, 2008)), (3, (60, 1925, 900, 2015)),
              (4, (48, 2385, 912, 2525)), (5, (130, 2530, 830, 2615))]),
        ],
        "legend": [
            "1  NEW GAUNTLET toggle (shows the number of events in the series)",
            "2  Tap events to add / remove them: numbered order badges, cyan outline",
            "3  NEW event 13 Steeplechase Jog (flight brain) in the list",
            "4  PLAY becomes PLAY GAUNTLET (n): one heat per event, 10-8-6-5-4-3-2-1 points per lane",
            "5  Gauntlet order + points rule; wallet coins and the last gauntlet's podium",
        ],
    },
}


def render(name: str) -> Path:
    spec = SPECS[name]
    font = ImageFont.truetype(FONT, 26)
    bold = ImageFont.truetype(FONT_B, 30)
    tag = ImageFont.truetype(FONT_B, 34)
    panels = []
    for label, path, crop, boxes in spec["panels"]:
        im = Image.open(path).convert("RGB")
        off = (0, 0)
        if crop:
            im = im.crop(crop)
            off = (crop[0], crop[1])
        d = ImageDraw.Draw(im)
        for n, (x0, y0, x1, y1) in boxes:
            b = (x0 - off[0], y0 - off[1], x1 - off[0], y1 - off[1])
            d.rectangle(b, outline=(255, 40, 40), width=6)
            d.ellipse((b[0] - 4, b[1] - 4, b[0] + 48, b[1] + 48), fill=(255, 40, 40))
            d.text((b[0] + 22, b[1] + 22), str(n), fill="white", font=tag, anchor="mm")
        panels.append((label, im))
    h = max(im.height for _, im in panels)
    pad, head = 24, 110
    width = sum(im.width for _, im in panels) + pad * (len(panels) + 1)
    legend_h = 44 * len(spec["legend"]) + 40
    out = Image.new("RGB", (width, head + h + legend_h + pad), (18, 22, 30))
    d = ImageDraw.Draw(out)
    d.text((pad, 18), spec["title"], fill=(255, 216, 74), font=bold)
    x = pad
    for label, im in panels:
        d.text((x, 66), label, fill="white", font=font)
        out.paste(im, (x, head))
        x += im.width + pad
    y = head + h + 24
    for line in spec["legend"]:
        d.text((pad, y), line, fill=(230, 234, 242), font=font)
        y += 44
    dst = UI / f"{name}.png"
    out.save(dst)
    return dst


if __name__ == "__main__":
    for n in sys.argv[1:] or list(SPECS):
        print(render(n))
