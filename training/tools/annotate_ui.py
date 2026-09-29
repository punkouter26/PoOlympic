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
    "fx_hud_before_after": {
        "title": "Broadcast FX (features 2, 3, 5, 6, 7, 10 + world records) — before vs after",
        "panels": [
            ("BEFORE · Terminal Velocity live", UI / "before" / "fx_race_live.png", FRAME, []),
            ("BEFORE · result card", UI / "before" / "fx_race_result.png", FRAME, []),
            ("AFTER · Terminal Velocity live", UI / "after" / "fx_race_live_1.png", FRAME,
             [(1, (487, 656, 647, 993)), (2, (440, 1683, 940, 1967)), (3, (24, 2000, 940, 2104)), (4, (0, 1173, 960, 1627))]),
            ("AFTER · Iron Pedestal, hot close-up", UI / "after" / "fx_pedestal_hot_1.png", FRAME,
             [(5, (387, 840, 587, 1613)), (6, (360, 1590, 560, 1650)), (7, (600, 1060, 960, 1600)), (8, (24, 2000, 940, 2104))]),
            ("AFTER · result card", UI / "after" / "fx_pedestal_result.png", FRAME,
             [(9, (77, 1573, 887, 1780)), (10, (77, 1787, 887, 1847))]),
        ],
        "legend": [
            "1  NEW CONF column: brain confidence (PPO critic → P(still on its feet in 2 s)), sparkline + %, per athlete",
            "2  NEW stats card (PrimeTween pop): speed + peak, power (W), cadence, ground contact, joint load + joint, confidence",
            "3  Ticker gains telemetry calls: a new heat top speed (after the start phase); near falls and saves in panel 8",
            "4  NEW foot-strike dust from MuJoCo contacts (Shuriken pool); body slams add a dust ring + shockwave",
            "5  NEW Cinemachine hot close-up: the director cuts to the athlete in danger (TensionMeter), hand-held noise",
            "6  NEW balance overlay: support polygon + centre-of-mass ring (green safe, amber edge, red off balance)",
            "7  NEW VFX Graph sparks where a cube strikes (impulse shake + 70 ms hit-stop on big hits); crowd audio follows tension",
            "8  NEW near-fall / save calls: 'M7 is wobbling — brain confidence 16%!' → 'What a save by M7!'",
            "9  NEW WORLD RECORDS panel: top 3 (holder, body, date), new record highlighted and pulsing, else the record to beat",
            "10 NEW heat bests from telemetry: top speed, peak power, closest call (lowest confidence of a survivor)",
        ],
    },
    "fx_menu_before_after": {
        "title": "Main menu — world records (before vs after)",
        "panels": [
            ("BEFORE", UI / "before" / "fx_menu.png", None, []),
            ("AFTER · menu", UI / "after" / "fx_menu.png", None, [(1, (260, 1133, 593, 1196)), (2, (53, 2380, 727, 2420))]),
            ("AFTER · WORLD RECORDS board", UI / "after" / "fx_menu_records.png", None,
             [(3, (53, 73, 913, 167)), (4, (53, 284, 909, 687)), (5, (53, 700, 909, 1890))]),
        ],
        "legend": [
            "1  NEW WORLD RECORDS button next to the gauntlet toggle",
            "2  NEW world record of the selected event under its rules",
            "3  NEW records board (full-screen overlay, CLOSE)",
            "4  Tap an event: its all-time top 5 (mark, holder · body, date)",
            "5  One card per playable event: world record + holder, or 'no mark yet'",
        ],
    },
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
