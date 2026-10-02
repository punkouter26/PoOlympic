"""Pick a walking brain by event heats, not by the G1 drill (2026-10-01 lesson: rs_v8 passed the squat drill and had 0
finishers in the Deep Squat event). Runs the same CPU heats of every Rung 2 event (mixed MATT / zombie / GRANDMA lineup,
seeds 7000+, as tools/fit_odds.py) once per candidate brain for one body and reports that body's lanes: mean place,
finish / fall / DQ counts and the event's own mark.

Usage: uv run python tools/compare_brain_events.py <body> <brain.onnx> [<brain.onnx> ...] [--heats 6] [--workers 6]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

EVENTS = (5, 9, 10, 11, 12, 13, 19, 22)


def run_one(job: tuple) -> dict:
    event, seed, body, brain = job
    import fit_odds as F
    from poolympic.events import crab, gauntlet, slalom, track, turntable
    _, kind, tag = F.EVENTS[event]
    scene, layout = F.A / f"scene_{tag}_{F.lineup_of(seed)}.xml", F.A / f"{tag}_{F.lineup_of(seed)}_layout.json"
    brains = dict(F.R2, **{body: Path(brain)})
    if kind.startswith("track:"):
        mode = kind.split(":")[1]
        if mode == "steeple":
            brains["matt"] = F.BR / "r2f_v3_it100.onnx"
        res = track.run_race(brains["matt"], mode, seed, scene=scene, layout_path=layout, brains=brains)
    else:
        mod = {"gauntlet": gauntlet, "crab": crab, "slalom": slalom, "turntable": turntable}[kind]
        res = mod.run_heat(brains["matt"], seed, scene=scene, layout_path=layout, brains=brains)
    bodies = [l.get("body", "matt") for l in json.loads(layout.read_text())["lanes"]]
    rows = []
    for l in res.lanes:
        if bodies[l.lane] != body:
            continue
        mark = next((getattr(l, k) for k in ("score_s", "score", "finish_s", "total_s", "gap_m") if getattr(l, k, None) is not None), None)
        rows.append({"place": l.place, "status": getattr(l, "status", "") or getattr(l, "reason", "") or "ok",
                     "mark": mark, "peak": getattr(l, "peak_mps", None)})
    return {"event": event, "seed": seed, "brain": Path(brain).name, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("body")
    ap.add_argument("brains", nargs="+")
    ap.add_argument("--heats", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    import fit_odds as F
    jobs = [(e, 7000 + s, a.body, b) for e in EVENTS for s in range(a.heats) for b in a.brains]
    with ProcessPoolExecutor(a.workers) as ex:
        out = list(ex.map(run_one, jobs))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
    for e in EVENTS:
        print(f"E{e:02d} {F.EVENTS[e][0]}")
        for b in a.brains:
            rows = [r for o in out if o["event"] == e and o["brain"] == Path(b).name for r in o["rows"]]
            marks = [r["mark"] for r in rows if r["mark"] is not None]
            status = {}
            for r in rows:
                status[r["status"]] = status.get(r["status"], 0) + 1
            peak = [r["peak"] for r in rows if r["peak"]]
            print(f"   {Path(b).name:28s} mean place {np.mean([r['place'] for r in rows]):.2f} | mark mean "
                  + (f"{np.mean(marks):.2f}" if marks else "-") + (f" | peak {np.mean(peak):.2f} m/s" if peak else "") + f" | {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
