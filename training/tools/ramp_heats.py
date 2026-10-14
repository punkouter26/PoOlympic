"""Event 14 The Alpine Ramp on CPU: heats of track.run_race(mode "ramp") in the mixed lineup (2 of each body).

Usage (training/): uv run python tools/ramp_heats.py [--heats 4] [--brain body=path.onnx ...] [--out ../parity/ramp_heats.json]
Without --brain every body runs its deployed Rung 2 (flat ground) brain = the baseline. Prints per body: finishers,
falls, mean finish time.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def main() -> None:
    import fit_odds as F
    from poolympic.events import track
    ap = argparse.ArgumentParser()
    ap.add_argument("--heats", type=int, default=4)
    ap.add_argument("--brain", action="append", default=[], help="body=brain.onnx instead of the deployed Rung 2 brain")
    ap.add_argument("--lineup", default="mzgkmzgk")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--slope", type=float, default=None, help="deg instead of the event's 15 (probes)")
    ap.add_argument("--vx", type=float, default=None, help="command speed in MATT units instead of the event's")
    a = ap.parse_args()
    if a.slope is not None:
        track.MODES["ramp"]["slope_deg"] = a.slope
    if a.vx is not None:
        track.MODES["ramp"]["vx"] = a.vx
    brains = dict(F.R2, **{b: Path(p) for b, p in (x.split("=", 1) for x in a.brain)})
    scene, layout = F.A / f"scene_track8_{a.lineup}.xml", F.A / f"track8_{a.lineup}_layout.json"
    per, runs = defaultdict(list), []
    for seed in range(14000, 14000 + a.heats):
        res = track.run_race(brains["matt"], "ramp", seed, scene=scene, layout_path=layout, brains=brains)
        runs.append(track.to_json(res))
        for l in res.lanes:
            per[l.body].append(l)
    for body, ls in per.items():
        fin = [l.finish_s for l in ls if l.status == "FINISHED"]
        print(f"{body:8s} finished {len(fin)} of {len(ls)}  fell {sum(l.status == 'FELL' for l in ls)}  "
              f"dnf {sum(l.status == 'DNF' for l in ls)}  mean finish {np.mean(fin) if fin else float('nan'):.1f} s  "
              f"peak {np.mean([l.peak_mps for l in ls]):.2f} m/s")
    if a.out:
        a.out.write_text(json.dumps(runs, indent=1))
        print("->", a.out)


if __name__ == "__main__":
    main()
