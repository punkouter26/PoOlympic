"""Event 23 (The Trench Crawl) feasibility — can the Event 8 crawl brains pass under a low ceiling without training?

A copy of an 8-lane track scene gets a trench ceiling box (x 2-14 m, all lanes, underside at --under metres) and runs
the all-fours race rules (events/all_fours.py) over DISTANCE metres. Reports finishes, times and tumbles per height.

Usage: uv run python tools/trench_probe.py [--under 0.62,0.70,0.78] [--seeds 1,2,3] [--lineup matt|mixed] [--distance 16]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BR = ROOT.parent / "parity" / "brains"
A = ROOT / "assets"
TRENCH_X = (2.0, 14.0)      # m along the course (build_venues.py: 12 m ceiling from 2 m past the start)
LANES_Y = (-0.61, 5.2)      # centre / half width across the 8 lanes (lane origins +3.66 .. -4.88)


def ceiling_xml(src: str, under: float, name: str = "trench_ceiling", half_t: float = 0.02) -> str:
    """Insert the ceiling box (same collision bits as the ground: every lane) after the ground geom."""
    g = re.search(r'<geom name="ground"[^>]*/>', src).group(0)
    bits = re.search(r'contype="(\d+)"', g).group(1)
    cx, hx = (TRENCH_X[0] + TRENCH_X[1]) / 2, (TRENCH_X[1] - TRENCH_X[0]) / 2
    box = (f'<geom name="{name}" type="box" pos="{cx} {LANES_Y[0]} {under + half_t}" size="{hx} {LANES_Y[1]} {half_t}" '
           f'contype="{bits}" conaffinity="{bits}" condim="3"/>')
    return src.replace(g, g + "\n    " + box)


def run(args):
    under, seed, lineup, distance = args
    from poolympic.events import all_fours
    all_fours.DISTANCE = distance
    tag = "track8" if lineup == "matt" else "track8_mzmzmzmz"
    p = A / f"_trench_{under:.2f}_{seed}_{lineup}.xml"
    p.write_text(ceiling_xml((A / f"scene_{tag}.xml").read_text(), under))
    try:
        r = all_fours.run_race({"matt": BR / "crawl_matt.onnx", "zombie": BR / "crawl_zombie.onnx"}, seed, scene=p,
                               layout_path=A / f"{tag}_layout.json")
    finally:
        p.unlink()
    return under, [(l.body, l.status, l.finish_s, l.tumbles) for l in r.lanes]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--under", default="0.62,0.66,0.70,0.74,0.78")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--lineup", default="matt", choices=["matt", "mixed"])
    ap.add_argument("--distance", type=float, default=16.0)
    a = ap.parse_args()
    jobs = [(float(u), int(s), a.lineup, a.distance) for u in a.under.split(",") for s in a.seeds.split(",")]
    agg = defaultdict(list)
    with ProcessPoolExecutor(min(15, len(jobs))) as ex:
        for under, lanes in ex.map(run, jobs):
            agg[under] += lanes
    for under, lanes in sorted(agg.items()):
        for body in sorted({l[0] for l in lanes}):
            ls = [l for l in lanes if l[0] == body]
            fin = [t for _, st, t, _ in ls if st == "FINISHED"]
            print(f"underside {under:.2f} m {body:6s}: finished {len(fin)}/{len(ls)}, "
                  f"{min(fin):.1f}-{max(fin):.1f} s" if fin else f"underside {under:.2f} m {body:6s}: finished 0/{len(ls)}",
                  f"tumbles {sum(l[3] for l in ls)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
