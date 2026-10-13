"""How much does athlete-athlete contact cost? CPU heats of the two contact-heavy races (19 Terminal Velocity with the
lane break, 11 mirror Slalom) per lineup, each seed run twice: crowd contact on (the event scene) and off (the same
scene with the crowd bits cleared = every athlete alone in its lane). Reports falls, finishers and "pair-s" (seconds of
athlete-athlete contact summed over athlete pairs, split by the two bodies).

Usage (training/): uv run python tools/crowd_heats.py [--heats 6] [--workers 8] [--events 19,11] [--lineups all,mzgkmzgk]
                   [--brain body=path.onnx ...]      # candidate Rung 2 brains instead of the deployed ones
Writes parity/crowd_heats.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

CROWD_ALL = 0xFF << 16            # = build_mjcf.CROWD_ALL
_REAL = None
LINEUPS = {"all": "", "mzmzmzmz": "_mzmzmzmz", "mzgmzgmg": "_mzgmzgmg", "mzgkmzgk": "_mzgkmzgk"}   # "all" = 8 MATTs


def run_one(job: tuple) -> dict:
    event, lineup, seed, contact, brains_over = job
    import mujoco
    import fit_odds as F
    global _REAL
    _REAL = _REAL or (mujoco.MjModel.from_xml_path, mujoco.mj_step)      # pool workers are reused: wrap the originals only
    from poolympic.events import slalom, track
    tag = F.EVENTS[event][2] + LINEUPS[lineup]
    scene, layout = F.A / f"scene_{tag}.xml", F.A / f"{tag}_layout.json"
    lay = json.loads(layout.read_text())["lanes"]
    bodies = [l.get("body", "matt") for l in lay]
    owner: dict[int, int] = {}
    pair_steps: Counter = Counter()
    load, step = _REAL

    def load_scene(path):
        m = load(path)
        if Path(path) != scene:        # the events also load the solo body scenes (fall heights)
            return m
        if not contact:
            m.geom_contype[:] &= ~CROWD_ALL
            m.geom_conaffinity[:] &= ~CROWD_ALL
        roots = [m.body(l["prefix"] + "pelvis").id for l in lay]             # lane order
        owner.update({g: roots.index(r) for g in range(m.ngeom) if (r := m.body_rootid[m.geom_bodyid[g]]) in roots})
        return m

    def counted_step(m, d):
        step(m, d)
        for a, b in {tuple(sorted((owner[g1], owner[g2]))) for g1, g2 in zip(d.contact.geom1[: d.ncon], d.contact.geom2[: d.ncon])
                     if g1 in owner and g2 in owner and owner[g1] != owner[g2]}:
            pair_steps["-".join(sorted((bodies[a], bodies[b])))] += 1

    mujoco.MjModel.from_xml_path, mujoco.mj_step = load_scene, counted_step      # the events load and step through these
    brains = dict(F.R2, **{b: Path(p) for b, p in brains_over})
    if event == 19:
        res = track.run_race(brains["matt"], "terminal", seed, scene=scene, layout_path=layout, brains=brains, recover=F.deployed("getup"))
    else:
        res = slalom.run_heat(brains["matt"], seed, scene=scene, layout_path=layout, brains=brains, tactics=True, recover=F.deployed("getup"))
    rows = [{"body": bodies[l.lane], "falls": l.falls + (l.status == "FELL"), "status": l.status, "place": l.place,
             "clips": getattr(l, "clips", 0)} for l in res.lanes]
    return {"event": event, "lineup": lineup, "seed": seed, "contact": contact, "rows": rows,
            "pair_s": {k: round(v * 0.005, 2) for k, v in pair_steps.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heats", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--events", default="19,11")
    ap.add_argument("--lineups", default=",".join(LINEUPS))
    ap.add_argument("--brain", action="append", default=[], help="body=brain.onnx (candidate instead of the deployed Rung 2 brain)")
    ap.add_argument("--out", type=Path, default=ROOT.parent / "parity" / "crowd_heats.json")
    a = ap.parse_args()
    over = tuple(tuple(x.split("=", 1)) for x in a.brain)
    jobs = [(int(e), l, 11000 + i, c, over) for e in a.events.split(",") for l in a.lineups.split(",")
            for i in range(a.heats) for c in (True, False)]
    with ProcessPoolExecutor(a.workers) as pool:
        heats = list(pool.map(run_one, jobs))
    a.out.write_text(json.dumps(heats, indent=1))
    print(f"{'event lineup':18s} {'contact':8s} {'falls/heat':>10s} {'heats with a fall':>18s} {'finish/8':>9s} {'clips':>6s}  pair-s per heat (by bodies)")
    for e in sorted({h["event"] for h in heats}):
        for l in a.lineups.split(","):
            for c in (True, False):
                hs = [h for h in heats if (h["event"], h["lineup"], h["contact"]) == (e, l, c)]
                falls = [sum(r["falls"] for r in h["rows"]) for h in hs]
                fin = np.mean([sum(r["status"] == "FINISHED" for r in h["rows"]) for h in hs])
                clips = np.mean([sum(r["clips"] for r in h["rows"]) for h in hs])
                ps = Counter()
                for h in hs:
                    ps.update(h["pair_s"])
                by_body = Counter()
                for h in hs:
                    for r in h["rows"]:
                        by_body[r["body"]] += r["falls"]
                print(f"E{e:02d} {l:14s} {'on' if c else 'off':8s} {np.mean(falls):10.2f} {sum(f > 0 for f in falls):>12d} of {len(hs):<3d} {fin:9.2f} {clips:6.1f}  "
                      + ", ".join(f"{k} {v / len(hs):.1f}" for k, v in sorted(ps.items()))
                      + ("   falls: " + ", ".join(f"{b} {n}" for b, n in sorted(by_body.items()) if n) if sum(falls) else ""))
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
