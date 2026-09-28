"""Iron Pedestal: compare brains on identical 8-runner heats (same seeds -> same traits and gust directions) and check
Rung 0 G1 does not regress.

Usage: uv run python tools/eval_pedestal.py <brain.onnx> [<brain.onnx> ...] [--heats 10]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poolympic import evaluate as E  # noqa: E402
from poolympic.events import iron_pedestal as IP  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("brains", nargs="+", type=Path)
    ap.add_argument("--heats", type=int, default=10)
    ap.add_argument("--g1", action="store_true", help="also run Rung 0 G1 (10 seeds)")
    a = ap.parse_args()
    for b in a.brains:
        outs, durs, reasons = [], [], {}
        for seed in range(1, a.heats + 1):
            r = IP.run_heat(b, seed)
            durs.append(r.duration_s)
            for l in r.lanes:
                outs.append(l.out_at_s if l.out_at_s is not None else r.duration_s)
                reasons[l.reason or "winner"] = reasons.get(l.reason or "winner", 0) + 1
        line = (f"{b.stem:28s} heats {a.heats}: mean heat {np.mean(durs):5.1f} s (min {min(durs):.1f}, max {max(durs):.1f}) | "
                f"mean survival {np.mean(outs):5.1f} s | first-out mean {np.mean(sorted(outs)[:a.heats]):.1f} s | {reasons}")
        if a.g1:
            sim = E.Sim(b)
            res = [E.rung0_episode(b, s, sim=sim) for s in range(1000, 1010)]
            v = E.rung0_verdict(res)
            line += f" | G1 rung0 {v['passed_seeds']}/10, worst foot excursion {max(r.max_foot_excursion_m for r in res):.2f} m"
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
