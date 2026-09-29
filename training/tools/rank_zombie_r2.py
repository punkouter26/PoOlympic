"""Rank zombie Rung 2 checkpoints (Z6): official G1 (per-tick yaw bar) and the stride-averaged yaw variant
(evaluate.py yaw_rms_stride, the open ruling), from eval_cpu.py reports.

A seed passes the stride variant when every official check passes except tracking_yaw, and every segment's
yaw_rms_stride < RUNG2_YAW_TOL.

Usage: POOLYMPIC_BODY=zombie uv run python tools/rank_zombie_r2.py ../parity/eval_stride_*.json
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poolympic import evaluate as E  # noqa: E402


def score(path: Path) -> dict:
    d = json.loads(path.read_text())
    stride_pass = 0
    for ep, checks in zip(d["episodes"], d["per_seed_checks"]):
        others = all(v for k, v in checks.items() if k != "tracking_yaw")
        segs = ep["segments"]
        stride_ok = bool(segs) and all(s.get("yaw_rms_stride") is not None and s["yaw_rms_stride"] < E.RUNG2_YAW_TOL
                                       for s in segs)
        stride_pass += int(others and stride_ok)
    segs = [s for ep in d["episodes"] for s in ep["segments"]]
    return {"report": path.name, "seeds": d["seeds"], "official": d["passed_seeds"], "stride": stride_pass,
            "lin_med": round(st.median(s["lin_rms"] for s in segs), 3),
            "yaw_med": round(st.median(s["yaw_rms"] for s in segs), 3),
            "yaw_stride_med": round(st.median(s["yaw_rms_stride"] for s in segs if s.get("yaw_rms_stride") is not None), 3),
            "turntable_s": round(st.mean(ep["turntable_s"] for ep in d["episodes"]), 2),
            "falls": sum(ep["fell"] is not None for ep in d["episodes"])}


if __name__ == "__main__":
    rows = sorted((score(Path(p)) for p in sys.argv[1:]), key=lambda r: (-r["official"], -r["stride"], r["lin_med"]))
    for r in rows:
        print(f"{r['report']:48s} official {r['official']}/{r['seeds']}  stride {r['stride']}/{r['seeds']}  lin {r['lin_med']}  "
              f"yaw {r['yaw_med']} (stride {r['yaw_stride_med']})  turntable {r['turntable_s']} s  falls {r['falls']}")
