"""Aggregate the per-tick action logs (poolympic/action_log.py, parity/actions/<run>/*.jsonl.gz) into a report.

Per run: episodes, success rate, falls and when they happen, mean |torque| per joint group (N·m and share of the cap),
action rate (mean |a_t - a_t-1| per joint and tick), share of ticks with any joint over the speed bar, and the joints
that spend the most ticks over it. An event heat counts one episode per lane (success = not FELL / DQ / STEPPED OFF).

Usage: uv run python tools/report_actions.py [run ...] [--md report.md] [--csv report.csv]      (default: every run)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poolympic import action_log as AL  # noqa: E402
from poolympic.biomech import JOINT_GROUPS, group_of  # noqa: E402

FALL_BINS = (2.0, 5.0, 10.0, 20.0)       # s: histogram edges of the fall time
BAD = ("FELL", "DQ", "STEPPED OFF")


def read(path: Path):
    """(header, {lane: [tick records]}, {lane: lane header}, summary) of one log file; lane None = a solo episode."""
    head, lanes, ticks, summary = None, {}, defaultdict(list), None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("header"):
                if "lane" in r:
                    lanes[r["lane"]] = r
                else:
                    head = r
            elif r.get("summary"):
                summary = r
            else:
                ticks[r.get("lane")].append(r)
    return head, ticks, lanes, summary


def outcomes(summary: dict | None, lane) -> tuple[bool, float | None]:
    """(success, fall time in s or None) of one episode / one lane of a heat."""
    if summary is None:                                   # the process died before the summary line
        return False, None
    if lane is None:
        ok = summary["passed"] if summary["passed"] is not None else summary["outcome"] == "ok"
        return bool(ok), summary["fall_s"]
    r = next(x for x in summary["result"]["lanes"] if x["lane"] == lane)
    why = str(r.get("status") or r.get("reason") or "")
    return why not in BAD, r.get("out_at_s") if why in BAD else None


def aggregate(run_dir: Path) -> dict:
    eps = ok = 0
    falls: list[float] = []
    n_ticks = 0
    tq = np.zeros(len(JOINT_GROUPS))
    tq_rel = np.zeros(len(JOINT_GROUPS))
    rate_sum, rate_n, over_any = 0.0, 0, 0
    over: dict[str, int] = defaultdict(int)
    for path in sorted(run_dir.glob("*.jsonl.gz")):
        head, ticks, lanes, summary = read(path)
        if head is None:
            continue
        names = head["actuators"]
        gi = np.array([group_of(n) for n in names])
        cap = np.array(head["force_cap"])
        for lane, recs in ticks.items():
            eps += 1
            good, fall_s = outcomes(summary, lane)
            ok += good
            if fall_s is not None:
                falls.append(float(fall_s))
            frc = np.abs(np.array([r["frc"] for r in recs]))
            act = np.array([r["act"] for r in recs])
            qd = np.abs(np.array([r["qd"] for r in recs]))
            limit = lanes[lane]["jv_limit"] if lane in lanes else head["jv_limit"]
            n_ticks += len(recs)
            for g in range(len(JOINT_GROUPS)):
                tq[g] += frc[:, gi == g].mean(axis=1).sum()
                tq_rel[g] += (frc[:, gi == g] / cap[gi == g]).mean(axis=1).sum()
            if len(act) > 1:
                rate_sum += float(np.abs(np.diff(act, axis=0)).mean(axis=1).sum())
                rate_n += len(act) - 1
            hot = qd > limit
            over_any += int(hot.any(axis=1).sum())
            for j in np.nonzero(hot.any(axis=0))[0]:
                over[names[j]] += int(hot[:, j].sum())
    n = max(n_ticks, 1)
    hist = np.histogram(falls, bins=(0.0, *FALL_BINS, np.inf))[0].tolist()
    return {"run": run_dir.name, "episodes": eps, "success_rate": ok / eps if eps else float("nan"), "falls": len(falls),
            "fall_hist": hist, "ticks": n_ticks,
            "torque_nm": dict(zip(JOINT_GROUPS, (tq / n).round(2).tolist())),
            "torque_of_cap": dict(zip(JOINT_GROUPS, (tq_rel / n).round(3).tolist())),
            "action_rate": rate_sum / max(rate_n, 1), "ticks_over_speed": over_any / n,
            "joints_over_speed": {k: v / n for k, v in sorted(over.items(), key=lambda kv: -kv[1])}}


def markdown(rows: list[dict]) -> str:
    edges = ["0-2", "2-5", "5-10", "10-20", "20+"]
    out = ["# Action-log report", "",
           "| run | episodes | success | falls | fall time (s) " + " / ".join(edges) + " | action rate | ticks over joint-speed bar |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['run']} | {r['episodes']} | {r['success_rate']:.0%} | {r['falls']} | "
                   f"{' / '.join(str(x) for x in r['fall_hist'])} | {r['action_rate']:.3f} | {r['ticks_over_speed']:.2%} |")
    out += ["", "Mean |torque| per joint group, N·m (share of the actuator cap):", "",
            "| run | " + " | ".join(JOINT_GROUPS) + " |", "|---|" + "---|" * len(JOINT_GROUPS)]
    for r in rows:
        out.append(f"| {r['run']} | " + " | ".join(f"{r['torque_nm'][g]:.1f} ({r['torque_of_cap'][g]:.0%})"
                                                 for g in JOINT_GROUPS) + " |")
    out += ["", "Joints over the speed bar (share of ticks, worst five per run):", ""]
    for r in rows:
        top = list(r["joints_over_speed"].items())[:5]
        out.append(f"- {r['run']}: " + (", ".join(f"{k} {v:.2%}" for k, v in top) if top else "none"))
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--dir", type=Path, default=AL.DIR)
    ap.add_argument("--md", type=Path)
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args()
    dirs = [a.dir / r for r in a.runs] if a.runs else sorted(p for p in a.dir.iterdir() if p.is_dir())
    rows = [aggregate(d) for d in dirs]
    text = markdown(rows)
    if a.md:
        a.md.write_text(text, encoding="utf-8")
    if a.csv:
        with a.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "episodes", "success_rate", "falls", *(f"falls_{e}s" for e in ("0-2", "2-5", "5-10", "10-20", "20+")),
                        "action_rate", "ticks_over_speed", *(f"torque_nm_{g}" for g in JOINT_GROUPS),
                        *(f"torque_of_cap_{g}" for g in JOINT_GROUPS)])
            for r in rows:
                w.writerow([r["run"], r["episodes"], round(r["success_rate"], 4), r["falls"], *r["fall_hist"],
                            round(r["action_rate"], 4), round(r["ticks_over_speed"], 5),
                            *(r["torque_nm"][g] for g in JOINT_GROUPS), *(r["torque_of_cap"][g] for g in JOINT_GROUPS)])
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
