"""Which checks of a G1 gate report fail, per seed, and (Rung 2) the tracking segments over the bar by kind, the
turntable times and the mean speed shortfall of the sprint commands.
    uv run python tools/gate_fails.py parity/watch/g1_rung2_<name>.json ...        (body = $POOLYMPIC_BODY: scaled bars)
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poolympic import bodies  # noqa: E402

B = bodies.current()
LIN_BAR = 0.2 * B.speed_scale

for path in sys.argv[1:]:
    j = json.load(open(path))
    fails = Counter(k for checks in j.get("per_seed_checks", []) for k, ok in checks.items() if not ok)
    print(f"{Path(path).stem.split('_')[-1]}: {j['passed_seeds']}/{j['seeds']}  failing checks {dict(fails)}")
    if not j["episodes"] or "segments" not in j["episodes"][0]:
        continue
    over = defaultdict(list)
    n = Counter()
    for e in j["episodes"]:
        for s in e["segments"]:
            n[s["kind"]] += 1
            if s["lin_rms"] >= LIN_BAR:
                over[s["kind"]].append((round(s["lin_rms"], 2), [round(c, 2) for c in s["cmd"]]))
    for kind in n:
        print(f"   {kind:7s} {len(over[kind])}/{n[kind]} segments over {LIN_BAR:.3f} m/s: {sorted(over[kind], reverse=True)[:6]}")
    tt = sorted(e["turntable_s"] if e["turntable_s"] is not None else 99.0 for e in j["episodes"])
    print(f"   turntable s (sorted): {tt}   drift max {max(e['turntable_drift_m'] or 0 for e in j['episodes']):.2f} m")
