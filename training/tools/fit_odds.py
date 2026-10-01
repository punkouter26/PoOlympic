"""Betting layer — odds from athlete traits (tasks.md D4 "odds from traits").

Runs CPU heats of every playable event with the mixed lineups (assets/scene_*_mzmzmzmz.xml MATT / zombie, seeds 5000+;
scene_*_mzgmzgmg.xml MATT / zombie / GRANDMA, seeds 7000+; brains as in Unity) and random traits, then fits one
Plackett-Luce rating model per event:

    rating_i = w_zombie * [body is zombie] + w_strength * (strength - 1) + w_latency * latency_substeps + w_noise * obs_noise
               + w_grandma * [body is grandma]
    P(i wins) = exp(rating_i) / sum_j exp(rating_j)            (Unity: PoOlympic.Odds)

fitted on the full finishing order (ties broken by lane), ridge-regularised. The model file is read by Unity:
Assets/PoOlympic/Models/odds_model.json (events as a list: Unity JsonUtility). Raw heats: parity/odds/heats.jsonl.

Usage: uv run python tools/fit_odds.py [--heats 40] [--workers 12] [--events 1,5,8,...] [--fit-only]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BR = ROOT.parent / "parity" / "brains"
A = ROOT / "assets"
OUT_DIR = ROOT.parent / "parity" / "odds"
MODEL = ROOT.parent / "Assets" / "PoOlympic" / "Models" / "odds_model.json"
FEATURES = ["zombie", "strength", "latency", "noise", "grandma"]   # Unity Odds.Rating reads them in this order
RIDGE = 0.5
R2 = {"matt": BR / "rung2.onnx", "zombie": BR / "zombie_rung2.onnx", "grandma": BR / "grandma_rung2.onnx"}
LINEUPS = {"mzmzmzmz": 5000, "mzgmzgmg": 7000}     # scene tag -> first seed (GRANDMA: no crawl brain, so not in 8 / 23)

# event -> (name, runner kind, scene tag)
EVENTS = {
    1: ("The Iron Pedestal", "pedestal", "pedestal8"),
    5: ("The Gust Gauntlet", "gauntlet", "shaker8"),
    8: ("30m All Fours", "all_fours", "crawl8"),
    9: ("The Inverted Sprint", "track:inverted", "track8"),
    10: ("Crab Shuffle Relay", "crab", "crab8"),
    11: ("Slalom Sprint", "slalom", "slalom8"),
    12: ("The 360 Turntable", "turntable", "turntable8"),
    13: ("Steeplechase Jog", "track:steeple", "track8"),
    19: ("Terminal Velocity Sprint", "track:terminal", "track8"),
    22: ("Emergency Brake", "track:brake", "track8"),
    23: ("The Trench Crawl", "trench", "trench8"),
}


def lineup_of(seed: int) -> str:
    return max((t for t, s0 in LINEUPS.items() if seed >= s0), key=lambda t: LINEUPS[t])


def run_one(event: int, seed: int) -> dict:
    from poolympic.events import all_fours, crab, gauntlet, iron_pedestal, slalom, track, turntable
    _, kind, tag = EVENTS[event]
    scene, layout = A / f"scene_{tag}_{lineup_of(seed)}.xml", A / f"{tag}_{lineup_of(seed)}_layout.json"
    if kind == "pedestal":
        brains = {"matt": BR / "r0_v2_it1000.onnx", "zombie": BR / "zombie_rung0.onnx", "grandma": BR / "grandma_rung0.onnx"}
        res = iron_pedestal.run_heat(brains["matt"], seed, scene=scene, layout_path=layout, brains=brains)
    elif kind in ("all_fours", "trench"):
        res = all_fours.run_race({"matt": BR / "crawl_matt.onnx", "zombie": BR / "crawl_zombie.onnx"}, seed, scene=scene,
                                 layout_path=layout, distance=all_fours.TRENCH_DISTANCE if kind == "trench" else None)
    elif kind.startswith("track:"):
        mode = kind.split(":")[1]
        brains = dict(R2, matt=BR / "r2f_v3_it100.onnx") if mode == "steeple" else R2
        res = track.run_race(brains["matt"], mode, seed, scene=scene, layout_path=layout, brains=brains)
    else:
        mod = {"gauntlet": gauntlet, "crab": crab, "slalom": slalom, "turntable": turntable}[kind]
        res = mod.run_heat(R2["matt"], seed, scene=scene, layout_path=layout, brains=R2)
    bodies = [l.get("body", "matt") for l in json.loads(layout.read_text())["lanes"]]
    return {"event": event, "seed": seed, "lanes": [
        {"lane": l.lane, "body": bodies[l.lane], "strength": l.traits.strength, "latency": l.traits.latency_substeps,
         "noise": l.traits.obs_noise, "place": l.place} for l in res.lanes]}


def features(lane: dict) -> np.ndarray:
    return np.array([1.0 if lane["body"] == "zombie" else 0.0, lane["strength"] - 1.0, float(lane["latency"]), lane["noise"],
                     1.0 if lane["body"] == "grandma" else 0.0])


def fit(heats: list[dict]) -> tuple[np.ndarray, dict]:
    """Plackett-Luce MLE on full finishing orders + L2 ridge (convex), L-BFGS with the exact gradient."""
    from scipy.optimize import minimize
    orders = []
    for h in heats:
        ls = sorted(h["lanes"], key=lambda l: (l["place"], l["lane"]))
        orders.append(np.stack([features(l) for l in ls]))

    def nll(w):
        f, g = 0.5 * RIDGE * w @ w, RIDGE * w
        for X in orders:
            r = X @ w
            for k in range(len(X) - 1):          # stage k: the k-th finisher beats everyone behind it
                rk = r[k:]
                m = rk.max()
                e = np.exp(rk - m)
                z = e.sum()
                f -= r[k] - (m + math.log(z))
                g -= X[k] - (e / z) @ X[k:]
        return f, g

    w = minimize(nll, np.zeros(len(FEATURES)), jac=True, method="L-BFGS-B").x
    # diagnostics: favourite win rate, mean log-likelihood of the actual winner vs uniform (in-sample)
    hits, ll = 0, 0.0
    for X in orders:
        r = X @ w
        p = np.exp(r - r.max())
        p /= p.sum()
        hits += int(np.argmax(r) == 0)
        ll += math.log(p[0])
    n = len(orders)
    return w, {"heats": n, "favourite_wins": hits / n, "winner_loglik": ll / n, "uniform_loglik": math.log(1 / 8)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heats", type=int, default=40)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--events", default=",".join(str(e) for e in EVENTS))
    ap.add_argument("--fit-only", action="store_true")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = OUT_DIR / "heats.jsonl"
    events = [int(e) for e in a.events.split(",")]
    if not a.fit_only:
        done = set()
        if raw.exists():
            done = {(h["event"], h["seed"]) for h in map(json.loads, raw.read_text().splitlines())}
        jobs = [(e, s0 + i) for e in events for tag, s0 in LINEUPS.items() for i in range(a.heats)
                if (e, s0 + i) not in done and (A / f"scene_{EVENTS[e][2]}_{tag}.xml").exists()]
        print(f"{len(jobs)} heats to run on {a.workers} workers", flush=True)
        with ProcessPoolExecutor(a.workers) as pool, raw.open("a") as f:
            futs = {pool.submit(run_one, e, s): (e, s) for e, s in jobs}
            for i, fu in enumerate(as_completed(futs), 1):
                f.write(json.dumps(fu.result()) + "\n")
                f.flush()
                if i % 10 == 0:
                    print(f"  {i}/{len(jobs)}", flush=True)
    heats = [json.loads(x) for x in raw.read_text().splitlines()]
    model = {"note": "PoOlympic betting odds (training/tools/fit_odds.py): rating = w . [zombie, strength-1, latency, "
                     "noise, grandma]; P(win) = softmax(rating); Plackett-Luce fit on CPU heats, mixed MATT / zombie / GRANDMA lineups",
             "features": FEATURES, "margin": 0.10, "events": []}
    for e in sorted({h["event"] for h in heats}):
        w, diag = fit([h for h in heats if h["event"] == e])
        model["events"].append({"number": e, "name": EVENTS[e][0], "weights": [round(float(x), 4) for x in w], **diag})
        print(f"E{e:02d} {EVENTS[e][0]:26s} w={np.round(w, 2)}  favourite wins {diag['favourite_wins']:.0%} "
              f"(uniform 12.5%), winner LL {diag['winner_loglik']:.2f} vs {diag['uniform_loglik']:.2f}")
    MODEL.write_text(json.dumps(model, indent=1) + "\n")
    print(f"wrote {MODEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
