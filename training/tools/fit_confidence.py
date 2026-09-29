"""Brain confidence (feature 5) — calibrate each event brain's critic into P(still on its feet HORIZON_S from now).

CPU MuJoCo rollouts on the mixed 8-lane track (assets/scene_track8_mzmzmzmz.xml: MATT lanes 1/3/5/7, zombie 2/4/6/8,
lane-isolated), every lane driven like its Unity event (tools/export_critic.py critics next to the brains):

  stand   r0_v2_it1000 / zombie_rung0     zero command                     (Iron Pedestal)
  rung2   rung2 / zombie_rung2            random envelope commands         (every other standing event)
  flight  r2f_v3_it100                    3.0-3.8 m/s runs                 (Steeplechase Jog)
  crawl   crawl_matt / crawl_zombie       prone start, 3 s setup, 1.2 m/s  (30m All Fours, Trench Crawl)

plus random shoves (0.3-2.4 m/s × √λ every 1.5-3.5 s) so that falls happen. Fall = the event fall rule (pelvis line,
60° torso tilt, non-foot contact); crawl = a tumble (pelvis below the crawl band after being on all fours). A lane that
falls is reset and starts a new episode. Every control tick gives one sample (critic value, label = no fall within
HORIZON_S); samples in the last HORIZON_S of an episode that never fell are dropped (unknown future).

Features per tick: the critic value V, its drop against its own ~1 s exponential average (V - EMA(V): the brain
suddenly expecting less), and the commanded speed |cmd_xy| in MATT units (the running brains' V scales with the
command, so V alone does not rank danger across speeds).
Fit per brain: logistic regression  P = sigmoid(w_value·V + w_drop·(V - EMA) + w_speed·|cmd| + b)  (IRLS, light
ridge), reported with ROC AUC (and the value-only AUC) and a
10-bin reliability table. Written to Assets/PoOlympic/Models/confidence_model.json (read by Unity BrainConfidence) and
parity/confidence/report.json.

Usage: uv run python tools/fit_confidence.py [--seconds 150] [--seeds 8] [--workers 12]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BR = ROOT.parent / "parity" / "brains"
A = ROOT / "assets"
MODEL = ROOT.parent / "Assets" / "PoOlympic" / "Models" / "confidence_model.json"
REPORT = ROOT.parent / "parity" / "confidence" / "report.json"
HORIZON_S = 2.0
SETTLE_S = 1.0
EMA_TAU_S = 1.0             # value baseline: exponential average over ~1 s of control ticks
EMA_ALPHA = 0.02            # = control dt (0.02 s) / EMA_TAU_S

GROUPS = {
    "stand": {"matt": "r0_v2_it1000", "zombie": "zombie_rung0"},
    "rung2": {"matt": "rung2", "zombie": "zombie_rung2"},
    "flight": {"matt": "r2f_v3_it100"},
    "crawl": {"matt": "crawl_matt", "zombie": "crawl_zombie"},
}


def rollout(group: str, seed: int, seconds: float) -> dict[str, list[tuple[float, int]]]:
    from poolympic import bodies
    from poolympic import contract as C
    from poolympic.evaluate import _envelope_command
    from poolympic.events import all_fours as AF
    from poolympic.events.iron_pedestal import Traits, body_contract, make_lanes

    scene, layout_path = A / "scene_track8_mzmzmzmz.xml", A / "track8_mzmzmzmz_layout.json"
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    layout = json.loads(layout_path.read_text())
    brains = {b: BR / f"{n}.onnx" for b, n in GROUPS[group].items()}
    rng = np.random.default_rng([seed, 77])
    traits = [Traits.sample(rng) for _ in layout["lanes"]]
    fallback = brains.get("matt") or next(iter(brains.values()))
    lanes = make_lanes(m, d, layout, seed, traits, fallback, {b: brains.get(b, fallback) for b in ("matt", "zombie")})
    active = [ln.body in brains for ln in lanes]
    critics = {b: ort.InferenceSession(str(BR / f"{n}.critic.onnx"), providers=["CPUExecutionProvider"])
               for b, n in GROUPS[group].items()}
    defaults = {b: body_contract(b)["default_joint_qpos"] for b in ("matt", "zombie")}
    lam = [bodies.BODIES[ln.body].length_scale for ln in lanes]
    torso = [m.body(ln.prefix + "torso").id for ln in lanes]
    crawl = group == "crawl"
    dt = m.opt.timestep * C.DECIMATION

    def reset(i):
        ln = lanes[i]
        ln.reset(m, d, defaults[ln.body])
        if crawl:
            AF.prone_start(m, d, ln)
        ln.phase, ln.last = 0.0, np.zeros(C.NUM_ACTIONS)
        ln.ctrl_now = ln.ctrl_prev = ln.ath.default_pos.copy()

    for i in range(len(lanes)):
        reset(i)
    mujoco.mj_forward(m, d)
    ep_start = [0.0] * len(lanes)
    samples: list[list[tuple[float, float]]] = [[] for _ in lanes]
    out: dict[str, list[tuple[float, int]]] = {GROUPS[group][ln.body]: [] for ln in lanes if ln.body in brains}
    cmd = [np.zeros(3) for _ in lanes]
    cmd_until = [0.0] * len(lanes)
    next_shove = [rng.uniform(1.5, 3.5) for _ in lanes]
    was_up = [False] * len(lanes)
    ema = [0.0] * len(lanes)

    def close(i, t_end, fell):
        name = GROUPS[group][lanes[i].body]
        for t, *f in samples[i]:
            if fell:
                out[name].append((*f, int(t_end - t > HORIZON_S)))
            elif t <= t_end - HORIZON_S:
                out[name].append((*f, 1))
        samples[i] = []

    tick = 0
    while tick * dt < seconds:
        t = tick * dt
        for i, ln in enumerate(lanes):
            if not active[i]:
                continue
            age = t - ep_start[i]
            r = ln.ath.root_qposadr
            if crawl:
                c = np.zeros(3) if age < AF.SETUP_S else np.array(
                    [AF.VX, 0.0, AF.crawl_steer(d.qpos[r + 3:r + 7], d.qpos[r + 1] - ln.origin[1])])
            else:
                if t >= cmd_until[i]:
                    if group == "stand":
                        cmd[i] = np.zeros(3)
                    elif group == "flight":
                        cmd[i] = np.array([rng.uniform(3.0, 3.8), 0.0, 0.0])
                    else:
                        cmd[i] = _envelope_command(rng)[1]
                    cmd_until[i] = t + rng.uniform(3.0, 5.0)
                c = cmd[i].copy()
                if group == "flight":
                    c[2] = C.steer_yaw_rate(d.qpos[r + 3:r + 7], d.qpos[r + 1] - ln.origin[1], c[0])
            # the lane's control step, exposing the observation for the critic (= _Lane.control)
            bc = np.asarray(c, float) if ln.body == "matt" else np.asarray(c, float) * np.array(
                [math.sqrt(lam[i]), math.sqrt(lam[i]), 1.0 / math.sqrt(lam[i])])
            ln.phase = ln.advance_phase(bc)
            obs = C.build_obs(ln.ath, d.qpos, d.qvel, bc, ln.phase, ln.last)
            if ln.traits.obs_noise > 0:
                obs = (obs + ln.rng.uniform(-1, 1, C.OBS_DIM) * ln.noise).astype(np.float32)
            ctrl, act = ln.sess.run(None, {"obs": obs[None]})
            ln.ctrl_prev, ln.ctrl_now = ln.ctrl_now, ctrl[0].astype(np.float64)
            ln.last = act[0].astype(np.float64)
            v = float(critics[ln.body].run(None, {"obs": obs.astype(np.float32)[None]})[0][0, 0])
            ema[i] = v if age == 0.0 else ema[i] + EMA_ALPHA * (v - ema[i])
            if age >= (AF.SETUP_S if crawl else SETTLE_S):
                samples[i].append((t, v, v - ema[i], float(np.hypot(c[0], c[1]))))
            if t >= next_shove[i] and age >= (AF.SETUP_S if crawl else SETTLE_S):
                a = rng.uniform(0, 2 * math.pi)
                dv = rng.uniform(0.3, 2.4) * math.sqrt(lam[i])
                d.qvel[ln.ath.root_dofadr: ln.ath.root_dofadr + 2] += dv * np.array([math.cos(a), math.sin(a)])
                next_shove[i] = t + rng.uniform(1.5, 3.5)
        for s in range(C.DECIMATION):
            for i, ln in enumerate(lanes):
                if active[i]:
                    ln.write_ctrl(d, s)
            mujoco.mj_step(m, d)
        tick += 1
        t = tick * dt
        for i, ln in enumerate(lanes):
            if not active[i]:
                continue
            if crawl:
                if t - ep_start[i] < AF.SETUP_S:
                    continue
                z = d.qpos[ln.ath.root_qposadr + 2]
                tilt = math.degrees(math.acos(max(-1.0, min(1.0, d.xmat[torso[i]][8]))))
                on4 = AF.BAND[0] * lam[i] < z < AF.BAND[1] * lam[i] and tilt > 50.0
                fell = was_up[i] and z < AF.BAND[0] * lam[i]
                was_up[i] = on4
            else:
                fell = ln.eliminated(m, d) is not None
            if fell:
                close(i, t, True)
                reset(i)
                was_up[i] = False
                ep_start[i] = t
                mujoco.mj_forward(m, d)
    for i in range(len(lanes)):
        if active[i]:
            close(i, tick * dt, False)
    return out


def logistic(F: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> tuple[np.ndarray, float]:
    """P = sigmoid(F @ w + b), IRLS on standardised features; returns raw-feature weights and bias."""
    mu, sd = F.mean(0), F.std(0)
    used = sd > 1e-3                         # a constant feature (e.g. the crawl's fixed 1.2 m/s command) carries no signal
    mu, sd = np.where(used, mu, 0.0), np.where(used, sd, 1.0)
    X = np.hstack([np.where(used, (F - mu) / sd, 0.0), np.ones((len(F), 1))])
    w = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(len(w))
        g = X.T @ (y - p) - ridge * w
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-9:
            break
    a = np.where(used, w[:-1] / sd, 0.0)
    return a, float(w[-1] - (a * mu).sum())


def auc(score: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=150.0)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    data: dict[str, list[tuple[float, int]]] = {}
    jobs = [(g, s) for g in GROUPS for s in range(args.seeds)]
    with ProcessPoolExecutor(args.workers) as ex:
        for (g, s), res in zip(jobs, ex.map(rollout, *zip(*jobs), [args.seconds] * len(jobs))):
            for name, rows in res.items():
                data.setdefault(name, []).extend(rows)
            print(f"  {g} seed {s}: " + ", ".join(f"{n} {len(r)}" for n, r in res.items()), flush=True)
    brains, report = [], {}
    for name, rows in sorted(data.items()):
        arr = np.asarray(rows, float)
        F, y = arr[:, :3], arr[:, 3]
        v = F[:, 0]
        falls = int((y == 0).sum())
        a, b = logistic(F, y)
        p = 1 / (1 + np.exp(-(F @ a + b)))
        score_auc = auc(p, y)
        bins = np.quantile(p, np.linspace(0, 1, 11))
        rel = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (p >= lo) & (p <= hi)
            if sel.any():
                rel.append({"p_pred": round(float(p[sel].mean()), 3), "p_obs": round(float(y[sel].mean()), 3), "n": int(sel.sum())})
        entry = {"brain": name, "w_value": float(a[0]), "w_drop": float(a[1]), "w_speed": float(a[2]), "b": b,
                 "auc": round(score_auc, 4), "auc_value_only": round(auc(v, y), 4), "samples": len(y), "danger_samples": falls,
                 "base_rate": round(float(y.mean()), 4), "v_p05": float(np.quantile(v, 0.05)),
                 "v_p50": float(np.quantile(v, 0.5)), "v_p95": float(np.quantile(v, 0.95)),
                 "calibrated": bool(falls >= 200 and score_auc > 0.6)}
        brains.append(entry)
        report[name] = {**entry, "reliability": rel}
        print(f"{name:14s} n={len(y):7d} danger={falls:6d} AUC={entry['auc']:.3f} (value only {entry['auc_value_only']:.3f}) "
              f"w=({a[0]:+.4f}, {a[1]:+.4f}, {a[2]:+.4f}) b={b:+.3f} "
              f"calibrated={entry['calibrated']}")
    MODEL.write_text(json.dumps({"horizon_s": HORIZON_S, "ema_tau_s": EMA_TAU_S, "brains": brains}, indent=1))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1))
    print(f"-> {MODEL}\n-> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
