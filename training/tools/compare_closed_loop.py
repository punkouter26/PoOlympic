"""G5 — closed-loop comparison: Unity play-mode recording vs golden CPU-MuJoCo reference (DESIGN.md §4).

Pass criteria (all over the 5 s window):
  * same fall outcome (fall = pelvis z < 0.55 m); if both fall, fall times within 0.1 s
  * pelvis-height RMS difference < 3 cm
  * mean |actuator torque| within ±10 %
  * step cadence within ±5 % (only when the reference actually steps; statues/fallers skip it)

Usage: uv run python tools/compare_closed_loop.py <reference_name> <unity_record_name>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PARITY = Path(__file__).resolve().parents[2] / "parity"
FALL_Z = 0.55


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def qpos_map(ref_meta: dict, uni_meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays so ref_qpos[i_ref] <-> unity_qpos[i_uni]."""
    u = {j["name"]: j for j in uni_meta["joints"]}
    ir, iu = [], []
    for j in ref_meta["joints"]:
        n = 7 if j["type"] == 0 else 4 if j["type"] == 1 else 1
        ir += range(j["qposadr"], j["qposadr"] + n)
        iu += range(u[j["name"]]["qposadr"], u[j["name"]]["qposadr"] + n)
    return np.array(ir), np.array(iu)


def fall_time(z: np.ndarray, t: np.ndarray) -> float | None:
    idx = np.nonzero(z < FALL_Z)[0]
    return float(t[idx[0]]) if len(idx) else None


def cadence_hz(foot_contact_like: np.ndarray, dt: float) -> float | None:
    """Stride frequency from zero crossings of a periodic signal (left-right hip flexion difference)."""
    x = foot_contact_like - foot_contact_like.mean()
    crossings = np.nonzero(np.diff(np.signbit(x)))[0]
    if len(crossings) < 4:
        return None
    return 0.5 / (np.mean(np.diff(crossings)) * dt)


def main(ref_name: str, uni_name: str) -> int:
    ref = load(PARITY / f"reference_trajectory_{ref_name}.json")
    uni = load(PARITY / f"unity_run_{uni_name}.json")
    rf, uf = ref["frames"], uni["frames"]
    n = min(len(rf), len(uf))
    ir, iu = qpos_map(ref["meta"], uni["meta"])
    rq = np.array([f["qpos"] for f in rf[:n]])[:, ir]
    uq = np.array([f["qpos"] for f in uf[:n]])[:, iu]
    t = np.array([f["t"] for f in rf[:n]])
    dt = t[1] - t[0]

    drift = np.abs(rq - uq).max(axis=1)
    rz, uz = rq[:, 2], uq[:, 2]
    fr, fu = fall_time(rz, t), fall_time(uz, t)
    same_fall = (fr is None) == (fu is None) and (fr is None or abs(fr - fu) <= 0.1)
    height_rms = float(np.sqrt(np.mean((rz - uz) ** 2)))

    # torque: Python records force after the tick's steps; Unity records the previous tick's force at tick start
    rt = np.abs(np.array([f["actuator_force"] for f in rf[: n - 1]])).mean()
    ut = np.abs(np.array([f["actuator_force_prev"] for f in uf[1:n]])).mean()
    torque_ratio = float(ut / rt) if rt > 0 else 1.0

    obs_diff = float(np.abs(np.array([f["obs"] for f in rf[:n]]) - np.array([f["obs"] for f in uf[:n]])).max())
    act_diff = float(np.abs(np.array([f["action_raw"] for f in rf[:n]]) - np.array([f["action_raw"] for f in uf[:n]])).max())

    hip = ref["meta"]["actuators"].index("hip_flex_l"), ref["meta"]["actuators"].index("hip_flex_r")
    ro = np.array([f["obs"] for f in rf[:n]])
    uo = np.array([f["obs"] for f in uf[:n]])
    jp = load(PARITY / "contract.json")["obs_layout"]["joint_pos_rel"]["offset"]
    cr = cadence_hz(ro[:, jp + hip[0]] - ro[:, jp + hip[1]], dt)
    cu = cadence_hz(uo[:, jp + hip[0]] - uo[:, jp + hip[1]], dt)
    cadence_ok = True if cr is None else (cu is not None and abs(cu - cr) / cr <= 0.05)

    checks = {k: bool(v) for k, v in {
        "same_fall_outcome": same_fall,
        "pelvis_height_rms_lt_3cm": height_rms < 0.03,
        "mean_abs_torque_within_10pct": abs(torque_ratio - 1.0) <= 0.10,
        "cadence_within_5pct": cadence_ok,
    }.items()}
    report = {
        "reference": ref_name, "unity_run": uni_name, "frames": n,
        "fall_time_ref": fr, "fall_time_unity": fu,
        "pelvis_height_rms_m": height_rms, "mean_abs_torque_ref": float(rt), "mean_abs_torque_unity": float(ut),
        "torque_ratio": torque_ratio, "cadence_ref_hz": cr, "cadence_unity_hz": cu,
        "qpos_max_abs_drift_1s": float(drift[: int(round(1.0 / dt)) + 1].max()), "qpos_max_abs_drift_5s": float(drift.max()),
        "obs_max_abs_diff": obs_diff, "action_raw_max_abs_diff": act_diff,
        "checks": checks, "G5_pass": all(checks.values()),
    }
    out = PARITY / f"g5_report_{uni_name}.json"
    out.write_text(json.dumps(report, indent=1))
    for k, v in report.items():
        if k != "checks":
            print(f"{k:28s} {v}")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("G5", "PASS" if report["G5_pass"] else "FAIL")
    return 0 if report["G5_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
