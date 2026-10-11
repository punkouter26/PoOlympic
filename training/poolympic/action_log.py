"""Per-tick action log of the CPU evaluation path (AGENTS.md "Data logging": movement profiles, win / loss patterns).

Off by default. Set $POOLYMPIC_LOG_ACTIONS to a run name (or "1" = the brain's file stem), or pass --log-actions to
tools/eval_cpu.py / tools/watch_gate.py; every G1 episode (evaluate.py, evaluate_stance.py) and every event heat
(poolympic/events/) then writes  parity/actions/<run>/<gate>_seed<N>.jsonl.gz :

  line 1      {"header": true, run, body, gate, seed, dt, actuators, force_cap, jv_limit}
  (events)    {"header": true, "lane": k, body, jv_limit}                      one per lane
  per tick    {"t": tick, ["lane": k,] "cmd": [vx, vy, wz], "pos": pelvis xyz, "tilt": torso tilt (deg),
               "vel": pelvis linear velocity (world), "qd": joint speeds, "act": action_raw, "ctrl": ctrl,
               "frc": actuator_force}          state at the tick start, force of the previous tick's last substep
                                               (the Unity run format, meet.py); all in contract actuator order
  last line   {"summary": true, outcome, passed, fall_s, fall_tick, ticks, result}   result = the episode's own record

Aggregate with tools/report_actions.py. No observation is stored (it is a function of the logged state).
"""

from __future__ import annotations

import dataclasses
import functools
import gzip
import inspect
import json
import math
import os
from pathlib import Path

import numpy as np

from . import bodies

ENV = "POOLYMPIC_LOG_ACTIONS"
DIR = bodies.PARITY / "actions"
current: "ActionLog | None" = None       # the open log of the running episode (one episode at a time per process)


def _r(a) -> list:
    return np.round(np.asarray(a, dtype=float), 4).tolist()


def _plain(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)


class ActionLog:
    def __init__(self, run: str, gate: str, seed: int):
        self.path = DIR / run / f"{gate}_seed{seed}.jsonl.gz"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = gzip.open(self.path, "wt", encoding="utf-8", compresslevel=3)
        self.meta = {"run": run, "body": bodies.current().name, "gate": gate, "seed": int(seed)}
        self.ticks, self.dt, self.header_done = 0, 0.02, False

    def _write(self, rec: dict) -> None:
        self.f.write(json.dumps(rec, separators=(",", ":"), default=_plain) + "\n")

    def lane(self, k: int, body: str) -> None:
        self._write({"header": True, "lane": k, "body": body,
                     "jv_limit": bodies.JOINT_VEL_LIMIT / bodies.BODIES[body].time_scale})

    def tick(self, m, d, ath, torso: int, command, action, ctrl, lane: int | None = None) -> None:
        if not self.header_done:
            self.header_done = True
            self.dt = float(m.opt.timestep) * 4          # contract.DECIMATION (no import: contract loads a scene)
            self._write({"header": True, **self.meta, "dt": self.dt, "actuators": list(ath.actuator_names),
                         "force_cap": _r(np.abs(m.actuator_forcerange[ath.actuator_ids]).max(axis=1)),
                         "jv_limit": bodies.JOINT_VEL_LIMIT / bodies.current().time_scale})
        r, v = ath.root_qposadr, ath.root_dofadr
        rec = {"t": int(round(d.time / self.dt))}        # ticks since the scene's last reset
        if lane is not None:
            rec["lane"] = lane
        rec.update(cmd=_r(command), pos=_r(d.qpos[r:r + 3]),
                   tilt=round(math.degrees(math.acos(max(-1.0, min(1.0, float(d.xmat[torso][8]))))), 2),
                   vel=_r(d.qvel[v:v + 3]), qd=_r(d.qvel[ath.joint_dofadr]), act=_r(action), ctrl=_r(ctrl),
                   frc=_r(d.actuator_force[ath.actuator_ids]))
        self._write(rec)
        self.ticks += 1

    def close(self, result=None, passed: bool | None = None) -> None:
        res = dataclasses.asdict(result) if dataclasses.is_dataclass(result) else result
        fell = bool(getattr(result, "fell", False))
        fall_s = getattr(result, "fall_time", None)
        if fell and fall_s is None and not hasattr(result, "drills"):
            fall_s = self.ticks * self.dt                # the episode stops at the fall (rung 2: no fall time kept)
        self._write({"summary": True, **self.meta, "outcome": "fell" if fell else "ok", "passed": passed,
                     "fall_s": fall_s, "fall_tick": None if fall_s is None else int(round(fall_s / self.dt)),
                     "ticks": self.ticks, "result": res})
        self.f.close()


def logged(gate: str | None = None, passed=None):
    """Decorator for an episode / heat function with a `seed` argument: while $POOLYMPIC_LOG_ACTIONS is set, the ticks
    of the call go to one log file and its return value becomes the summary line. gate: file name stem (default: the
    function's module, plus the `mode` argument when there is one); passed: result -> bool (the gate's own verdict)."""
    def deco(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def run(*a, **k):
            global current
            name = os.environ.get(ENV, "")
            if not name or current is not None:          # off, or already inside a logged episode
                return fn(*a, **k)
            args = sig.bind(*a, **k).arguments
            if name.lower() in ("1", "true", "yes"):
                src = args.get("onnx_path", args.get("onnx"))
                name = Path(src).stem if isinstance(src, (str, Path)) else "mixed"
            g = gate or fn.__module__.rsplit(".", 1)[-1]
            if "mode" in args:
                g += f"_{args['mode']}"
            current = ActionLog(name, g, args["seed"])
            result = None
            try:
                result = fn(*a, **k)
                return result
            finally:
                log, current = current, None
                log.close(result, None if passed is None or result is None else bool(passed(result)))
        return run
    return deco
