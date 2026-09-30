"""Throughput benchmark for an athlete task — environment stepping only, NO learning (safe to run any time the GPU is free).

A trained actor (rsl_rl checkpoint) drives the envs so contact counts and resets look like a real run. Reports env-steps/s,
the share of each env.step phase (physics graph vs the eager torch managers), per-reward-term cost, mean solver
iterations, peak contacts / constraints per world (to size nconmax / njmax) and GPU memory.

Usage (from training/):
  uv run python tools/bench_env.py                                   # Rung 2 final recipe (r2_v8), 4096 envs
  uv run python tools/bench_env.py --task stance --envs 8192
  uv run python tools/bench_env.py --tolerance 1e-6 --ls-iterations 10 --cubes 1 --njmax 160 --nconmax 40
Knobs change the TRAINING copy of the physics only (CPU MuJoCo / Unity keep scene_matt.xml); every change must still
pass G1 on CPU MuJoCo before it is adopted (DESIGN §4: the golden reference is CPU MuJoCo).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

DEFAULT_CKPT = ROOT / "runs" / "matt_rung2" / "2026-09-28_04-54-44_r2_v8" / "model_600.pt"


def make_cfg(task: str):
    if task.endswith("_v5") and "rung" in task:
        from poolympic.tasks import matt_env
        return getattr(matt_env, f"matt_{task}_env_cfg")()
    if task == "stance":
        from poolympic.tasks.stance_env import matt_stance_env_cfg
        return matt_stance_env_cfg()
    from poolympic.tasks import matt_env
    return getattr(matt_env, f"matt_{task}_env_cfg")()


def load_actor(ckpt: Path, obs_dim: int, device: str):
    """rsl_rl actor: mlp((obs - mean) / (std + 0.01)); a checkpoint with fewer inputs gets zero columns (Rung S warm start)."""
    from poolympic.policy_export import make_mlp
    sd = torch.load(ckpt, map_location=device, weights_only=False)["actor_state_dict"]
    mean, std = sd["obs_normalizer._mean"][0], sd["obs_normalizer._std"][0] + 1e-2
    n = mean.shape[0]
    mlp = make_mlp(n).to(device)
    mlp.load_state_dict({k[len("mlp."):]: v for k, v in sd.items() if k.startswith("mlp.")})
    mlp.eval()

    @torch.no_grad()
    def act(obs):
        return mlp((obs[:, :n] - mean) / std)
    return act


def apply_knobs(cfg, a):
    mj = cfg.sim.mujoco
    for k in ("tolerance", "iterations", "ls_iterations", "ls_tolerance"):
        v = getattr(a, k)
        if v is not None:
            setattr(mj, k, v)
    if a.njmax is not None:
        cfg.sim.njmax = a.njmax
    if a.nconmax is not None:
        cfg.sim.nconmax = a.nconmax
    if a.cubes is not None:
        names = [n for n in cfg.scene.entities if n.startswith("cube")]
        for n in names[a.cubes:]:
            cfg.scene.entities.pop(n)
        keep = tuple(names[:a.cubes])
        if keep:
            cfg.events["drop_cube"].params["cube_names"] = keep
        else:
            cfg.events.pop("drop_cube", None)
    cfg.scene.num_envs = a.envs
    return cfg


class Timers:
    """cuda-synchronised wall time per wrapped callable (only in the --breakdown pass: syncing costs a little)."""

    def __init__(self):
        self.t = defaultdict(float)
        self.n = defaultdict(int)

    def wrap(self, obj, attr, label):
        fn = getattr(obj, attr)

        def timed(*args, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(*args, **kw)
            torch.cuda.synchronize()
            self.t[label] += time.perf_counter() - t0
            self.n[label] += 1
            return out
        setattr(obj, attr, timed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="rung2_sym5")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=240, help="policy steps timed (24 = one PPO iteration)")
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--tolerance", type=float)
    ap.add_argument("--iterations", type=int)
    ap.add_argument("--ls-iterations", dest="ls_iterations", type=int)
    ap.add_argument("--ls-tolerance", dest="ls_tolerance", type=float)
    ap.add_argument("--njmax", type=int)
    ap.add_argument("--nconmax", type=int)
    ap.add_argument("--cubes", type=int, help="pool cubes per env (training copy only; default = task's 4)")
    ap.add_argument("--no-breakdown", action="store_true")
    ap.add_argument("--json", type=Path, help="append the result as one JSON line")
    a = ap.parse_args()

    dev = "cuda:0"
    torch.manual_seed(0)
    cfg = apply_knobs(make_cfg(a.task), a)
    env = ManagerBasedRlEnv(cfg=cfg, device=dev)
    obs = env.reset()[0]["actor"]
    act = load_actor(a.ckpt, obs.shape[1], dev)
    wd = env.sim.data

    def run(n, track=False):
        nonlocal obs
        peak_con = peak_efc = 0
        niter = 0.0
        for _ in range(n):
            o, *_ = env.step(act(obs))
            obs = o["actor"]
            if track:
                peak_efc = max(peak_efc, int(wd.nefc.max()))
                peak_con = max(peak_con, int(wd.nacon[0]))
                niter += float(wd.solver_niter.float().mean())
        return peak_con, peak_efc, niter / max(n, 1)

    run(a.warmup)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    run(a.steps)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    sps = a.envs * a.steps / wall
    peak_con, peak_efc, niter = run(48, track=True)

    res = {"task": a.task, "envs": a.envs, "steps_per_s": round(sps), "s_per_ppo_collection": round(24 * wall / a.steps, 3),
           "solver_iters_mean": round(niter, 2), "peak_contacts_per_world": round(peak_con / a.envs, 1),
           "peak_nefc_world": peak_efc, "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
           "knobs": {k: getattr(a, k) for k in ("tolerance", "iterations", "ls_iterations", "ls_tolerance", "njmax",
                                                 "nconmax", "cubes") if getattr(a, k) is not None}}

    if not a.no_breakdown:
        T = Timers()
        for obj, attr, label in [
            (env.sim, "step", "physics (mj_step graph)"), (env.sim, "forward", "physics (forward graph)"),
            (env.sim, "sense", "sensors"), (env.scene, "update", "scene.update"),
            (env.scene, "write_data_to_sim", "scene.write"), (env.action_manager, "apply_action", "actions"),
            (env.action_manager, "process_action", "actions"),
            (env.termination_manager, "compute", "terminations"), (env.reward_manager, "compute", "rewards"),
            (env.metrics_manager, "compute", "metrics"), (env.metrics_manager, "compute_substep", "metrics"),
            (env.event_manager, "apply", "events"), (env, "_reset_idx", "resets"),
            (env.command_manager, "compute", "commands"), (env.observation_manager, "compute", "observations"),
        ]:
            T.wrap(obj, attr, label)
        rm = env.reward_manager
        for name, tc in zip(rm._term_names, rm._term_cfgs):
            f = tc.func
            box = type("B", (), {})()
            box.f = f
            T.wrap(box, "f", f"reward:{name}")
            tc.func = box.f
        t0 = time.perf_counter()
        n = 48
        for _ in range(n):
            torch.cuda.synchronize()
            p0 = time.perf_counter()
            actions = act(obs)
            torch.cuda.synchronize()
            T.t["policy inference"] += time.perf_counter() - p0
            o, *_ = env.step(actions)
            obs = o["actor"]
        total = time.perf_counter() - t0
        top = {k: v for k, v in T.t.items() if not k.startswith("reward:")}
        res["breakdown_pct"] = {k: round(100 * v / total, 1) for k, v in sorted(top.items(), key=lambda kv: -kv[1])}
        res["reward_terms_ms_per_step"] = {k[7:]: round(1e3 * v / n, 2)
                                           for k, v in sorted(T.t.items(), key=lambda kv: -kv[1]) if k.startswith("reward:")}

    rm = env.reward_manager                      # weighted per-step value of every term on the current state
    res["reward_terms_weighted_mean"] = {n: round(float(tc.func(env, **tc.params).float().mean()) * tc.weight, 4)
                                         for n, tc in zip(rm._term_names, rm._term_cfgs) if n != "terminated"}
    print(json.dumps(res, indent=1))
    if a.json:
        with a.json.open("a") as fh:
            fh.write(json.dumps(res) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
