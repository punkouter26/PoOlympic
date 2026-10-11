"""Smoke test of a registered task: build the env (few envs), step it with random actions, check that every observation
group and reward term is finite, mirror the observations if the task trains with the symmetric runner. No learning.

Usage (body = $POOLYMPIC_BODY): uv run python tools/smoke_task.py PoOlympic-Kim-Rung2-Bio [--envs 16] [--steps 120] [--play]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--action-std", type=float, default=0.3)
    a = ap.parse_args()

    import poolympic.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg, load_runner_cls

    cfg = load_env_cfg(a.task, play=a.play)
    cfg.scene.num_envs = a.envs
    torch.manual_seed(0)
    env = ManagerBasedRlEnv(cfg=cfg, device=a.device)
    obs = env.reset()[0]
    dims = {k: tuple(v.shape) for k, v in obs.items()}
    n_act = sum(env.action_manager.action_term_dim)
    ok, resets = True, 0
    sums: dict[str, float] = {}
    rm = env.reward_manager
    for _ in range(a.steps):
        obs, rew, term, trunc, _ = env.step(torch.randn(a.envs, n_act, device=a.device) * a.action_std)
        resets += int((term | trunc).sum())
        ok &= all(bool(torch.isfinite(v).all()) for v in obs.values()) and bool(torch.isfinite(rew).all())
        for name, tc in zip(rm._term_names, rm._term_cfgs):
            v = tc.func(env, **tc.params).float()
            ok &= bool(torch.isfinite(v).all())
            sums[name] = sums.get(name, 0.0) + float(v.mean()) * tc.weight / a.steps
    print(f"{a.task}: obs {dims}, {a.steps} steps x {a.envs} envs, {resets} resets, finite {ok}")
    print("  weighted reward means:", {k: round(v, 4) for k, v in sums.items() if k != "terminated"})
    if "symmetry" in (load_runner_cls(a.task) or type).__module__:
        from poolympic.tasks.symmetry import mirror_obs
        twice = {k: float((mirror_obs(mirror_obs(v)) - v).abs().max()) for k, v in obs.items()}
        print("  mirror(mirror(obs)) - obs:", twice)
        ok &= all(v < 1e-6 for v in twice.values())
    for attr, label in (("_poolympic_gust", "shove level (m/s)"), ("_poolympic_bio", "fatigue")):
        st = getattr(env, attr, None)
        if st is not None:
            if attr == "_poolympic_gust":
                print(f"  {label}: {st:.3f}")
            else:
                fr = env.sim.model.actuator_forcerange[:, st.ctrl]
                print(f"  {label}: mean level per group {[round(float(x), 4) for x in st.level.mean(0)]}, "
                      f"force range / nominal min {float((fr.abs() / st.base.abs()).min()):.3f} max {float((fr.abs() / st.base.abs()).max()):.3f}")
    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
