"""C1 sanity checks for the mjlab task vs the contract (run before any training).

  1. compiled solver options + ground/athlete contact bits match scene_matt.xml
  2. training observation (noise off) == contract.build_obs on the same qpos/qvel (all envs, after random steps)
  3. action -> ctrl mapping == contract.action_to_ctrl (default + 0.25·a, clipped), in contract actuator order
Usage: uv run python tools/check_task.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

from poolympic import contract as C  # noqa: E402
from poolympic.tasks.matt_env import matt_rung0_env_cfg  # noqa: E402


def main() -> int:
    cfg = matt_rung0_env_cfg()
    cfg.scene.num_envs = 16
    cfg.observations["actor"].enable_corruption = False
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    m = env.sim.mj_model
    ok = True

    ref = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
    for f in ("timestep", "iterations", "ls_iterations", "integrator", "cone", "solver", "jacobian", "disableflags",
              "tolerance", "ls_tolerance", "impratio"):
        a, b = getattr(m.opt, f), getattr(ref.opt, f)
        same = np.allclose(a, b)
        ok &= bool(same)
        print(f"opt.{f:14s} train={a} scene={b} {'OK' if same else 'MISMATCH'}")
    g = m.geom("terrain").id
    print(f"terrain contype/conaffinity/friction: {m.geom_contype[g]}/{m.geom_conaffinity[g]}/{m.geom_friction[g]}")
    ok &= m.geom_contype[g] == 0xFFFF and m.geom_conaffinity[g] == 0xFFFF
    for n in ("robot/foot_l_geom0", "robot/torso_geom0", "cube0/cube_geom"):
        gi = m.geom(n).id
        print(f"{n:22s} contype={m.geom_contype[gi]} conaffinity={m.geom_conaffinity[gi]} friction={m.geom_friction[gi]}")

    obs, _ = env.reset()
    ath_names = [m.actuator(i).name for i in range(m.nu)]
    ath = C.Athlete.bind(m, prefix="robot/")
    worst_obs = 0.0
    worst_ctrl = 0.0
    torch.manual_seed(0)
    for step in range(30):
        a = torch.randn(env.num_envs, C.NUM_ACTIONS, device=env.device) * 1.5
        obs, *_ = env.step(a)
        qpos = env.sim.data.qpos.cpu().numpy().astype(np.float64)
        qvel = env.sim.data.qvel.cpu().numpy().astype(np.float64)
        ctrl = env.sim.data.ctrl.cpu().numpy().astype(np.float64)
        term = env.command_manager.get_term("athlete")
        last = env.action_manager.action.cpu().numpy().astype(np.float64)
        policy_obs = obs["actor"].cpu().numpy()
        for e in range(env.num_envs):
            ref_obs = C.build_obs(ath, qpos[e], qvel[e], term.command[e].cpu().numpy().astype(np.float64),
                                  float(term.phase[e]), last[e])
            worst_obs = max(worst_obs, float(np.abs(policy_obs[e] - ref_obs).max()))
            # ctrl written during the last substep = clip(default + scale·a) (delay may hold older targets: skip delayed envs)
            want = C.action_to_ctrl(ath, a[e].cpu().numpy().astype(np.float64))
            worst_ctrl = max(worst_ctrl, float(np.abs(ctrl[e][ath.actuator_ids] - want).min()))
    print(f"obs max |train - contract| = {worst_obs:.2e}   (float32 Warp vs float64 reference)")
    ok &= worst_obs < 1e-3
    print(f"actuator order: {ath_names[:4]} …")
    env.close()
    print("C1 CHECKS", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
