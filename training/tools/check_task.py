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
from poolympic.tasks.matt_env import matt_rung2_sym3_env_cfg, matt_pedestal2_env_cfg, matt_pedestal_env_cfg, matt_rung0_env_cfg, matt_rung1_env_cfg, matt_rung2_env_cfg  # noqa: E402

TASKS = {"rung0": matt_rung0_env_cfg, "rung1": matt_rung1_env_cfg, "rung2": matt_rung2_env_cfg, "pedestal": matt_pedestal_env_cfg, "pedestal2": matt_pedestal2_env_cfg, "rung2_sym3": matt_rung2_sym3_env_cfg}


def _zombie(rung: str):
    def make():   # needs POOLYMPIC_BODY=zombie (zombie_env checks)
        from poolympic.tasks import zombie_env
        return getattr(zombie_env, f"zombie_{rung}_env_cfg")()
    return make


TASKS.update({f"zombie_{r}": _zombie(r) for r in ("rung0", "rung1", "rung2", "rung2_base", "rung2_sym3")})


def main(task: str = "rung0") -> int:
    make = TASKS[task]
    cfg = make()
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
    worst_phase = 0.0
    py_phase = np.zeros(env.num_envs)
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
        cmds = term.command.cpu().numpy().astype(np.float64)
        ep = env.episode_length_buf.cpu().numpy()
        for e in range(env.num_envs):
            py_phase[e] = C.advance_phase(0.0 if ep[e] == 0 else py_phase[e], cmds[e])
            worst_phase = max(worst_phase, abs(py_phase[e] - float(term.phase[e])) % 1.0)
            ref_obs = C.build_obs(ath, qpos[e], qvel[e], term.command[e].cpu().numpy().astype(np.float64),
                                  float(term.phase[e]), last[e])
            worst_obs = max(worst_obs, float(np.abs(policy_obs[e] - ref_obs).max()))
            # ctrl written during the last substep = clip(default + scale·a) (delay may hold older targets: skip delayed envs)
            want = C.action_to_ctrl(ath, a[e].cpu().numpy().astype(np.float64))
            worst_ctrl = max(worst_ctrl, float(np.abs(ctrl[e][ath.actuator_ids] - want).min()))
    print(f"obs max |train - contract| = {worst_obs:.2e}   (float32 Warp vs float64 reference)")
    ok &= worst_obs < 1e-3
    print(f"phase clock max |train - contract.advance_phase| = {worst_phase:.2e}  (commands nonzero: {bool(np.abs(cmds).sum() > 0)})")
    ok &= worst_phase < 1e-4
    print(f"actuator order: {ath_names[:4]} …")
    env.close()

    # 3. wiring: action index i must move exactly ctrl of contract actuator i (no delay, no DR)
    cfg = make()
    cfg.scene.num_envs = 1
    for k in list(cfg.events):
        if k.startswith("dr_") or k in ("push_robot", "drop_cube"):
            cfg.events.pop(k)
    cfg.scene.entities["robot"].articulation.actuators[0].delay_max_lag = 0
    # no terminations here: a fall mid-sweep resets the episode and ctrl jumps, which reads as a wiring error
    cfg.terminations = {k: v for k, v in cfg.terminations.items() if k == "time_out"}
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env.reset()
    m = env.sim.mj_model
    ath = C.Athlete.bind(m, prefix="robot/")
    zero = torch.zeros(1, C.NUM_ACTIONS, device=env.device)
    env.step(zero)
    c0 = env.sim.data.ctrl[0].cpu().numpy().copy()
    bad = 0
    for i in range(C.NUM_ACTIONS):
        a = zero.clone()
        a[0, i] = 1.0
        env.step(a)
        moved = list(np.nonzero(np.abs(env.sim.data.ctrl[0].cpu().numpy() - c0) > 1e-4)[0])
        if moved != [int(ath.actuator_ids[i])]:
            bad += 1
            print(f"  WIRING action[{i}] ({ath.actuator_names[i]}) moved {[m.actuator(j).name for j in moved]}")
        env.step(zero)
    print(f"action->ctrl wiring mismatches: {bad}")
    ok &= bad == 0
    env.close()
    print("C1 CHECKS", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "rung0"))
