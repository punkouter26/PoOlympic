"""GRANDMA athlete environments (2026-09-30) — MATT's task recipe (matt_env.py) on the GRANDMA body (poolympic/bodies.py:
1.60 m, 65 kg, λ = 0.871, 60 % of the size-scaled torque, full self-collision), with every size-dependent number
Froude-scaled exactly like zombie_env.py, plus her movement personality "frail but steady" (user decision) as style
rewards: a slight forward stoop instead of MATT's upright trunk, a steady hip-width stance.

Use with POOLYMPIC_BODY=grandma (contract / mdp then load her scene, actuators, default stance and gait clock).
"""

from __future__ import annotations

import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg

from .. import bodies
from .. import contract as C
from . import matt_env as M
from . import mdp

G = bodies.BODIES["grandma"]
LAM, SS, TS = G.length_scale, G.speed_scale, G.time_scale
WS = 1.0 / TS                     # angular-rate scale


def _default_root_z() -> float:
    m = mujoco.MjModel.from_xml_path(str(G.scene_xml))
    return float(m.key("default").qpos[2])


ROOT_Z = _default_root_z()                       # soft-kneed stance, ~0.678 m
FALL_Z = 0.55 * ROOT_Z / M.DEFAULT_ROOT_Z        # MATT's fall rule at the same fraction of standing pelvis height
# style targets
STOOP_DEG = 10.0                                  # trunk lean (= the default pose's abdomen_flex)
STANCE_WIDTH = 0.26                               # m between foot centres (her hip joints are 0.25 m apart)


def grandma_entity_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(str(G.robot_xml)),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, ROOT_Z),
            joint_pos={n: v for n, v in zip(mdp.CONTRACT_ACTUATORS, mdp.CONTRACT_DEFAULTS)} | {"toe_.*": 0.0},
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=(".*",), delay_min_lag=0, delay_max_lag=4,
                                      delay_update_period=4000),),
            soft_joint_pos_limit_factor=0.95,
        ),
    )


def _check_body():
    if C.BODY.name != "grandma":
        raise RuntimeError("grandma tasks need POOLYMPIC_BODY=grandma (contract/mdp load the active body's scene)")


def _scale_xy_range(r: dict, k: float) -> dict:
    return {a: (lo * k, hi * k) for a, (lo, hi) in r.items()}


def _grandmafy(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Swap in the GRANDMA body and scale MATT's size-dependent numbers (= zombie_env._zombify's scaling)."""
    cfg.scene.entities["robot"] = grandma_entity_cfg()
    cfg.terminations["pelvis_low"].params["minimum_height"] = FALL_Z
    ev = cfg.events
    ev["reset_base"].params["pose_range"]["x"] = (-0.05 * LAM, 0.05 * LAM)
    ev["reset_base"].params["pose_range"]["y"] = (-0.05 * LAM, 0.05 * LAM)
    ev["reset_base"].params["velocity_range"] = _scale_xy_range(ev["reset_base"].params["velocity_range"], SS)
    ev["push_robot"].params["velocity_range"] = _scale_xy_range(ev["push_robot"].params["velocity_range"], SS)
    ev["drop_cube"].params.update({"height_above_shoulder": 1.5 * LAM, "shoulder_above_pelvis": 0.55 * LAM,
                                   "xy_jitter": 0.15 * LAM})
    rw = cfg.rewards
    if "height" in rw:
        rw["height"].params.update({"target": ROOT_Z, "std": rw["height"].params["std"] * LAM})
    if "near_origin" in rw:
        rw["near_origin"].params["std"] *= LAM
    if "still_lin" in rw:
        rw["still_lin"].params["std"] *= SS
    rw["joint_vel_limit"].params["limit"] = 18.0 * WS
    rw["torques"].weight = rw["torques"].weight / G.torque_scale ** 2      # same penalty for the same relative effort
    # personality "frail but steady": a slight stoop instead of upright; a steady hip-width stance
    rw.pop("upright", None)
    rw["stoop"] = RewardTermCfg(func=mdp.torso_pitch_tracking, weight=1.0, params={"target_deg": STOOP_DEG, "std_deg": 10.0})
    rw["steady_stance"] = RewardTermCfg(func=mdp.feet_width, weight=0.5, params={"target": STANCE_WIDTH, "std": 0.06})
    # full self-collision: more contacts per env
    cfg.sim.nconmax, cfg.sim.njmax = 160, 800
    cfg.viewer.distance = 3.5 * LAM
    return cfg


def grandma_rung0_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rung 0 — stand + shove / cube recovery in the stooped soft-kneed stance."""
    _check_body()
    return _grandmafy(M.matt_rung0_env_cfg(play=play))
