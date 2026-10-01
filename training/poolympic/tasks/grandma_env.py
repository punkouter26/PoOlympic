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
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity import mdp as vel_mdp

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
    cfg.scene.num_envs = 4096        # g0_v1 was launched without --env.scene.num-envs and trained ONE environment
    cfg.viewer.distance = 3.5 * LAM
    return cfg


def grandma_rung0_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rung 0 — stand + shove / cube recovery in the stooped soft-kneed stance."""
    _check_body()
    return _grandmafy(M.matt_rung0_env_cfg(play=play))


# ---- locomotion rungs: zombie_env's chain (the recipes that worked there), on GRANDMA's scale ----------------------
FOOT_MAX_H = 0.09            # m, foot box centre height allowed in swing (stance ≈ 0.03): low, careful steps


def grandma_rung1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rung 1 — forward walk / run at a commanded speed (Froude-scaled envelope), short careful steps (low feet)."""
    _check_body()
    cfg = _grandmafy(M.matt_rung1_env_cfg(play=play))
    cmd = cfg.commands["athlete"]
    cmd.ranges.lin_vel_x = (0.3 * SS, 1.0 * SS)
    cmd.ranges.ang_vel_z = (-0.5 * WS, 0.5 * WS)
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
        {"step": s["step"], "lin_vel_x": (s["lin_vel_x"][0] * SS, s["lin_vel_x"][1] * SS)}
        for s in cfg.curriculum["command_vel"].params["velocity_stages"]]
    rw = cfg.rewards
    rw["air_time"].params.update({"threshold_min": 0.1 * TS, "threshold_max": 0.6 * TS, "command_threshold": 0.3 * SS})
    rw["posture"].params.update({"walking_threshold": 0.1 * SS, "running_threshold": 1.8 * SS})
    rw["careful_steps"] = RewardTermCfg(func=mdp.feet_low, weight=0.3, params={"max_height": FOOT_MAX_H, "std": 0.03})
    if play:
        cmd.ranges.lin_vel_x = (0.5 * SS, 4.0 * SS)
    return cfg


def _scaled_env(env: dict) -> dict:
    return {"lin_vel_x": (env["lin_vel_x"][0] * SS, env["lin_vel_x"][1] * SS),
            "lin_vel_y": (env["lin_vel_y"][0] * SS, env["lin_vel_y"][1] * SS),
            "ang_vel_z": (env["ang_vel_z"][0] * WS, env["ang_vel_z"][1] * WS)}


RUNG2_STAGES = [dict(_scaled_env(s), step=s["step"]) for s in M.RUNG2_STAGES]
RUNG2_TRAIN_ENVELOPE = _scaled_env(M.RUNG2_TRAIN_ENVELOPE)


def grandma_rung2_base_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rung 2 first run = MATT's r2_v1 recipe on the scaled envelope (= zombie_rung2_base_env_cfg, the zombie's z2_v2):
    direct yaw commands with the strong yaw kernels, 15 % stops, the RUNG2_STAGES curriculum; none of the later
    additions. On the zombie, starting straight on the final recipe drove the warm-started policy into standing still.
    Warm start: the best Rung 1 checkpoint via tools/warm_start.py."""
    _check_body()
    cfg = grandma_rung1_env_cfg(play=play)
    cmd = cfg.commands["athlete"]
    cmd.heading_command = False
    cmd.rel_heading_envs = 0.0
    cmd.rel_standing_envs = 0.15
    cmd.resampling_time_range = (3.0 * TS, 8.0 * TS)
    s0 = RUNG2_STAGES[0]
    cmd.ranges = mdp.AthleteCommandCfg.Ranges(lin_vel_x=s0["lin_vel_x"], lin_vel_y=s0["lin_vel_y"],
                                              ang_vel_z=s0["ang_vel_z"], heading=None)
    rw = cfg.rewards
    rw["track_ang"] = RewardTermCfg(func=vel_mdp.track_angular_velocity, weight=2.0,
                                    params={"command_name": "athlete", "std": 0.5 * WS})
    rw["track_ang_coarse"] = RewardTermCfg(func=vel_mdp.track_angular_velocity, weight=1.0,
                                           params={"command_name": "athlete", "std": 1.0 * WS})
    rw["posture"].weight = 0.25
    cfg.curriculum = {"command_vel": CurriculumTermCfg(func=vel_mdp.commands_vel, params={
        "command_name": "athlete", "velocity_stages": RUNG2_STAGES})}
    if play:
        last = RUNG2_STAGES[-1]
        cmd.ranges.lin_vel_x, cmd.ranges.lin_vel_y, cmd.ranges.ang_vel_z = last["lin_vel_x"], last["lin_vel_y"], last["ang_vel_z"]
    return cfg


def grandma_rung2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rung 2 fine-tune = the zombie's final recipe (z2_v11, zombie_rung2_wobble_env_cfg) on GRANDMA's scale, in one
    function: symmetric runner, full widened envelope from the start, lateral-acceleration cap 3 m/s², wide linear kernel
    (std 0.5) with speed weights 3 + 1.5, yaw terms on the stride-filtered yaw rate, yaw-wobble penalty, no sprint
    focus. Each piece fixed a failure on the weak zombie (standing still once yaw paid; braking for hard turns; charging
    the per-stride pelvis rock as yaw error) that a 60 %-strength body is likely to share. Warm start: the base run."""
    _check_body()
    cfg = grandma_rung1_env_cfg(play=play)
    cmd = cfg.commands["athlete"]
    cmd.heading_command = False
    cmd.rel_heading_envs = 0.0
    cmd.rel_standing_envs = 0.20
    cmd.resampling_time_range = (3.0 * TS, 8.0 * TS)
    env = RUNG2_TRAIN_ENVELOPE
    cmd.ranges = mdp.AthleteCommandCfg.Ranges(lin_vel_x=env["lin_vel_x"], lin_vel_y=env["lin_vel_y"],
                                              ang_vel_z=env["ang_vel_z"], heading=None)
    cmd.max_lateral_accel = 3.0
    cmd.sprint_fraction = 0.0
    cmd.sprint_vx = (2.5 * SS, 4.0 * SS)
    cmd.sprint_wz = 0.6 * WS
    rw = cfg.rewards
    rw["track_lin"] = RewardTermCfg(func=vel_mdp.track_linear_velocity, weight=3.0,
                                    params={"command_name": "athlete", "std": 0.5})
    rw["track_lin_coarse"].weight = 1.5
    rw["track_ang"] = RewardTermCfg(func=vel_mdp.track_angular_velocity, weight=2.0,
                                    params={"command_name": "athlete", "std": 0.5 * WS})
    rw["track_ang_coarse"] = RewardTermCfg(func=vel_mdp.track_angular_velocity, weight=1.0,
                                           params={"command_name": "athlete", "std": 1.0 * WS})
    rw["track_yaw_filt"] = RewardTermCfg(func=mdp.track_yaw_rate_filtered, weight=2.0,
                                         params={"command_name": "athlete", "std": 0.5 * WS, "tau": 0.5})
    rw["yaw_filt_l1"] = RewardTermCfg(func=mdp.yaw_rate_filtered_l1, weight=-0.5,
                                      params={"command_name": "athlete", "tau": 0.5})
    rw["yaw_wobble"] = RewardTermCfg(func=mdp.yaw_wobble_l2, weight=-1.0, params={"command_name": "athlete", "tau": 0.5})
    rw["posture"].weight = 0.25
    cfg.curriculum = {"command_vel": CurriculumTermCfg(func=vel_mdp.commands_vel, params={
        "command_name": "athlete", "velocity_stages": [dict(env, step=0)]})}
    return cfg
