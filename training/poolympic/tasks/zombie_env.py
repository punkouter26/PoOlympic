"""ZOMBIE athlete environments (tasks.md Phase Z) — MATT's task recipe (matt_env.py, incl. every Rung 2 fix) on the
zombie body, with every size-dependent number Froude-scaled (poolympic/bodies.py: λ = 0.618, speeds × √λ, times × √λ,
angular rates ÷ √λ) plus the zombie movement personality as style rewards: forward hunch, arms held out in front,
wide stance, low shuffling feet.

Use with POOLYMPIC_BODY=zombie (contract / mdp then load the zombie's scene, actuators, default stance and gait clock).
"""

from __future__ import annotations

import math

import mujoco

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.actuator import XmlActuatorCfg
from mjlab.tasks.velocity import mdp as vel_mdp

from .. import bodies
from .. import contract as C
from . import matt_env as M
from . import mdp

Z = bodies.BODIES["zombie"]
LAM, SS, TS = Z.length_scale, Z.speed_scale, Z.time_scale
WS = 1.0 / TS                     # angular-rate scale


def _default_root_z() -> float:
    m = mujoco.MjModel.from_xml_path(str(Z.scene_xml))
    return float(m.key("default").qpos[2])


ROOT_Z = _default_root_z()                       # crouched zombie stance, ~0.562 m
FALL_Z = 0.55 * ROOT_Z / M.DEFAULT_ROOT_Z        # MATT's fall rule at the same fraction of standing pelvis height
# style targets
HUNCH_DEG = 20.0                                  # torso lean
STANCE_WIDTH = 0.22                               # m between foot centres (hips 0.12 m apart)
FOOT_MAX_H = 0.05                                 # m, foot box centre height allowed in swing (stance ≈ 0.02)


def zombie_entity_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(str(Z.robot_xml)),
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
    if C.BODY.name != "zombie":
        raise RuntimeError("zombie tasks need POOLYMPIC_BODY=zombie (contract/mdp load the active body's scene)")


def _scale_xy_range(r: dict, k: float) -> dict:
    return {a: (lo * k, hi * k) for a, (lo, hi) in r.items()}


def _zombify(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Swap in the zombie body and scale MATT's size-dependent numbers (shared by every rung)."""
    cfg.scene.entities["robot"] = zombie_entity_cfg()
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
    rw["torques"].weight = rw["torques"].weight / Z.torque_scale ** 2      # same penalty for the same relative effort
    # personality: hunch instead of upright; arms held forward at every speed; wide stance
    rw.pop("upright", None)
    rw["hunch"] = RewardTermCfg(func=mdp.torso_pitch_tracking, weight=1.0,
                                params={"target_deg": HUNCH_DEG, "std_deg": 15.0})
    rw["arms_forward"] = RewardTermCfg(func=envs_mdp.posture, weight=0.5, params={
        "std": {".*": 0.3}, "asset_cfg": SceneEntityCfg("robot", joint_names=("shoulder_.*", "elbow_.*"))})
    rw["wide_stance"] = RewardTermCfg(func=mdp.feet_width, weight=0.5, params={"target": STANCE_WIDTH, "std": 0.06})
    # full self-collision: more contacts per env
    cfg.sim.nconmax, cfg.sim.njmax = 160, 800
    cfg.viewer.distance = 3.5 * LAM
    return cfg


def zombie_rung0_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rung 0 — stand + shove/cube recovery in the zombie stance."""
    _check_body()
    return _zombify(M.matt_rung0_env_cfg(play=play))


def zombie_rung1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rung 1 — forward shuffle/run at a commanded speed (scaled envelope), low feet."""
    _check_body()
    cfg = _zombify(M.matt_rung1_env_cfg(play=play))
    cmd = cfg.commands["athlete"]
    cmd.ranges.lin_vel_x = (0.3 * SS, 1.0 * SS)
    cmd.ranges.ang_vel_z = (-0.5 * WS, 0.5 * WS)
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
        {"step": s["step"], "lin_vel_x": (s["lin_vel_x"][0] * SS, s["lin_vel_x"][1] * SS)}
        for s in cfg.curriculum["command_vel"].params["velocity_stages"]]
    rw = cfg.rewards
    rw["air_time"].params.update({"threshold_min": 0.1 * TS, "threshold_max": 0.6 * TS, "command_threshold": 0.3 * SS})
    rw["posture"].params.update({"walking_threshold": 0.1 * SS, "running_threshold": 1.8 * SS})
    rw["shuffle"] = RewardTermCfg(func=mdp.feet_low, weight=0.5, params={"max_height": FOOT_MAX_H, "std": 0.03})
    if play:
        cmd.ranges.lin_vel_x = (0.5 * SS, 4.0 * SS)
    return cfg


def _scaled_env(env: dict) -> dict:
    return {"lin_vel_x": (env["lin_vel_x"][0] * SS, env["lin_vel_x"][1] * SS),
            "lin_vel_y": (env["lin_vel_y"][0] * SS, env["lin_vel_y"][1] * SS),
            "ang_vel_z": (env["ang_vel_z"][0] * WS, env["ang_vel_z"][1] * WS)}


RUNG2_STAGES = [dict(_scaled_env(s), step=s["step"]) for s in M.RUNG2_STAGES]
RUNG2_TRAIN_ENVELOPE = _scaled_env(M.RUNG2_TRAIN_ENVELOPE)


def zombie_rung2_base_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rung 2 first run = MATT's r2_v1 recipe (matt_rung2_env_cfg) on the scaled envelope: direct yaw commands with the
    strong yaw kernels, 15 % stops, MATT's RUNG2_STAGES curriculum — none of the later r2 additions (symmetric runner,
    lateral-acceleration cap, sprint focus, sharp linear kernel). Starting straight on the final recipe (z2_v1) drove
    the warm-started zombie into standing still (track_lin 1.52 -> 0.30, velocity error rising)."""
    _check_body()
    cfg = zombie_rung1_env_cfg(play=play)
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


def zombie_rung2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rung 2 fine-tune — MATT's final recipe (r2_v8: symmetric runner, full widened
    envelope, lateral-acceleration cap, sharp linear tracking, sprint focus) on the scaled envelope. Warm start: the best
    base-run checkpoint (z2_v2, PoOlympic-Zombie-Rung2-Omni-Base) via tools/warm_start.py."""
    _check_body()
    cfg = zombie_rung1_env_cfg(play=play)
    cmd = cfg.commands["athlete"]
    cmd.heading_command = False
    cmd.rel_heading_envs = 0.0
    cmd.rel_standing_envs = 0.20
    cmd.resampling_time_range = (3.0 * TS, 8.0 * TS)
    s0 = RUNG2_STAGES[0]
    cmd.ranges = mdp.AthleteCommandCfg.Ranges(lin_vel_x=s0["lin_vel_x"], lin_vel_y=s0["lin_vel_y"],
                                              ang_vel_z=s0["ang_vel_z"], heading=None)
    cmd.max_lateral_accel = 4.0                   # accelerations do not scale (Froude): same cap as MATT
    cmd.sprint_fraction = 0.3
    cmd.sprint_vx = (2.5 * SS, 4.0 * SS)
    cmd.sprint_wz = 0.6 * WS
    rw = cfg.rewards
    rw["track_lin"] = RewardTermCfg(func=vel_mdp.track_linear_velocity, weight=2.0,
                                    params={"command_name": "athlete", "std": 0.3 * SS})
    rw["track_ang"] = RewardTermCfg(func=vel_mdp.track_angular_velocity, weight=2.0,
                                    params={"command_name": "athlete", "std": 0.5 * WS})
    rw["track_ang_coarse"] = RewardTermCfg(func=vel_mdp.track_angular_velocity, weight=1.0,
                                           params={"command_name": "athlete", "std": 1.0 * WS})
    rw["posture"].weight = 0.25
    # fine-tune after the base run (z2_v2): the curriculum already ran — full widened envelope from the start
    # (MATT's Sym2+ tasks did the same)
    env = RUNG2_TRAIN_ENVELOPE
    cmd.ranges.lin_vel_x, cmd.ranges.lin_vel_y, cmd.ranges.ang_vel_z = env["lin_vel_x"], env["lin_vel_y"], env["ang_vel_z"]
    cfg.curriculum = {"command_vel": CurriculumTermCfg(func=vel_mdp.commands_vel, params={
        "command_name": "athlete", "velocity_stages": [dict(env, step=0)]})}
    if play:
        cmd.ranges.lin_vel_x, cmd.ranges.lin_vel_y, cmd.ranges.ang_vel_z = (
            RUNG2_TRAIN_ENVELOPE["lin_vel_x"], RUNG2_TRAIN_ENVELOPE["lin_vel_y"], RUNG2_TRAIN_ENVELOPE["ang_vel_z"])
    return cfg
