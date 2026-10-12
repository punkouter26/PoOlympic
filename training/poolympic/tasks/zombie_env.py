"""ZOMBIE athlete environments (tasks.md Phase Z) — MATT's task recipe (matt_env.py, incl. every Rung 2 fix) on the
zombie body, with every size-dependent number Froude-scaled (poolympic/bodies.py: λ = 0.618, speeds × √λ, times × √λ,
angular rates ÷ √λ) plus the zombie movement personality as style rewards: forward hunch, arms held out in front,
wide stance, low shuffling feet.

Use with POOLYMPIC_BODY=zombie (contract / mdp then load the zombie's scene, actuators, default stance and gait clock).
"""

from __future__ import annotations

import math

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as vel_mdp

from .. import bodies
from . import matt_env as M
from . import mdp

Z = bodies.BODIES["zombie"]
LAM, SS, TS = Z.length_scale, Z.speed_scale, Z.time_scale
WS = 1.0 / TS                     # angular-rate scale


ROOT_Z = M.scene_root_z(Z.scene_xml)             # crouched zombie stance, ~0.562 m
FALL_Z = M.body_fall_z(ROOT_Z)                   # MATT's fall rule at the same fraction of standing pelvis height
# style targets
HUNCH_DEG = 20.0                                  # torso lean
STANCE_WIDTH = 0.22                               # m between foot centres (hips 0.12 m apart)
FOOT_MAX_H = 0.05                                 # m, foot box centre height allowed in swing (stance ≈ 0.02)


def zombie_entity_cfg() -> EntityCfg:
    return M.body_entity_cfg(Z.robot_xml, ROOT_Z)


def _check_body():
    M.check_body("zombie")


def _zombify(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Swap in the zombie body and scale MATT's size-dependent numbers (matt_env.scale_to_body), plus its style."""
    cfg = M.scale_to_body(cfg, Z, zombie_entity_cfg(), ROOT_Z)
    rw = cfg.rewards
    # personality: hunch instead of upright; arms held forward at every speed; wide stance
    rw.pop("upright", None)
    rw["hunch"] = RewardTermCfg(func=mdp.torso_pitch_tracking, weight=1.0,
                                params={"target_deg": HUNCH_DEG, "std_deg": 15.0})
    rw["arms_forward"] = RewardTermCfg(func=envs_mdp.posture, weight=0.5, params={
        "std": {".*": 0.3}, "asset_cfg": SceneEntityCfg("robot", joint_names=("shoulder_.*", "elbow_.*"))})
    rw["wide_stance"] = RewardTermCfg(func=mdp.feet_width, weight=0.5, params={"target": STANCE_WIDTH, "std": 0.06})
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
    return M.scaled_envelope(env, SS, WS)


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


def zombie_rung2_sym3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rung 2 fine-tune = MATT's r2_v6 recipe (Sym3, his first G1 10/10): the final recipe WITHOUT the sharp linear
    kernel and WITHOUT sprint focus — symmetric runner, full widened envelope, lateral-acceleration cap, track_lin std
    0.5. On the zombie the sharp kernel + 30 % sprints (z2_v1, z2_v3) wrecked linear tracking (sprint RMS 0.15 -> 0.68)
    while the cap fixed yaw (turntable 2.5 -> 2.0 s, yaw segments 30 -> 50 / 50)."""
    cfg = zombie_rung2_env_cfg(play=play)
    cfg.commands["athlete"].sprint_fraction = 0.0
    cfg.rewards["track_lin"] = RewardTermCfg(func=vel_mdp.track_linear_velocity, weight=2.0,
                                             params={"command_name": "athlete", "std": 0.5})
    return cfg


def zombie_rung2_sym3yaw_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rung 2 fine-tune (z2_v5 / z2_v6, 2026-09-29) = Sym3 with a stronger, sharper yaw-rate kernel. z2_v3 (sharp linear
    kernel) tracked yaw (RMS 0.10) but gave up on speed (0.4-0.7 m/s); z2_v4 (wide linear kernel, same init) tracked
    speed (0.10-0.14) but let a gait yaw wobble grow to 0.56-0.77 rad/s (bar 0.382): reward balance, not capability.
    Keep the wide linear kernel, raise track_ang 2 -> 3 and sharpen it 0.5 -> 0.35 (× √λ-scaled rate)."""
    cfg = zombie_rung2_sym3_env_cfg(play=play)
    cfg.rewards["track_ang"] = RewardTermCfg(func=vel_mdp.track_angular_velocity, weight=3.0,
                                             params={"command_name": "athlete", "std": 0.35 * WS})
    return cfg


def zombie_rung2_yawgrad_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rung 2 fine-tune (z2_v7, 2026-09-29) = Sym3 (keeps z2_v4's running: 2.82 m/s at the 3.14 envelope top) + yaw
    terms with a gradient everywhere. On the z2_v4 line the yaw kernels paid ~nothing in training (track_ang 0.15 / 2.0,
    yaw error ~4x MATT's) — the misses are large, and mjlab's kernel also counts the roll/pitch rates of the shuffle.
    Adds a yaw-only kernel (std 1.0 x the scaled rate) and an L1 yaw-rate penalty. (z2_v6 from the z2_v3 line kept
    its precise steering but had stopped running: 0.14 m/s at a 2.0 m/s command.)"""
    cfg = zombie_rung2_sym3_env_cfg(play=play)
    cfg.rewards["track_yaw_only"] = RewardTermCfg(func=mdp.track_yaw_rate, weight=1.5,
                                                  params={"command_name": "athlete", "std": 1.0 * WS})
    cfg.rewards["yaw_l1"] = RewardTermCfg(func=mdp.yaw_rate_l1, weight=-0.5, params={"command_name": "athlete"})
    return cfg


def zombie_rung2_yawfilt_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rung 2 fine-tune (z2_v8) = Sym3 + yaw terms on the stride-filtered yaw rate (EMA tau 0.5 s). z2_v7's per-tick
    yaw-only kernel + L1 penalty fixed the turns (turntable 1.97 s, yaw fails 2/10) but the zombie stopped running
    (0.65 m/s at a 2.0 m/s command): a fast shuffle rocks the pelvis every stride and every per-tick yaw term charges for
    that wobble, so standing still was the cheapest way to cut "yaw error". Filtered terms punish sustained turn
    errors only."""
    cfg = zombie_rung2_sym3_env_cfg(play=play)
    cfg.rewards["track_yaw_filt"] = RewardTermCfg(func=mdp.track_yaw_rate_filtered, weight=2.0,
                                                  params={"command_name": "athlete", "std": 0.5 * WS, "tau": 0.5})
    cfg.rewards["yaw_filt_l1"] = RewardTermCfg(func=mdp.yaw_rate_filtered_l1, weight=-0.5,
                                               params={"command_name": "athlete", "tau": 0.5})
    return cfg


def zombie_rung2_yawfilt_cap3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """z2_v9 = z2_v8 + lateral-acceleration cap 4 -> 3 m/s² (|wz| <= 3 / |v|). Every z2 fine-tune that pushed yaw
    (v5, v7, v8) slowed the zombie down: at 70 % strength the physical way to follow a hard turn at speed is to brake.
    The G1 drill needs <= 3.0 m/s² (turns vx <= 1.5·√λ at |wz| <= 2·/√λ; 2.25 m/s² seen), MATT's cap was 4."""
    cfg = zombie_rung2_yawfilt_env_cfg(play=play)
    cfg.commands["athlete"].max_lateral_accel = 3.0
    return cfg


def zombie_rung2_yawfilt_cap3_lin3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """z2_v10 = z2_v9 + speed weights up (track_lin 2 -> 3, track_lin_coarse 1 -> 1.5). z2_v9 it 200 was the first seed
    pass and still ran (2.0 m/s -> 1.79) but by it 400 it had drifted to standing again (0.66 m/s): the zombie style
    rewards (low feet, hunch, arms forward, wide stance) are all easier slow, so once yaw pays, speed (2 + 1) loses.
    Warm start: z2_v9 it 200."""
    cfg = zombie_rung2_yawfilt_cap3_env_cfg(play=play)
    cfg.rewards["track_lin"].weight = 3.0
    cfg.rewards["track_lin_coarse"].weight = 1.5
    return cfg


def zombie_rung2_wobble_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """z2_v11 = z2_v10 + yaw-wobble penalty (per-tick yaw rate minus its stride-filtered value, squared, w -1.0).
    z2_v10 runs (2.0 -> 1.98 m/s, 2.8 -> 2.67) and steers (stride-averaged yaw 0.05-0.15, turntable 1.9 s); what fails
    the official per-tick bar (0.382) is the sprint wobble: per-tick 0.41-0.44 on segments at 0.07-0.15 averaged."""
    cfg = zombie_rung2_yawfilt_cap3_lin3_env_cfg(play=play)
    cfg.rewards["yaw_wobble"] = RewardTermCfg(func=mdp.yaw_wobble_l2, weight=-1.0,
                                              params={"command_name": "athlete", "tau": 0.5})
    return cfg


def zombie_rung2_wobble_sharp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """z2_v13 = z2_v11 + track_lin std 0.5 -> 0.35. z2_v11 it400 (21/30 margin) misses are steady 0.16-0.20 m/s speed
    shortfalls on 2.4-2.85 m/s sprints — the 0.5 kernel pays ~90 % there (MATT's r2_v7 lesson). The heavier speed
    weights (3 + coarse 1.5) keep a gradient at large errors, which z2_v3's sharp kernel lacked."""
    cfg = zombie_rung2_wobble_env_cfg(play=play)
    cfg.rewards["track_lin"].params["std"] = 0.35
    return cfg


def zombie_rung2_wobble_sprint_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """z2_v14 = z2_v11 + 15 % sprint-band commands (vx 2.5-4.0 × √λ, |wz| <= 0.6 / √λ). z2_v11 it400's margin misses are
    2.4-2.85 m/s sprints 0.003-0.04 m/s over the speed bar. z2_v3's sprint focus (30 %) came with the sharp kernel and
    the old reward balance that let speed collapse."""
    cfg = zombie_rung2_wobble_env_cfg(play=play)
    cmd = cfg.commands["athlete"]
    cmd.sprint_fraction = 0.15
    cmd.sprint_vx = (2.5 * SS, 4.0 * SS)
    cmd.sprint_wz = 0.6 * WS
    return cfg


def zombie_rung2_bands_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """z2_v16 (2026-10-12) = the recipe that took GRANDMA (g2_v9) and KIM (k2_v3) to a stable 10/10, on the zombie:
    grandma_env's Cap -> Sharp -> Bands line (MATT's base recipe + lateral-acceleration cap, track_lin std 0.3 x SS,
    forward / backward / side-step command bands, symmetric runner, entropy 0.001), which binds its scaling to the
    process body. The zombie's own final line (Wobble = z2_v11, 8/10 at one checkpoint, 1-4/10 next to it) is the
    yaw-heavy one; z2_v12-v14 tried the sharp kernel, the low entropy and a sprint band only on top of that line.
    Without GRANDMA's careful-step term (as KIM). Warm start: the zombie's Rung2-Omni-Base run (plain_init, std <= 0.3).
    A walking brain of another body is no start for it: z2_v15 (KIM's k2_v3 on the Wobble task) never stood."""
    _check_body()
    from . import grandma_env as GE                # G = the zombie in a zombie process
    cfg = GE.grandma_rung2_bands_env_cfg(play=play)
    cfg.rewards.pop("careful_steps", None)
    return cfg


ZOMBIE_VX_TOP = 3.6              # MATT units (x SS = 2.83 m/s): G1 asks up to 2.76; z2_v11 ran 2.8 -> 2.74


def zombie_rung2_bands2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """z2_v19 (from the base run z2b_v1 it 1999: 4/10, it runs and turns, 0 falls). z2_v18 (Bands from the same start)
    went 2 -> 5/10 with the turns, the backward walk and most side-steps fixed, but the fast runs with a turn fell apart
    (2.23 m/s, wz -0.45: 0.25 -> 1.87 m/s RMS between the base and it 600): Bands commands up to 4.0 x SS = 3.14 m/s,
    past the zombie's 2.99 m/s envelope, under the sharp kernel — GRANDMA's "fast command = do not run" (g2_v4), which
    her 2.8 m/s cut fixed (g2_v5). Here the same cut for the zombie (forward top 3.6 x SS = 2.83 m/s in the range, the
    curriculum and the forward band) and KIM's band shares (k2_v3): forward 15 %, backward 10 %, fast side-steps 30 %
    (|vy| 0.7-1.1 x SS = 0.55-0.86 m/s; left over at 0.65-0.74 m/s)."""
    from . import grandma_env as GE
    cfg = zombie_rung2_bands_env_cfg(play=play)
    cmd = cfg.commands["athlete"]
    top = ZOMBIE_VX_TOP * SS
    cmd.ranges.lin_vel_x = (cmd.ranges.lin_vel_x[0], top)
    for stage in cfg.curriculum["command_vel"].params["velocity_stages"]:
        stage["lin_vel_x"] = (stage["lin_vel_x"][0], top)
    cmd.bands = ((0.15, (2.0 * SS, top), (0.0, 0.0), 0.6 * GE.WS),
                 (0.10, (-1.5 * SS, -0.8 * SS), (0.0, 0.0), 0.5 * GE.WS),
                 (0.30, (-0.5 * SS, 0.5 * SS), (0.7 * SS, 1.1 * SS), 0.0))
    return cfg


def zombie_rung2_capbands_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """z2_v21 (from the base run z2b_v1 it 1999, 4/10). The base run's REWARDS kept the zombie running and turning; the
    sharp speed kernel of Bands / Bands2 (std 0.3 x SS = 0.24 m/s) is where z2_v18 stopped running fast and z2_v19 ran
    5-8 % slow and stopped turning in a sprint. So: the base rewards (kernel 0.5), and from the Bands2 line only the
    command side — lateral-acceleration cap, forward top 2.83 m/s, the three bands — with the symmetric runner."""
    from . import grandma_env as GE
    _check_body()
    cfg = GE.grandma_rung2_cap_env_cfg(play=play)
    cfg.rewards.pop("careful_steps", None)
    cmd = cfg.commands["athlete"]
    top = ZOMBIE_VX_TOP * SS
    cmd.ranges.lin_vel_x = (cmd.ranges.lin_vel_x[0], top)
    for stage in cfg.curriculum["command_vel"].params["velocity_stages"]:
        stage["lin_vel_x"] = (stage["lin_vel_x"][0], top)
    cmd.bands = ((0.15, (2.0 * SS, top), (0.0, 0.0), 0.6 * GE.WS),
                 (0.10, (-1.5 * SS, -0.8 * SS), (0.0, 0.0), 0.5 * GE.WS),
                 (0.30, (-0.5 * SS, 0.5 * SS), (0.7 * SS, 1.1 * SS), 0.0))
    return cfg


def zombie_rung2_polish_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """z2_v20 = a careful polish of the DEPLOYED brain (z2_v11 it 400, rebuilt from zombie_rung2.onnx + its critic with
    tools/onnx_to_checkpoint.py: the September checkpoints are gone). On 2026-10-12 it is 8/10 with two near misses:
    a 2.76 m/s run with a -0.42 rad/s turn (0.21 vs the 0.157 m/s bar) and a slow diagonal side-step (0.2, -0.17;
    0.17). Its own recipe (Wobble) unchanged, 4096 envs, plus two command bands on exactly those parts: 20 % fast runs
    with a turn (2.0-3.6 x SS = 1.57-2.83 m/s, |wz| <= 0.6 / TS) and 20 % side-steps over the whole range (|vy|
    0.1-1.1 x SS with vx +-0.5 x SS). The trainer takes small steps (desired KL 0.004, entropy 0.001, action std 0.15
    at the start) and saves every 50 iterations: z2_v11's neighbours scored 1-4/10, so every checkpoint is gated."""
    cfg = zombie_rung2_wobble_env_cfg(play=play)
    cfg.scene.num_envs = 4096
    cfg.commands["athlete"].bands = ((0.20, (2.0 * SS, ZOMBIE_VX_TOP * SS), (0.0, 0.0), 0.6 * WS),
                                     (0.20, (-0.5 * SS, 0.5 * SS), (0.1 * SS, 1.1 * SS), 0.0))
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


# ---- 2026-10-11 audit: opt-in variants ---------------------------------------------------------------------------------
def zombie_rung0_full_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rung 0 + DESIGN §5's foot-slip and CoM-over-support terms. Warm start: z0_v1 it 1499 (plain_init)."""
    return M.add_rung0_full(zombie_rung0_env_cfg(play=play), LAM)


def zombie_rung2_ramp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """z2_v11's recipe (Wobble = zombie_rung2) + shove ramp, solref DR, privileged critic. Warm start z2_v11 it 400."""
    return M.add_ramp_recipe(zombie_rung2_wobble_env_cfg(play=play))
