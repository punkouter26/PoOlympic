"""All-fours crawl (event 8 → "30m All Fours", user request 2026-09-29: race on all fours, survive falling down; both
bodies; only natural falls). Built on the get-up task (poolympic/tasks/matt_env.py, getup_v3): no fall terminations
(only the time-out), episodes start lying (mostly prone) with the fading torso assist, so recovering onto all fours
after a tumble is part of every episode.

Crawl = torso near horizontal, pelvis low, hands carrying weight, forward speed tracked on the contract command
(vx + lane-keeping yaw rate), standing up penalised. The get-up rewards and the gait-clock stance terms (bipedal
footfalls) are removed; the phase clock stays in the observation as a rhythm the policy may use.

Zombie: the same recipe with the zombie body and Froude scaling (heights × λ, speeds × √λ, yaw rates ÷ √λ,
torque penalty ÷ torque_scale²), without the bipedal style rewards (hunch / arms forward / wide stance / shuffle).
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp as vel_mdp

from .. import bodies
from . import matt_env as M
from . import mdp

# MATT crawl geometry (m / deg): on hands and knees his pelvis sits ~0.5 m up; bear-crawl (knees off) ~0.7 m
CRAWL_PELVIS_Z, CRAWL_PELVIS_STD = 0.5, 0.15
CRAWL_TILT_DEG, CRAWL_TILT_STD = 80.0, 25.0
CRAWL_VX = (0.3, 1.5)          # m/s, commanded crawl speeds (a human bear crawl ~1-1.5 m/s)
CRAWL_WZ = 0.5                 # rad/s, lane-keeping range (contract steering limit)
STAND_Z = 0.8                  # pelvis above this with the torso within 40° of vertical = standing up (penalised)


def _crawl(cfg: ManagerBasedRlEnvCfg, lam: float = 1.0, ss: float = 1.0, ws: float = 1.0) -> ManagerBasedRlEnvCfg:
    """Turn a get-up config (no terminations, lying resets, assist) into the crawl task. lam / ss / ws = the body's
    length, speed and angular-rate scales (1 for MATT)."""
    cfg.episode_length_s = 20.0
    cfg.scene.sensors = cfg.scene.sensors + (
        ContactSensorCfg(name=mdp.HAND_SENSOR,
                         primary=ContactMatch(mode="subtree", pattern=r"^(forearm_l|forearm_r)$", entity="robot"),
                         secondary=ContactMatch(mode="body", pattern="terrain"),
                         fields=("found",), reduce="netforce", num_slots=1),
    )
    cmd = cfg.commands["athlete"]
    cmd.resampling_time_range = (3.0, 8.0)
    cmd.rel_standing_envs = 0.1
    cmd.rel_heading_envs = 0.0
    cmd.heading_command = False
    cmd.ranges = mdp.AthleteCommandCfg.Ranges(lin_vel_x=(CRAWL_VX[0] * ss, CRAWL_VX[1] * ss), lin_vel_y=(0.0, 0.0),
                                              ang_vel_z=(-CRAWL_WZ * ws, CRAWL_WZ * ws))
    cfg.events["reset_base"].params["prone_fraction"] = 0.7
    rw = cfg.rewards
    for k in ("height", "upright", "height_progress", "upright_linear", "standing_tall", "rise"):
        rw.pop(k, None)
    rw.update({
        "track_lin": RewardTermCfg(func=vel_mdp.track_linear_velocity, weight=3.0,
                                   params={"command_name": "athlete", "std": 0.3 * ss}),
        "track_lin_coarse": RewardTermCfg(func=vel_mdp.track_linear_velocity, weight=1.0,
                                          params={"command_name": "athlete", "std": 1.0 * ss}),
        "track_yaw": RewardTermCfg(func=mdp.track_yaw_rate, weight=1.0,
                                   params={"command_name": "athlete", "std": 0.5 * ws}),
        "crawl_height": RewardTermCfg(func=mdp.base_height_tracking, weight=1.5,
                                      params={"target": CRAWL_PELVIS_Z * lam, "std": CRAWL_PELVIS_STD * lam}),
        "crawl_tilt": RewardTermCfg(func=mdp.torso_tilt_target, weight=1.5,
                                    params={"target_deg": CRAWL_TILT_DEG, "std_deg": CRAWL_TILT_STD}),
        "hands_down": RewardTermCfg(func=mdp.hands_on_ground, weight=0.5),
        "standing_up": RewardTermCfg(func=mdp.standing_tall, weight=-3.0,
                                     params={"min_height": STAND_Z * lam, "max_tilt_deg": 40.0}),
    })
    return cfg


def matt_crawl_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = _crawl(M.matt_getup3_env_cfg(play=play))
    if "getup_assist" in cfg.events:
        cfg.events["getup_assist"].params.update({"max_fraction": 0.4, "decay_steps": 600 * 24})
    return cfg


def zombie_crawl_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """The zombie crawl (POOLYMPIC_BODY=zombie): MATT's crawl with the zombie body, Froude-scaled."""
    from .zombie_env import FALL_Z, zombie_entity_cfg  # noqa: F401  (zombie process only)
    z = bodies.BODIES["zombie"]
    lam, ss, ws = z.length_scale, z.speed_scale, 1.0 / z.time_scale
    cfg = M.matt_getup3_env_cfg(play=play)
    cfg.scene.entities["robot"] = zombie_entity_cfg()
    cfg.events["reset_base"].params["z"] = (0.2 * lam, 0.25 * lam)
    if "getup_assist" in cfg.events:
        cfg.events["getup_assist"].params.update({"max_fraction": 0.4, "decay_steps": 600 * 24,
                                                  "body_weight_n": z.total_mass * 9.81})
    cfg = _crawl(cfg, lam, ss, ws)
    rw = cfg.rewards
    rw["joint_vel_limit"].params["limit"] = 18.0 * ws
    rw["torques"].weight = rw["torques"].weight / z.torque_scale ** 2
    cfg.sim.nconmax, cfg.sim.njmax = 160, 800          # full self-collision: more contacts
    cfg.viewer.distance = 3.5 * lam
    return cfg


def _crawl2(cfg: ManagerBasedRlEnvCfg, ss: float = 1.0) -> ManagerBasedRlEnvCfg:
    """crawl_v2: crawl_v1 held a perfect static all-fours pose (100 % on all fours, hands down, 0 m progress by it 400):
    posture terms paid ~3.5 per step, the exp speed kernels ~0 until already moving. Add a forward-progress reward
    (speed / command, w 3), halve the posture terms, and a speed curriculum 0.2-0.8 → 0.3-1.5 m/s (× √λ)."""
    from mjlab.managers.curriculum_manager import CurriculumTermCfg
    rw = cfg.rewards
    rw["progress"] = RewardTermCfg(func=mdp.forward_progress, weight=3.0, params={"command_name": "athlete"})
    rw["crawl_height"].weight = 0.75
    rw["crawl_tilt"].weight = 0.75
    rw["hands_down"].weight = 0.3
    cmd = cfg.commands["athlete"]
    cmd.ranges.lin_vel_x = (0.2 * ss, 0.8 * ss)
    cfg.curriculum = {"command_vel": CurriculumTermCfg(func=vel_mdp.commands_vel, params={
        "command_name": "athlete", "velocity_stages": [
            {"step": 0, "lin_vel_x": (0.2 * ss, 0.8 * ss)},
            {"step": 600 * 24, "lin_vel_x": (CRAWL_VX[0] * ss, CRAWL_VX[1] * ss)}]})}
    return cfg


def matt_crawl2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _crawl2(matt_crawl_env_cfg(play=play))


def zombie_crawl2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _crawl2(zombie_crawl_env_cfg(play=play), bodies.BODIES["zombie"].speed_scale)


def _crawl3(cfg: ManagerBasedRlEnvCfg, ss: float = 1.0, ws: float = 1.0) -> ManagerBasedRlEnvCfg:
    """crawl_v3: crawl_v2 crawled backwards (−5 m at a 0.6 m/s command). Every "forward" was measured along the pelvis
    x axis (mjlab's body-frame speed kernel, the contract yaw), which points at the ground on all fours and flips towards
    the feet once the hips are above the shoulders; the yaw terms measured rotation about the (horizontal) spine.
    Speed, progress and turn rate now use mdp.crawl_heading (x + z axis projection) and the world vertical."""
    rw = cfg.rewards
    rw["track_lin"] = RewardTermCfg(func=mdp.crawl_track_lin, weight=3.0, params={"std": 0.3 * ss})
    rw["track_lin_coarse"] = RewardTermCfg(func=mdp.crawl_track_lin, weight=1.0, params={"std": 1.0 * ss})
    rw["progress"] = RewardTermCfg(func=mdp.crawl_progress, weight=3.0)
    rw["track_yaw"] = RewardTermCfg(func=mdp.yaw_rate_world, weight=1.0, params={"std": 0.5 * ws})
    return cfg


def matt_crawl3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _crawl3(matt_crawl2_env_cfg(play=play))


def zombie_crawl3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    z = bodies.BODIES["zombie"]
    return _crawl3(zombie_crawl2_env_cfg(play=play), z.speed_scale, 1.0 / z.time_scale)


def zombie_crawl4_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """zcrawl_v2: zcrawl_v1 (MATT crawl warm start) belly-slid forward — progress 2.45 / 3, pelvis below the crawl band,
    hands ~0. Progress only pays up on all fours (× (z / crawl height)²), a linear climb-to-crawl-height term gives a
    gradient from the floor, hands 0.3 → 1.0. Warm start: the zombie's own Rung 0 brain (MATT's crawl actions are
    offsets from MATT's default pose; the zombie's default holds the arms out in front)."""
    z = bodies.BODIES["zombie"]
    cfg = zombie_crawl3_env_cfg(play=play)
    target = CRAWL_PELVIS_Z * z.length_scale
    cfg.rewards["progress"] = RewardTermCfg(func=mdp.crawl_progress_gated, weight=3.0, params={"target_z": target})
    cfg.rewards["crawl_up"] = RewardTermCfg(func=mdp.height_progress, weight=1.5, params={"target": target})
    cfg.rewards["hands_down"].weight = 1.0
    return cfg


def zombie_crawl5_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """zcrawl_v3 (from zcrawl_v2 it700): v2 crawls at race pace on all fours (30 m in 33-34.5 s at 0.94 m/s, hands 76 %)
    but ignores turn commands — lane keeping asks +0.5 rad/s, it turns ~−0.1; track_yaw flat at 0.48 (MATT 0.65) —
    and curves 11-14 m off its lane. Turn tracking ×3 plus a linear turn-error penalty."""
    cfg = zombie_crawl4_env_cfg(play=play)
    cfg.rewards["track_yaw"].weight = 3.0
    cfg.rewards["yaw_l1"] = RewardTermCfg(func=mdp.yaw_rate_world_l1, weight=-0.5)
    return cfg
