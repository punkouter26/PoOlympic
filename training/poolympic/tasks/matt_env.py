"""MATT athlete environments (DESIGN.md §2/§3/§5) on mjlab 1.6 + MuJoCo Warp.

Physics = training/assets/matt.xml (the same MJCF Unity imports) with its own <position> actuators (XmlActuator),
ground plane with the scene's exact contact bits/friction, and a 4-cube pool per env. Every choice that affects
the deployed policy's input/output is tied to poolympic/contract.py.
"""

from __future__ import annotations

import math

import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig

from .. import contract as C
from . import mdp

MATT_XML = C.ROOT / "assets" / "matt.xml"
CUBE_XML = C.ROOT / "assets" / "cube.xml"
N_CUBES = 4
DEFAULT_ROOT_Z = 0.9549291  # scene_matt.xml keyframe "default"
CUBE_NAMES = tuple(f"cube{i}" for i in range(N_CUBES))
FOOT_BODIES = ("foot_l", "toe_l", "foot_r", "toe_r")
ALL_BITS = 0xFFFF


def matt_entity_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(str(MATT_XML)),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, DEFAULT_ROOT_Z),
            joint_pos={n: v for n, v in zip(mdp.CONTRACT_ACTUATORS, mdp.CONTRACT_DEFAULTS)} | {"toe_.*": 0.0},
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            # Use MATT's own <position kp kv forcerange> actuators — identical to what Unity's MuJoCo steps.
            actuators=(XmlActuatorCfg(target_names_expr=(".*",), delay_min_lag=0, delay_max_lag=4,
                                      delay_update_period=4000),),
            soft_joint_pos_limit_factor=0.95,
        ),
    )


def cube_entity_cfg(i: int) -> EntityCfg:
    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(str(CUBE_XML)),
        init_state=EntityCfg.InitialStateCfg(pos=(50.0 + 2.0 * i, 0.0, 0.1)),
    )


def matt_rung0_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rung 0 — stand + shove/cube recovery (Iron Pedestal, Gust Gauntlet)."""
    scene = SceneCfg(
        num_envs=1,
        env_spacing=3.0,
        terrain=TerrainEntityCfg(
            terrain_type="plane",
            # the scene's ground exactly: contacts every lane's bits, friction (1, 0.005, 0.0001)
            collisions=(CollisionCfg(geom_names_expr=("terrain",), contype=ALL_BITS, conaffinity=ALL_BITS,
                                     condim=3, priority=0, friction=(1.0, 0.005, 0.0001)),),
        ),
        entities={"robot": matt_entity_cfg(), **{n: cube_entity_cfg(i) for i, n in enumerate(CUBE_NAMES)}},
        sensors=(
            ContactSensorCfg(
                name="nonfoot_ground",
                primary=ContactMatch(mode="body", pattern=r".*", entity="robot", exclude=FOOT_BODIES),
                secondary=ContactMatch(mode="body", pattern="terrain"),
                fields=("found",),
                reduce="none",
                num_slots=1,
            ),
        ),
    )

    def term(func, noise=None, **params):
        return ObservationTermCfg(func=func, params=params, noise=noise)

    # Contract order (poolympic/contract.py OBS_LAYOUT) — do not reorder.
    actor_terms = {
        "base_lin_vel_heading": term(mdp.obs_base_lin_vel_heading, Unoise(n_min=-0.1, n_max=0.1)),
        "base_ang_vel_local": term(mdp.obs_base_ang_vel_local, Unoise(n_min=-0.2, n_max=0.2)),
        "projected_gravity": term(mdp.obs_projected_gravity, Unoise(n_min=-0.05, n_max=0.05)),
        "base_height": term(mdp.obs_base_height, Unoise(n_min=-0.02, n_max=0.02)),
        "command": term(mdp.obs_command),
        "gait_phase_sincos": term(mdp.obs_gait_phase),
        "joint_pos_rel": term(mdp.obs_joint_pos_rel, Unoise(n_min=-0.01, n_max=0.01)),
        "joint_vel_scaled": term(mdp.obs_joint_vel_scaled, Unoise(n_min=-0.05, n_max=0.05)),
        "last_action": term(mdp.obs_last_action),
    }
    critic_terms = {k: ObservationTermCfg(func=v.func, params=v.params) for k, v in actor_terms.items()}

    observations = {
        "actor": ObservationGroupCfg(terms=actor_terms, concatenate_terms=True, enable_corruption=not play),
        "critic": ObservationGroupCfg(terms=critic_terms, concatenate_terms=True, enable_corruption=False),
    }

    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=tuple(f"^{n}$" for n in mdp.CONTRACT_ACTUATORS),
            preserve_order=True,  # contract actuator order == ONNX output order
            scale=C.ACTION_SCALE,
            use_default_offset=True,
            # identical to the ONNX graph: clip(default + 0.25·a, joint range)
            clip={f"^{n}$": (lo, hi) for n, lo, hi in
                  zip(mdp.CONTRACT_ACTUATORS, mdp.CONTRACT_RANGE_LO, mdp.CONTRACT_RANGE_HI)},
        )
    }

    commands = {
        "athlete": mdp.AthleteCommandCfg(
            entity_name="robot",
            resampling_time_range=(1e9, 1e9),
            rel_standing_envs=1.0,
            rel_heading_envs=0.0,
            heading_command=False,
            ranges=mdp.AthleteCommandCfg.Ranges(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
        )
    }

    events = {
        "reset_scene_to_default": EventTermCfg(func=envs_mdp.reset_scene_to_default, mode="reset"),
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform, mode="reset",
            params={"pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-math.pi, math.pi)},
                    "velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}},
        ),
        "reset_joints": EventTermCfg(
            func=envs_mdp.reset_joints_by_offset, mode="reset",
            params={"position_range": (-0.05, 0.05), "velocity_range": (-0.1, 0.1),
                    "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        # robustness curriculum (DESIGN §1): shoves + dropped cubes + DR
        "push_robot": EventTermCfg(
            func=envs_mdp.push_by_setting_velocity, mode="interval", interval_range_s=(3.0, 5.0),
            params={"velocity_range": {"x": (-0.6, 0.6), "y": (-0.6, 0.6)}},
        ),
        "drop_cube": EventTermCfg(
            func=mdp.drop_cube_on_athlete, mode="interval", interval_range_s=(3.0, 5.0),
            params={"cube_names": CUBE_NAMES, "height_above_shoulder": 1.5},
        ),
        "dr_mass": EventTermCfg(
            func=dr.body_mass, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=(".*",)), "ranges": (0.85, 1.15),
                    "operation": "scale"},
        ),
        "dr_foot_friction": EventTermCfg(
            func=dr.geom_friction, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", geom_names=(r"(foot|toe)_[lr]_geom0",)), "ranges": (0.8, 1.2),
                    "operation": "scale", "shared_random": True},
        ),
        "dr_pd_gains": EventTermCfg(
            func=dr.pd_gains, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)), "kp_range": (0.85, 1.15),
                    "kd_range": (0.85, 1.15), "operation": "scale"},
        ),
        "dr_strength": EventTermCfg(  # per-lane "athlete trait": strength scale
            func=dr.effort_limits, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)), "effort_limit_range": (0.85, 1.15),
                    "operation": "scale"},
        ),
    }

    torso = SceneEntityCfg("robot", body_names=("torso",))
    rewards = {
        "upright": RewardTermCfg(func=mdp.torso_upright, weight=1.0, params={"std": math.radians(20)}),
        "height": RewardTermCfg(func=mdp.base_height_tracking, weight=1.0,
                                params={"target": DEFAULT_ROOT_Z, "std": 0.1}),
        "still_lin": RewardTermCfg(func=vel_mdp.track_linear_velocity, weight=1.0,
                                   params={"command_name": "athlete", "std": 0.5}),
        "still_ang": RewardTermCfg(func=vel_mdp.track_angular_velocity, weight=0.5,
                                   params={"command_name": "athlete", "std": math.sqrt(0.5)}),
        "near_origin": RewardTermCfg(func=mdp.stay_near_origin, weight=0.5, params={"std": 0.3}),
        "posture": RewardTermCfg(func=envs_mdp.posture, weight=0.5,
                                 params={"std": {".*": 0.35}, "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
        "torso_ang_vel": RewardTermCfg(func=vel_mdp.body_angular_velocity_penalty, weight=-0.05,
                                       params={"asset_cfg": torso}),
        "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.05),
        "joint_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-1.0),
        "torques": RewardTermCfg(func=envs_mdp.joint_torques_l2, weight=-1e-5),
        "joint_vel_limit": RewardTermCfg(func=mdp.joint_vel_limit_excess, weight=-0.5, params={"limit": 18.0}),
        "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-200.0),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "pelvis_low": TerminationTermCfg(func=mdp.pelvis_below, params={"minimum_height": 0.55}),
        "torso_tilt": TerminationTermCfg(func=mdp.torso_tilt_exceeds, params={"limit_deg": 60.0}),
        "nonfoot_contact": TerminationTermCfg(func=mdp.nonfoot_ground_contact, params={"sensor_name": "nonfoot_ground"}),
    }

    cfg = ManagerBasedRlEnvCfg(
        decimation=C.DECIMATION,
        episode_length_s=20.0,
        scene=scene,
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="pelvis",
                            distance=3.5, elevation=-5.0, azimuth=90.0),
        sim=SimulationCfg(
            nconmax=96,
            njmax=500,
            mujoco=MujocoCfg(timestep=0.005, integrator="implicitfast", cone="pyramidal", jacobian="auto",
                             solver="newton", iterations=20, tolerance=1e-8, ls_iterations=50, ls_tolerance=0.01,
                             impratio=1.0, gravity=(0.0, 0.0, -9.81), disableflags=("multiccd",)),
        ),
    )
    if play:
        cfg.episode_length_s = int(1e9)
        for k in ("dr_mass", "dr_foot_friction", "dr_pd_gains", "dr_strength"):
            cfg.events.pop(k, None)
    return cfg


def matt_ppo_cfg(experiment: str, max_iterations: int) -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
                            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.8, "std_type": "scalar"}),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
                                       entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
                                       learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95,
                                       desired_kl=0.01, max_grad_norm=1.0),
        experiment_name=experiment,
        logger="tensorboard",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=max_iterations,
    )


def matt_rung1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rung 1 — forward walk/run at a commanded speed with heading hold (30m Dash, Terminal Velocity)."""
    cfg = matt_rung0_env_cfg(play=play)
    cfg.scene.sensors = cfg.scene.sensors + (
        ContactSensorCfg(
            name=mdp.FOOT_SENSOR,
            primary=ContactMatch(mode="subtree", pattern=r"^(foot_l|foot_r)$", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
        ),
    )
    cmd = cfg.commands["athlete"]
    cmd.resampling_time_range = (5.0, 10.0)
    cmd.rel_standing_envs = 0.1
    cmd.heading_command = True
    cmd.rel_heading_envs = 1.0
    cmd.heading_control_stiffness = 0.5
    cmd.ranges = mdp.AthleteCommandCfg.Ranges(lin_vel_x=(0.3, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.5, 0.5),
                                              heading=(-math.pi, math.pi))

    cfg.events["push_robot"].params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
    cfg.events["drop_cube"].interval_range_s = (5.0, 8.0)

    for k in ("near_origin", "still_lin", "still_ang", "posture", "height"):
        cfg.rewards.pop(k)
    cfg.rewards.update({
        "track_lin": RewardTermCfg(func=vel_mdp.track_linear_velocity, weight=2.0,
                                   params={"command_name": "athlete", "std": 0.5}),
        # wide kernel: a policy standing still at a 1-2 m/s command still sees a gradient towards moving
        # (std 0.5 alone gives exp(-6) there — r1_v1 collapsed into standing)
        "track_lin_coarse": RewardTermCfg(func=vel_mdp.track_linear_velocity, weight=1.0,
                                          params={"command_name": "athlete", "std": 1.0}),
        "track_ang": RewardTermCfg(func=vel_mdp.track_angular_velocity, weight=1.0,
                                   params={"command_name": "athlete", "std": math.sqrt(0.5)}),
        "height": RewardTermCfg(func=mdp.base_height_tracking, weight=0.5,
                                params={"target": DEFAULT_ROOT_Z, "std": 0.15}),
        "posture": RewardTermCfg(func=vel_mdp.variable_posture, weight=0.5, params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)), "command_name": "athlete",
            "std_standing": {".*": 0.2}, "std_walking": {".*": 0.5}, "std_running": {".*": 0.8},
            "walking_threshold": 0.1, "running_threshold": 1.8}),
        "phase_contact": RewardTermCfg(func=mdp.phase_contact, weight=0.5),
        "air_time": RewardTermCfg(func=vel_mdp.feet_air_time, weight=0.5, params={
            "sensor_name": mdp.FOOT_SENSOR, "threshold_min": 0.1, "threshold_max": 0.6,
            "command_name": "athlete", "command_threshold": 0.3}),
        "foot_slip": RewardTermCfg(func=mdp.foot_slip, weight=-0.1),
    })
    cfg.curriculum = {
        "command_vel": CurriculumTermCfg(func=vel_mdp.commands_vel, params={
            "command_name": "athlete",
            "velocity_stages": [
                {"step": 0, "lin_vel_x": (0.3, 1.0)},
                {"step": 600 * 24, "lin_vel_x": (0.5, 2.0)},
                {"step": 1200 * 24, "lin_vel_x": (0.5, 3.0)},
                {"step": 2000 * 24, "lin_vel_x": (0.5, 4.0)},
            ]}),
    }
    if play:
        cmd.ranges.lin_vel_x = (0.5, 4.0)
    return cfg


# Rung 2 command envelope (DESIGN §1/§5). wz reaches 2.5 so the 360° Turntable bar (< 3 s ⇒ wz ≥ 2.1) is in-distribution.
RUNG2_STAGES = [
    {"step": 0, "lin_vel_x": (-1.0, 3.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
    {"step": 500 * 24, "lin_vel_x": (-1.5, 3.5), "lin_vel_y": (-1.0, 1.0), "ang_vel_z": (-2.0, 2.0)},
    {"step": 1000 * 24, "lin_vel_x": (-1.5, 4.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_z": (-2.5, 2.5)},
]


def matt_rung2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rung 2 — omnidirectional + yaw (Inverted Sprint, Crab Shuffle, Slalom, 360 Turntable, Emergency Brake).
    Warm-started from the Rung 1 brain (tools/warm_start.py re-seeds the command normalizer)."""
    cfg = matt_rung1_env_cfg(play=play)
    cmd = cfg.commands["athlete"]
    cmd.heading_command = False  # direct yaw-rate commands; races steer with contract.steer_yaw_rate
    cmd.rel_heading_envs = 0.0
    cmd.rel_standing_envs = 0.15  # zero-command stops (Emergency Brake) at every resample
    cmd.resampling_time_range = (3.0, 8.0)  # more transitions: brakes, reversals, turn-in / turn-out
    s0 = RUNG2_STAGES[0]
    cmd.ranges = mdp.AthleteCommandCfg.Ranges(lin_vel_x=s0["lin_vel_x"], lin_vel_y=s0["lin_vel_y"],
                                              ang_vel_z=s0["ang_vel_z"], heading=None)
    # yaw tracking is a primary objective now: tighter kernel (Rung 1: std √0.5 was nearly flat for small yaw-rate
    # errors → heading bias) + a wide one for the fast turntable.
    # r2_v2 (r2_v1 yaw decayed after wz ±2 entered): fine w 1.5 → 2.0; coarse w 0.5 → 1.0, std 1.5 → 1.0 so a 1 rad/s
    # miss still pays and improving it pays more; posture 0.5 → 0.25 so hips / trunk may rotate for fast pivots.
    cfg.rewards["track_ang"] = RewardTermCfg(func=vel_mdp.track_angular_velocity, weight=2.0,
                                             params={"command_name": "athlete", "std": 0.5})
    cfg.rewards["track_ang_coarse"] = RewardTermCfg(func=vel_mdp.track_angular_velocity, weight=1.0,
                                                    params={"command_name": "athlete", "std": 1.0})
    cfg.rewards["posture"].weight = 0.25
    cfg.curriculum = {
        "command_vel": CurriculumTermCfg(func=vel_mdp.commands_vel, params={
            "command_name": "athlete", "velocity_stages": RUNG2_STAGES}),
    }
    if play:
        last = RUNG2_STAGES[-1]
        cmd.ranges.lin_vel_x, cmd.ranges.lin_vel_y, cmd.ranges.ang_vel_z = last["lin_vel_x"], last["lin_vel_y"], last["ang_vel_z"]
    return cfg


def matt_pedestal_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Iron Pedestal fine-tune (Event 1 heat): Rung 0 + stepping off the 1 m x 1 m pedestal ends the episode, feet-centred
    reward, harder and more frequent gusts (the heat escalates 0.3 → ~1 m/s every 3 s). Warm start: r0_v2 it 1000."""
    cfg = matt_rung0_env_cfg(play=play)
    cfg.terminations["off_pedestal"] = TerminationTermCfg(func=mdp.feet_off_pedestal, params={"half": 0.5})
    cfg.rewards["feet_centred"] = RewardTermCfg(func=mdp.feet_centred, weight=1.0, params={"std": 0.3})
    cfg.events["push_robot"].params["velocity_range"] = {"x": (-0.8, 0.8), "y": (-0.8, 0.8)}
    cfg.events["push_robot"].interval_range_s = (2.0, 4.0)
    return cfg


def matt_rung2_sym_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rung 2 fine-tune with left/right symmetry augmentation (r2_v3): r2_v2 rewards, full command envelope from the
    start (the curriculum already ran in r2_v2). Warm start: r2_v2 it 1500. Runner: symmetry.SymmetricRunner."""
    cfg = matt_rung2_env_cfg(play=play)
    last = RUNG2_STAGES[-1]
    cfg.curriculum = {
        "command_vel": CurriculumTermCfg(func=vel_mdp.commands_vel, params={
            "command_name": "athlete", "velocity_stages": [dict(last, step=0)]}),
    }
    cmd = cfg.commands["athlete"]
    cmd.ranges.lin_vel_x, cmd.ranges.lin_vel_y, cmd.ranges.ang_vel_z = last["lin_vel_x"], last["lin_vel_y"], last["ang_vel_z"]
    return cfg


def matt_pedestal2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Iron Pedestal v2 (ped_v1 failed: fixed ±0.8 m/s gusts gave no learning signal). Heat-shaped gusts (fixed
    magnitude, random direction, every 3 s like the heat) whose magnitude adapts to success: 0.3 → up to 1.2 m/s."""
    cfg = matt_pedestal_env_cfg(play=play)
    cfg.events["push_robot"] = EventTermCfg(func=mdp.push_gust, mode="interval", interval_range_s=(2.5, 3.5))
    cfg.curriculum = {
        "gust": CurriculumTermCfg(func=mdp.gust_curriculum, params={
            "start": 0.3, "step": 0.02, "max_level": 1.2, "up": 0.6, "down": 0.3}),
    }
    return cfg


RUNG2_TRAIN_ENVELOPE = {"lin_vel_x": (-1.5, 4.0), "lin_vel_y": (-1.2, 1.2), "ang_vel_z": (-3.0, 3.0)}


def matt_rung2_sym2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """r2_v5: r2_v4 (symmetric, contract v3) with the training envelope ~20 % wider than the G1 envelope in the hard
    directions (wz ±3.0 so the 2.5 rad/s turntable is not at the edge; vy ±1.2 for ~1 m/s crabs) and more zero-command
    stops (0.20) — the r2_v4 misses were exactly turntable-at-edge, crab ~0.9 m/s and holding still after shoves."""
    cfg = matt_rung2_sym_env_cfg(play=play)
    env = RUNG2_TRAIN_ENVELOPE
    cfg.curriculum = {
        "command_vel": CurriculumTermCfg(func=vel_mdp.commands_vel, params={
            "command_name": "athlete", "velocity_stages": [dict(env, step=0)]}),
    }
    cmd = cfg.commands["athlete"]
    cmd.ranges.lin_vel_x, cmd.ranges.lin_vel_y, cmd.ranges.ang_vel_z = env["lin_vel_x"], env["lin_vel_y"], env["ang_vel_z"]
    cmd.rel_standing_envs = 0.20
    return cfg


def matt_rung2_sym3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """r2_v6 candidate: r2_v5 + feasibility-aware commands (|wz| <= 4 m/s^2 / |v|). Every G1 command stays inside
    (turns 1.5 m/s x 2 rad/s = 3 m/s^2, sprints 4 m/s x 0.5 = 2 m/s^2); 4 m/s x 3 rad/s style combos are clipped."""
    cfg = matt_rung2_sym2_env_cfg(play=play)
    cfg.commands["athlete"].max_lateral_accel = 4.0
    return cfg
