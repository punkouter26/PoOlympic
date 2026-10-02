"""Rung S — contract v4 stance skills for MATT (events 2 Torso Archer, 3 Deep Squat, 4 Javelin Reach, 6 Flamingo,
7 Cadence March). docs/CONTRACT_V4_STANCE_PROPOSAL.md (approved 2026-09-29): ONE shared brain, warm-started from the
final Rung 2 recipe (r2_v8 = rung2.onnx) with the 11 new input columns zero-initialised (tools/expand_obs.py), so the
first iteration behaves exactly like Rung 2. 20 % of the command resamples stay plain locomotion (keeps the Rung 2
skills); the rest draw one stance skill each (skill_mdp.AthleteSkillCommand).

Terms that would fight a stance skill are made skill-aware: the pelvis-height reward follows the squat target, posture
pays only in locomotion, uprightness is off while aiming the torso / squatting, the pelvis fall line drops with the
squat, and the phase-contact reward knows the march clock and the flamingo stance.
"""

from __future__ import annotations

import dataclasses
import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from . import skill_mdp as S
from .matt_env import add_bio_rewards, fast_sim, matt_rung2_sym5_env_cfg

SKILL_REWARD_WEIGHT = 3.0


def matt_stance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = matt_rung2_sym5_env_cfg(play=play)

    # command: the Rung 2 velocity command + the skill block (same fields, skill-aware class)
    old = cfg.commands["athlete"]
    cfg.commands["athlete"] = S.AthleteSkillCommandCfg(**{f.name: getattr(old, f.name) for f in dataclasses.fields(old)
                                                          if f.init})

    # observation: the 11-value skill block after the 84 v3 terms (contract v4 order)
    for group in ("actor", "critic"):
        cfg.observations[group].terms["skill"] = ObservationTermCfg(func=S.obs_skill, params={})

    r = cfg.rewards
    # skill-aware versions of the Rung 0/1/2 terms
    r["height"] = RewardTermCfg(func=S.skill_pelvis_height, weight=r["height"].weight, params={"std": 0.15})
    r["posture"] = RewardTermCfg(func=S.posture_locomotion, weight=r["posture"].weight, params=dict(r["posture"].params))
    r["upright"] = RewardTermCfg(func=S.upright_unless_aiming, weight=r["upright"].weight, params=dict(r["upright"].params))
    r["phase_contact"] = RewardTermCfg(func=S.phase_contact_v4, weight=r["phase_contact"].weight)
    # the five skills
    w = SKILL_REWARD_WEIGHT
    r["skill_squat"] = RewardTermCfg(func=S.skill_squat, weight=w, params={"std": 0.05})
    r["skill_flamingo"] = RewardTermCfg(func=S.skill_flamingo, weight=w, params={"clearance": 0.10, "slip_speed": 0.05})
    r["skill_march"] = RewardTermCfg(func=S.skill_march, weight=w, params={"std": 0.06})
    # coarse terms (rs_v3), same fix as reach: rs_v2 squatted ~0.29 m short of the target (std 0.05 pays e^-33) and
    # barely lifted a knee for the march (std 0.06 vs 0.15-0.30 m lifts) -> both flat for 300 iterations
    r["skill_squat_coarse"] = RewardTermCfg(func=S.skill_squat, weight=w / 2, params={"std": 0.25})
    # rs_v4: the march coarse kernel paid ~73 % for standing still -> replaced by the swing-knee lift fraction
    r["skill_march_lift"] = RewardTermCfg(func=S.skill_march_lift, weight=w / 2)
    # rs_v5: joint-space leg guides for squat / flamingo / march (rs_v4 it950: no knee lift at all, squats 0.25 m short —
    # the task-space terms above pay ~0 until the pose is already close). Progress 0 at the default pose -> 1 at target.
    r["skill_leg_progress"] = RewardTermCfg(func=S.skill_leg_progress, weight=w / 2)
    r["skill_leg_pose"] = RewardTermCfg(func=S.skill_leg_pose, weight=w / 2, params={"std": 0.25})
    r["skill_torso"] = RewardTermCfg(func=S.skill_torso, weight=w, params={"std": math.radians(9)})
    # coarse term (rs_v2): the resting hand starts 0.5-1.2 m from the target, where std 0.10 pays ~0 (rs_v1 reach stayed 0)
    r["skill_reach_coarse"] = RewardTermCfg(func=S.skill_reach, weight=w / 2, params={"std": 0.50})
    r["skill_reach"] = RewardTermCfg(func=S.skill_reach, weight=w / 2, params={"std": 0.10})
    r["skill_reach_fine"] = RewardTermCfg(func=S.skill_reach, weight=w / 2, params={"std": 0.03})

    cfg.terminations["pelvis_low"] = TerminationTermCfg(func=S.pelvis_below_skill, params={"minimum_height": 0.55})

    # recipe v5 (staged 2026-09-29, before the first Rung S run): the joints a skill does not use hold the default pose
    # (posture is off outside locomotion, so idle arms / legs would drift), bio-realism terms, fast simulation (8192 envs,
    # one pool cube, sized buffers — tasks/matt_env.fast_sim).
    r["posture_idle"] = RewardTermCfg(func=S.posture_skill_idle, weight=r["posture"].weight, params={"std": 0.5})
    add_bio_rewards(cfg)
    fast_sim(cfg)
    return cfg


def matt_stance_v6_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """rs_v6 (from rs_v5c it2300). rs_v5c: squat 10/10 from it1800, torso 5-8/10, march 1/10, flamingo / reach 0/10,
    Rung 2 G1 eroding 9 -> 5-8/10 with only 20 % locomotion envs. Changes: locomotion share 0.20 -> 0.35 (skills 0.13
    each), march leg progress on the swing leg only, flamingo stance-slip penalty. Reach unchanged (needs a new idea)."""
    cfg = matt_stance_env_cfg(play=play)
    cfg.commands["athlete"].mode_probs = (0.35, 0.13, 0.13, 0.13, 0.13, 0.13)
    r = cfg.rewards
    r["skill_leg_progress"] = RewardTermCfg(func=S.skill_leg_progress_v6, weight=r["skill_leg_progress"].weight)
    r["skill_flamingo_slip"] = RewardTermCfg(func=S.skill_flamingo_slip, weight=-2.0)
    return cfg


def matt_stance_v7_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """rs_v7 (from rs_v6 it3299: squat 10/10, march lifts 83-97 % but drifts, flamingo one trial in four, torso 5/10,
    Rung 2 6/10). Adds `skill_spot` (stay on the spot in every stance mode, std 0.15 m, w/2) and `skill_flamingo_lift`
    (lifted-foot height with or without contact, w/2). Mode shares as rs_v6 (35 % locomotion)."""
    cfg = matt_stance_v6_env_cfg(play=play)
    r = cfg.rewards
    w = SKILL_REWARD_WEIGHT
    r["skill_spot"] = RewardTermCfg(func=S.skill_spot, weight=w / 2, params={"std": 0.15})
    r["skill_flamingo_lift"] = RewardTermCfg(func=S.skill_flamingo_lift, weight=w / 2, params={"clearance": 0.10})
    return cfg


def matt_stance_reach_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """rs_reach_v1 — an Event 4 brain (per-event stance brains, user 2026-10-01), from rs_v8 it 1199. The reach drill
    was 0/10 in every run, but the skill is mostly there: on 24 single targets (CPU, 2026-10-02) the hand ends a median
    0.095 m from the target, 8 of 24 inside the 0.08 m bar; the drill needs 6 of 6. The error grows with the target's
    distance (r = 0.65): targets past the straight arm's 0.61 m reach fall 0.10-0.16 m short and high ones 0.09-0.17 m
    low. FK: 92 % of the targets are reachable by the arm alone, all of them with a trunk lean of a few degrees.
    Changes: half of the commands are reaches (13 % before), a kernel between the 0.10 and 0.03 m ones (std 0.06), and
    the uprightness reward is off while reaching (a 12 deg lean cost 0.28 per step)."""
    cfg = matt_stance_v7_env_cfg(play=play)
    cfg.commands["athlete"].mode_probs = (0.20, 0.06, 0.06, 0.06, 0.12, 0.50)
    r = cfg.rewards
    r["skill_reach_mid"] = RewardTermCfg(func=S.skill_reach, weight=SKILL_REWARD_WEIGHT / 2, params={"std": 0.06})
    r["upright"].params["also"] = ("reach",)
    return cfg


def matt_stance_reach2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """rs_reach_v2 (from rs_reach_v1 it 1199). v1 brought single targets to 22 of 24 inside the bar (median 0.031 m at it
    900) but the drill stayed at 1/10: 88 % of its 6-target chains hold a target past the straight arm's 0.61 m, and
    those end 0.10-0.30 m off (hand low and short towards the arm's side, trunk tilt 1-5 deg: it does not lean).
    Uniform sampling puts 30 % of reaches there; here half of them are drawn from 80-95 % of the reach radius."""
    cfg = matt_stance_reach_env_cfg(play=play)
    cfg.commands["athlete"].reach_far_fraction = 0.5
    return cfg
