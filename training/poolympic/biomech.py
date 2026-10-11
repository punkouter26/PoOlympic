"""Biomechanics in the PHYSICS (opt-in; docs/CONTRACT_V5_FATIGUE_PROPOSAL.md): a torque-speed limit and a fatigue
model that scale each actuator's force range once per control tick. Existing brains / tasks do not use this module's
limits (their physics keeps the flat caps); only tasks that add mdp.bio_force_limits do.

One set of formulas for training (torch, tasks/mdp.py), the CPU gates (numpy, BioLimits below) and Unity: every function
here uses only operators and methods that numpy arrays and torch tensors share.

  torque-speed (Hill)  the torque available in the direction the joint is MOVING falls linearly from the cap at rest to
                       0 at w_max (the same curve the reward mdp.torque_speed_excess assumes); against the motion
                       (eccentric, braking) the full cap stays
  fatigue              per joint group g (both sides share one level, 0 fresh .. 1 exhausted):
                         effort_g = mean over the group's actuators of |torque| / nominal cap          (0 .. 1)
                         x_g      = effort_g / FATIGUE_EFFORT_REF
                         level_g += dt * (x_g^2 / FATIGUE_TAU_LOAD - level_g / FATIGUE_TAU_REST),       clipped to [0, 1]
                       cap scale = 1 - (1 - FATIGUE_FLOOR) * level_g   (never below FATIGUE_FLOOR)
  force range          lo = base_lo * fatigue * (hill if qd < 0 else 1),  hi = base_hi * fatigue * (hill if qd > 0 else 1)
"""

from __future__ import annotations

import numpy as np

# actuator name prefix -> joint group (contract actuator names: abdomen_*, shoulder_*_l/r, elbow_l/r, hip_*, knee_*, ankle_*)
JOINT_GROUPS = ("abdomen", "shoulder", "elbow", "hip", "knee", "ankle")
# Hill-type torque-velocity envelope: joint-group shortening speed at which the available torque reaches 0 (rad/s;
# sprint data: knee ~20+, hip ~12-15, ankle ~15-20). MATT units — as in the reward, not Froude-scaled per body.
HILL_W_MAX = {"abdomen": 8.0, "shoulder": 20.0, "elbow": 20.0, "hip": 15.0, "knee": 22.0, "ankle": 18.0}
FATIGUE_EFFORT_REF = 0.25    # effort (share of the cap) that tires a fresh group in FATIGUE_TAU_LOAD seconds
FATIGUE_TAU_LOAD = 10.0      # s at the reference effort from fresh to exhausted (no recovery term)
FATIGUE_TAU_REST = 30.0      # s time constant of the recovery at rest
#   -> a sustained effort e settles at level min(1, 3 (e / 0.25)^2), 63 % of the way after 30 s. Measured on KIM's
#      Rung 2 brain (G1 drill, action log, 2026-10-11): mean effort per group 0.03-0.10 overall (level 0.05-0.5),
#      0.10-0.13 for hip / knee / ankle while sprinting (level 0.5-0.8 if held). The caps are peak isometric torques, so
#      the reference sits at a quarter of them; with effort / cap alone (REF = 1) no group passed level 0.04 in 55 s.
#      Game time scale (heats last 10-45 s), not physiology's minutes. All four constants are proposals to tune.
FATIGUE_FLOOR = 0.6          # force cap scale of an exhausted group


def group_of(name: str) -> int:
    return next(i for i, g in enumerate(JOINT_GROUPS) if name.startswith(g))


def tables(actuator_names) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(group index per actuator [A], group mean matrix [A, G] (effort_g = (|tau| / cap) @ M), w_max per actuator [A])."""
    gi = np.array([group_of(n) for n in actuator_names])
    mean = np.zeros((len(gi), len(JOINT_GROUPS)))
    mean[np.arange(len(gi)), gi] = 1.0
    mean /= np.maximum(mean.sum(axis=0, keepdims=True), 1.0)      # a group without actuators stays at effort 0
    return gi, mean, np.array([HILL_W_MAX[JOINT_GROUPS[g]] for g in gi])


def fatigue_step(level, effort, dt: float):
    """One control tick of the leaky effort integral. level: [..., G] in 0 .. 1; effort: |torque| / cap, [..., G]."""
    x = effort / FATIGUE_EFFORT_REF
    return (level + dt * (x * x / FATIGUE_TAU_LOAD - level / FATIGUE_TAU_REST)).clip(0.0, 1.0)


def force_scale(level_act, qd, w_max, hill: bool = True, fatigue: bool = True):
    """(scale of the force range's lower bound, scale of its upper bound) per actuator. level_act: the group level of
    each actuator [..., A]; qd: joint speed [..., A]."""
    f = 1.0 - (1.0 - FATIGUE_FLOOR) * level_act if fatigue else level_act * 0.0 + 1.0
    if not hill:
        return f, f
    h = (1.0 - abs(qd) / w_max).clip(0.0, 1.0)
    pos, neg = (qd > 0) * 1.0, (qd < 0) * 1.0
    return f * (h * neg + (1.0 - neg)), f * (h * pos + (1.0 - pos))


class BioLimits:
    """CPU reference (G1 gates, events): call tick(m, d) once per control tick BEFORE building the observation and
    stepping — training applies the same update at the end of the previous env step (mjlab "step" event), so both use
    the state at the tick start and the actuator force of the previous tick's last substep. The first tick after a
    reset changes nothing (level 0, nominal force range), as the training reset does."""

    def __init__(self, m, ath, hill: bool = True, fatigue: bool = True):
        self.ath, self.hill, self.fatigue = ath, hill, fatigue
        self.base = m.actuator_forcerange[ath.actuator_ids].copy()
        self.cap = np.abs(self.base).max(axis=1)
        self.gi, self.mean, self.w_max = tables(ath.actuator_names)
        self.dt = float(m.opt.timestep) * 4              # contract.DECIMATION
        self.reset(m)

    def reset(self, m) -> None:
        self.level = np.zeros(len(JOINT_GROUPS))
        self.fresh = True
        m.actuator_forcerange[self.ath.actuator_ids] = self.base

    def tick(self, m, d) -> np.ndarray:
        """Update the fatigue levels and write the force ranges; returns the levels (the v5 observation block)."""
        if self.fresh:
            self.fresh = False
            return self.level
        effort = (np.abs(d.actuator_force[self.ath.actuator_ids]) / self.cap) @ self.mean
        self.level = fatigue_step(self.level, effort, self.dt)
        lo, hi = force_scale(self.level[self.gi], d.qvel[self.ath.joint_dofadr], self.w_max, self.hill, self.fatigue)
        m.actuator_forcerange[self.ath.actuator_ids, 0] = self.base[:, 0] * lo
        m.actuator_forcerange[self.ath.actuator_ids, 1] = self.base[:, 1] * hi
        return self.level
