"""KIM athlete environments (2026-10-10) — the GRANDMA module's recipes (grandma_env.py: MATT's tasks, Froude-scaled)
on the KIM body (poolympic/bodies.py: 1.55 m, 54 kg, λ = 0.844, 90 % of the size-scaled torque, full self-collision),
without GRANDMA's stoop / careful-step style. Her personality "light and quick" (Claude's pick, user: "give her unique
characteristics") is the body itself (the best strength-to-weight after MATT, short legs) plus quick short steps.

Use with POOLYMPIC_BODY=kim (grandma_env then binds its scaling to her body).
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from . import grandma_env as GE

QUICK_STEP_MAX_AIR = 0.45        # s (MATT units; MATT / GRANDMA 0.6): swing time that still earns the air-time reward


def kim_rung0_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rung 0 — stand + shove / cube recovery, upright athletic stance (MATT's default pose)."""
    return GE.grandma_rung0_env_cfg(play=play)


def kim_rung1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rung 1 — forward walk / run at a commanded speed (Froude-scaled envelope) with a quick cadence."""
    cfg = GE.grandma_rung1_env_cfg(play=play)
    cfg.rewards.pop("careful_steps")
    cfg.rewards["air_time"].params["threshold_max"] = QUICK_STEP_MAX_AIR * GE.TS
    return cfg


def _quick(fn):
    """A grandma_env Rung 2 recipe (each builds on grandma_rung1_env_cfg) with KIM's Rung 1 style instead of GRANDMA's."""
    def make(play: bool = False) -> ManagerBasedRlEnvCfg:
        cfg = fn(play=play)
        cfg.rewards.pop("careful_steps", None)
        cfg.rewards["air_time"].params["threshold_max"] = QUICK_STEP_MAX_AIR * GE.TS
        return cfg
    return make


# Rung 2 chain = GRANDMA's that worked (base -> + lateral-acceleration cap -> sharp kernel + command bands, symmetric
# runner), on KIM's full Froude envelope (grandma_env.GRANDMA_VX_TOP = 4.0 for her)
kim_rung2_base_env_cfg = _quick(GE.grandma_rung2_base_env_cfg)
kim_rung2_cap_env_cfg = _quick(GE.grandma_rung2_cap_env_cfg)
kim_rung2_bands_env_cfg = _quick(GE.grandma_rung2_bands_env_cfg)


def kim_rung2_bands2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """k2_v3 (from k2_v2: 8/10 at it 600 and 900, turntable / sprints / turns all pass). The segments left over the
    0.184 m/s bar are the fast side-steps (|vy| 0.76-0.87 m/s: 0.20-0.28 RMS), slower ones pass. The side-step band
    moves to the fast end (|vy| 0.7-1.1 x SS = 0.64-1.01 m/s) and takes 30 % of the commands; forward 15 %, backward 10 %."""
    cfg = kim_rung2_bands_env_cfg(play=play)
    ss, ws = GE.SS, GE.WS
    cfg.commands["athlete"].bands = ((0.15, (2.0 * ss, GE.GRANDMA_VX_TOP * ss), (0.0, 0.0), 0.6 * ws),
                                     (0.10, (-1.5 * ss, -0.8 * ss), (0.0, 0.0), 0.5 * ws),
                                     (0.30, (-0.5 * ss, 0.5 * ss), (0.7 * ss, 1.1 * ss), 0.0))
    return cfg


def kim_rung0_tight_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """k0_v2 (from k0_v1 it 1499: G1 10/10, 30 seeds 27/30 with 0 falls). The three misses are recovery steps that end
    0.43-0.51 m from the start against the 0.42 m foot box (0.5 m x λ). The stay-near-the-start reward is three times as
    heavy and its kernel 0.6 times as wide, so stepping back after a shove pays more."""
    cfg = kim_rung0_env_cfg(play=play)
    cfg.rewards["near_origin"].weight = 1.5
    cfg.rewards["near_origin"].params["std"] *= 0.6
    return cfg


def kim_rung2_bands3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """k2_v4 (from k2_v3 it 600: G1 10/10, 30 seeds 29/30). The miss is a top-speed sprint with a turn (3.45 m/s, wz
    -0.42: 0.23-0.29 m/s RMS against the 0.184 bar). The forward band moves to the top of her envelope (3.0-4.0 x SS)
    and takes 30 % of the commands; the fast side-step band keeps 20 %, backward 10 %."""
    cfg = kim_rung2_bands_env_cfg(play=play)
    ss, ws = GE.SS, GE.WS
    cfg.commands["athlete"].bands = ((0.30, (3.0 * ss, GE.GRANDMA_VX_TOP * ss), (0.0, 0.0), 0.6 * ws),
                                     (0.10, (-1.5 * ss, -0.8 * ss), (0.0, 0.0), 0.5 * ws),
                                     (0.20, (-0.5 * ss, 0.5 * ss), (0.7 * ss, 1.1 * ss), 0.0))
    return cfg
