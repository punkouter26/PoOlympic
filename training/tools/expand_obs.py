"""Warm start for Rung S (contract v4): grow a v3 checkpoint (84 obs) to the v4 input (95 obs = + stance-skill block).

  * actor + critic first layer: 11 zero columns → the network computes EXACTLY the v3 function at iteration 0, whatever
    the skill values (the new inputs have no effect until PPO moves those weights)
  * observation normaliser: the v3 stats are kept; the 11 skill dims get mean / variance of the Rung S skill command
    distribution (Monte Carlo of tasks/skill_mdp.AthleteSkillCommand's sampling), count lowered so all stats adapt
  * action std floored (fresh exploration), iter = 0, infos dropped (the curriculum restarts)

Verifies: expanded actor(obs95) == source actor(obs84) for random obs + random skill blocks.
Usage: uv run python tools/expand_obs.py <src model_XXXX.pt> <dst run dir> [--count 1e6] [--min-std 0.25]
       e.g.  tools/expand_obs.py runs/matt_rung2/2026-09-28_04-54-44_r2_v8/model_600.pt runs/matt_stance/rs_init
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poolympic import contract as C  # noqa: E402
from poolympic import skills as K  # noqa: E402

MODE_PROBS = (0.20, 0.16, 0.16, 0.16, 0.16, 0.16)   # = skill_mdp.AthleteSkillCommandCfg.mode_probs


def skill_samples(n: int, rng: np.random.Generator) -> np.ndarray:
    """n skill blocks drawn like AthleteSkillCommand._resample_command (MATT ranges from contract.skill_block)."""
    import mujoco
    r = C.skill_block()["ranges"]
    m = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("default").id)
    mujoco.mj_forward(m, d)
    shoulder = {a: K.shoulder_in_heading(d, m, a) for a in (-1, 1)}
    out = np.zeros((n, C.SKILL_DIM))
    modes = rng.choice(len(MODE_PROBS), size=n, p=MODE_PROBS)
    for i, mode in enumerate(modes):
        s = out[i]
        if mode == 1:
            s[0] = rng.uniform(*r["pelvis_height"])
        elif mode == 2:
            s[1 if rng.uniform() < 0.5 else 2] = 1.0
        elif mode == 3:
            s[3], s[4] = rng.uniform(*r["march_hz"]), rng.uniform(*r["knee_lift"])
        elif mode == 4:
            s[5], s[6] = rng.uniform(*r["torso_yaw"]), rng.uniform(*r["torso_pitch"])
        elif mode == 5:
            arm = -1 if rng.uniform() < 0.5 else 1
            s[7:10] = K.sample_reach_target(rng, shoulder[arm], arm, r["hand_reach"])
            s[10] = arm
    return out


def mlp_forward(sd: dict, x: torch.Tensor) -> torch.Tensor:
    x = (x - sd["obs_normalizer._mean"]) / (sd["obs_normalizer._std"] + 1e-2)
    for i in (0, 2, 4):
        x = torch.nn.functional.elu(x @ sd[f"mlp.{i}.weight"].T + sd[f"mlp.{i}.bias"])
    return x @ sd["mlp.6.weight"].T + sd["mlp.6.bias"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--count", type=float, default=1e6)
    ap.add_argument("--min-std", type=float, default=0.25)
    a = ap.parse_args()
    sd = torch.load(a.src, map_location="cpu", weights_only=False)
    src = {g: {k: v.clone() for k, v in sd[g].items()} for g in ("actor_state_dict", "critic_state_dict")}
    assert sd["actor_state_dict"]["mlp.0.weight"].shape[1] == C.OBS_DIM, "source is not a v3 (84-obs) checkpoint"
    sk = skill_samples(200_000, np.random.default_rng(0))
    mean = torch.as_tensor(sk.mean(0), dtype=torch.float32)[None]
    var = torch.as_tensor(np.maximum(sk.var(0), 1e-4), dtype=torch.float32)[None]
    for group in ("actor_state_dict", "critic_state_dict"):
        g = sd[group]
        w = g["mlp.0.weight"]
        g["mlp.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], C.SKILL_DIM, dtype=w.dtype)], dim=1)
        g["obs_normalizer._mean"] = torch.cat([g["obs_normalizer._mean"], mean], dim=1)
        g["obs_normalizer._var"] = torch.cat([g["obs_normalizer._var"], var], dim=1)
        g["obs_normalizer._std"] = torch.cat([g["obs_normalizer._std"], var.sqrt()], dim=1)
        g["obs_normalizer.count"].fill_(a.count)
        print(f"{group}: first layer {tuple(w.shape)} -> {tuple(g['mlp.0.weight'].shape)}")
    std = sd["actor_state_dict"]["distribution.std_param"]
    std.clamp_(min=a.min_std)
    sd["iter"] = 0
    sd["infos"] = None
    # first-layer shapes changed: keep the parameter groups (lr, betas, ...), drop the Adam moments
    sd["optimizer_state_dict"] = {"state": {}, "param_groups": sd["optimizer_state_dict"]["param_groups"]}

    # the expanded actor is the source function at iteration 0
    x84 = torch.randn(512, C.OBS_DIM)
    x95 = torch.cat([x84, torch.as_tensor(sk[:512], dtype=torch.float32)], dim=1)
    worst = float((mlp_forward(sd["actor_state_dict"], x95) - mlp_forward(src["actor_state_dict"], x84)).abs().max())
    print(f"expanded actor(obs95) vs source actor(obs84): max |diff| {worst:.2e}")
    assert worst < 1e-4, worst
    names = [n for n, _ in C.SKILL_LAYOUT for _ in range(dict(C.SKILL_LAYOUT)[n])]
    print("skill stats (mean / std):", [(n, round(float(mm), 3), round(math.sqrt(float(v)), 3))
                                        for n, mm, v in zip(names, mean[0], var[0])])
    a.dst.mkdir(parents=True, exist_ok=True)
    out = a.dst / "model_0.pt"
    torch.save(sd, out)
    print(f"action std floor {a.min_std}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
