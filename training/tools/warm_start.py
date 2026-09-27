"""Prepare a warm-start checkpoint for the next rung (lesson of r1_v1, rl_optimization_log.md).

An rsl_rl checkpoint carries running observation statistics. Command dims that were constant in the previous rung
have var = 0, so the next rung's commands would reach the network scaled by 1/(0 + 0.01) = 100x, and a count of
~1e8 samples makes the stats adapt only very slowly. This tool:
  * re-seeds mean/var of the command dims (actor + critic) with the new rung's uniform command distribution
    (standing fraction included),
  * lowers the normalizer count so every stat adapts within a few iterations,
  * floors the policy action std (fresh exploration),
  * resets iter = 0 and drops env_state (the command curriculum restarts at stage 0).

Usage: uv run python tools/warm_start.py <src model_XXXX.pt> <dst run dir> vx_lo vx_hi vy_lo vy_hi wz_lo wz_hi
         [--standing 0.15] [--count 1e6] [--min-std 0.3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poolympic import contract as C  # noqa: E402

CMD = C.OBS_LAYOUT.index(("command", 3))
CMD_OFFSET = sum(s for _, s in C.OBS_LAYOUT[:CMD])


def uniform_stats(lo: float, hi: float, standing: float) -> tuple[float, float]:
    """Mean / variance of a mixture: `standing` mass at 0, the rest U(lo, hi)."""
    m, v = (lo + hi) / 2, (hi - lo) ** 2 / 12
    mean = (1 - standing) * m
    second = (1 - standing) * (v + m * m)
    return mean, max(second - mean * mean, 1e-4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("ranges", type=float, nargs=6)
    ap.add_argument("--standing", type=float, default=0.15)
    ap.add_argument("--count", type=float, default=1e6)
    ap.add_argument("--min-std", type=float, default=0.3)
    a = ap.parse_args()
    sd = torch.load(a.src, map_location="cpu", weights_only=False)
    stats = [uniform_stats(a.ranges[2 * i], a.ranges[2 * i + 1], a.standing) for i in range(3)]
    for group in ("actor_state_dict", "critic_state_dict"):
        g = sd[group]
        for i, (mean, var) in enumerate(stats):
            j = CMD_OFFSET + i
            g["obs_normalizer._mean"][0, j] = mean
            g["obs_normalizer._var"][0, j] = var
            g["obs_normalizer._std"][0, j] = var ** 0.5
        g["obs_normalizer.count"].fill_(a.count)
        print(f"{group}: command mean {g['obs_normalizer._mean'][0, CMD_OFFSET:CMD_OFFSET + 3].tolist()} "
              f"std {g['obs_normalizer._std'][0, CMD_OFFSET:CMD_OFFSET + 3].tolist()}")
    std = sd["actor_state_dict"]["distribution.std_param"]
    std.clamp_(min=a.min_std)
    sd["iter"] = 0
    sd["infos"] = None
    a.dst.mkdir(parents=True, exist_ok=True)
    out = a.dst / "model_0.pt"
    torch.save(sd, out)
    print(f"action std now {[round(x, 3) for x in std.tolist()]}\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
