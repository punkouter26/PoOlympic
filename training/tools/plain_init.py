"""Plain warm start (same rung / same command distribution): copy a checkpoint to <dst>/model_0.pt with iter = 0 and the
env state dropped, keeping the observation normaliser (unlike warm_start.py, which re-seeds the command statistics for
a NEW rung). Optionally floor the action std for fresh exploration.

Usage: uv run python tools/plain_init.py <src model_XXXX.pt> <dst run dir> [--min-std 0.2] [--max-std 0.3]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--min-std", type=float, default=None)
    ap.add_argument("--max-std", type=float, default=None, help="cap the action std (sharper start for a fine-tune)")
    a = ap.parse_args()
    sd = torch.load(a.src, map_location="cpu", weights_only=False)
    if a.min_std is not None:
        sd["actor_state_dict"]["distribution.std_param"].clamp_(min=a.min_std)
    if a.max_std is not None:
        sd["actor_state_dict"]["distribution.std_param"].clamp_(max=a.max_std)
    sd["iter"] = 0
    sd["infos"] = None
    a.dst.mkdir(parents=True, exist_ok=True)
    torch.save(sd, a.dst / "model_0.pt")
    std = sd["actor_state_dict"]["distribution.std_param"]
    print(f"{a.src} -> {a.dst / 'model_0.pt'} (action std {std.min():.3f}..{std.max():.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
