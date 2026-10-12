"""Rebuild an rsl_rl checkpoint from a DEPLOYED brain when its run directory is gone (training/runs is git-ignored).

A contract brain carries everything the actor needs (observation mean / std and the MLP, tools/export_brain.py) and its
critic file carries the value net (tools/export_critic.py). This writes <dst>/model_0.pt with those weights in the
layout of a template checkpoint of the same architecture (any run of the same body: 84 obs, 512-256-128, 23 actions):
  * actor / critic MLP and normalisers from the ONNX files (std = exported std - 0.01, the epsilon export_brain adds)
  * normaliser count kept high (the statistics are the brain's own; they should drift slowly)
  * action std set to --std (not stored in a brain; 0.15-0.2 = a careful fine-tune)
  * optimiser moments cleared, iter = 0, infos dropped
Verifies: the rebuilt actor reproduces the brain's pre-clip action on random observations.

Usage: uv run python tools/onnx_to_checkpoint.py <brain.onnx> <template model_XXXX.pt> <dst run dir> [--std 0.15]
       (the critic is read from <brain>.critic.onnx next to the brain)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

EPS = 0.01


def _init(path: Path) -> dict[str, np.ndarray]:
    return {i.name: numpy_helper.to_array(i).copy() for i in onnx.load(str(path)).graph.initializer}


def _fill(sd: dict, w: dict[str, np.ndarray], mean: np.ndarray, std_plus_eps: np.ndarray) -> None:
    for k in [k for k in sd if k.startswith("mlp.")]:
        assert tuple(sd[k].shape) == w[k].shape, f"{k}: template {tuple(sd[k].shape)} vs brain {w[k].shape}"
        sd[k] = torch.from_numpy(w[k]).to(sd[k].dtype)
    std = torch.from_numpy(std_plus_eps - EPS).clamp(min=0.0)
    sd["obs_normalizer._mean"] = torch.from_numpy(mean).to(sd["obs_normalizer._mean"].dtype)
    sd["obs_normalizer._std"] = std.to(sd["obs_normalizer._std"].dtype)
    sd["obs_normalizer._var"] = (std ** 2).to(sd["obs_normalizer._var"].dtype)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("brain", type=Path)
    ap.add_argument("template", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--std", type=float, default=0.15)
    a = ap.parse_args()
    actor, critic = _init(a.brain), _init(a.brain.with_suffix("").with_suffix(".critic.onnx"))
    sd = torch.load(a.template, map_location="cpu", weights_only=False)
    _fill(sd["actor_state_dict"], actor, actor["obs_mean"], actor["obs_std"])
    _fill(sd["critic_state_dict"], critic, critic["mean"], critic["std"])
    sd["actor_state_dict"]["distribution.std_param"].fill_(a.std)
    if "optimizer_state_dict" in sd:
        sd["optimizer_state_dict"]["state"] = {}          # Adam moments belong to the template's weights
    sd["iter"] = 0
    sd["infos"] = None

    # check: rebuilt actor == the brain's graph up to the action mapping (ctrl = default + 0.25 * action, clipped)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(a.brain), providers=["CPUExecutionProvider"])
    A = sd["actor_state_dict"]
    rng = np.random.default_rng(0)
    worst = 0.0
    for o in (actor["obs_mean"] + rng.normal(0, 1, (200, actor["obs_mean"].shape[1])) * actor["obs_std"]).astype(np.float32):
        x = (o - A["obs_normalizer._mean"][0].numpy()) / (A["obs_normalizer._std"][0].numpy() + EPS)
        for i in (0, 2, 4, 6):
            x = x @ A[f"mlp.{i}.weight"].numpy().T + A[f"mlp.{i}.bias"].numpy()
            if i < 6:
                x = np.where(x > 0, x, np.expm1(x))
        ctrl = np.clip(actor["default_pos"][0] + 0.25 * x, actor["range_lo"][0], actor["range_hi"][0])
        worst = max(worst, float(np.abs(sess.run(None, {sess.get_inputs()[0].name: o[None]})[0][0] - ctrl).max()))
    a.dst.mkdir(parents=True, exist_ok=True)
    torch.save(sd, a.dst / "model_0.pt")
    print(f"{a.brain.name} -> {a.dst / 'model_0.pt'} (template {a.template.name}, action std {a.std}): ctrl vs brain {worst:.2e}")
    return 0 if worst < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
