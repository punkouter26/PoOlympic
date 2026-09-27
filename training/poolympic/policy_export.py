"""A11 — ONNX policy export (DESIGN.md §3).

Graph:  obs[1,84] --(obs - mean) / std, clip ±CLIP_OBS--> MLP 512-256-128 ELU --> action_raw[1,23]
        ctrl[1,23] = clip(default + ACTION_SCALE * action_raw, range_lo, range_hi)
Unity copies ctrl straight into mjData.ctrl and feeds action_raw back as the next `last_action` obs term.
Metadata carries the contract + fingerprint hash; Unity refuses to run on mismatch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

from . import contract as C

OPSET = 17
HIDDEN = (512, 256, 128)
CLIP_OBS = 100.0


def make_mlp(obs_dim: int = C.OBS_DIM, act_dim: int = C.NUM_ACTIONS) -> nn.Sequential:
    layers, d = [], obs_dim
    for h in HIDDEN:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    layers.append(nn.Linear(d, act_dim))
    return nn.Sequential(*layers)


class ExportedPolicy(nn.Module):
    def __init__(self, mlp: nn.Module, obs_mean, obs_std, default_pos, range_lo, range_hi, clip_obs: float | None = CLIP_OBS):
        super().__init__()
        self.mlp = mlp
        self.clip_obs = clip_obs
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(1, -1)  # noqa: E731
        self.register_buffer("obs_mean", f(obs_mean))
        self.register_buffer("obs_std", f(obs_std))
        self.register_buffer("default_pos", f(default_pos))
        self.register_buffer("range_lo", f(range_lo))
        self.register_buffer("range_hi", f(range_hi))

    def forward(self, obs: torch.Tensor):
        x = (obs - self.obs_mean) / self.obs_std
        if self.clip_obs is not None:
            x = torch.clamp(x, -self.clip_obs, self.clip_obs)
        action_raw = self.mlp(x)
        ctrl = torch.minimum(torch.maximum(self.default_pos + C.ACTION_SCALE * action_raw, self.range_lo), self.range_hi)
        return ctrl, action_raw


def export(policy: ExportedPolicy, path: Path, metadata: dict[str, str]) -> Path:
    policy = policy.eval().cpu()
    dummy = torch.zeros(1, C.OBS_DIM)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(policy, (dummy,), str(path), input_names=["obs"], output_names=["ctrl", "action_raw"],
                      opset_version=OPSET, dynamic_axes=None, dynamo=False)
    model = onnx.load(str(path))
    for k, v in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = k, v
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    # Sidecar for Unity: the Inference Engine does not expose ONNX metadata_props at runtime.
    sidecar = {k.replace("poolympic.", ""): v for k, v in metadata.items() if k != "poolympic.contract_json"}
    sidecar["onnx_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(str(path) + ".json").write_text(json.dumps(sidecar, indent=1))
    return path


def standard_metadata(contract: dict, name: str) -> dict[str, str]:
    return {
        "poolympic.name": name,
        "poolympic.contract_version": str(contract["contract_version"]),
        "poolympic.fingerprint_sha256": contract["fingerprint_sha256"] or "",
        "poolympic.mujoco_version": contract["mujoco_version"],
        "poolympic.timestep": repr(contract["timestep"]),
        "poolympic.decimation": str(contract["decimation"]),
        "poolympic.actuators": json.dumps([a["name"] for a in contract["actuators"]]),
        "poolympic.contract_json": json.dumps(contract, separators=(",", ":")),
    }
