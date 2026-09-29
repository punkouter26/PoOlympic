"""Brain confidence (feature 5) — export the PPO critic of an event brain as its own ONNX graph.

The critic is the value head trained next to the actor: V(obs) = the return the policy expects from here on. It reads
the same 84-term contract observation as the actor (checked: critic obs normalizer is (1, 84), no privileged terms), so
Unity can run it on PolicyRunner's observation without any extra state. The actor brains stay byte-identical (parity
gates untouched); the critic is a separate file next to them:

    parity/brains/<brain>.critic.onnx        obs[1,84] --(obs - mean) / (std + 0.01)--> MLP 512-256-128 ELU --> value[1,1]
    parity/brains/<brain>.critic.onnx.json   sidecar (brain name, checkpoint, fingerprint, onnx sha)

ParityHarness.SyncArtifacts copies them into Assets/PoOlympic/Models/Brains with the brains. The raw value is mapped to a
probability by tools/fit_confidence.py (Models/confidence_model.json).

Usage: uv run python tools/export_critic.py <brain.onnx> [<brain.onnx> ...]      (names in parity/brains)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poolympic.policy_export import HIDDEN, OPSET  # noqa: E402

BRAINS = ROOT.parent / "parity" / "brains"
EPS = 1e-2  # rsl_rl EmpiricalNormalization eps (added to std), as tools/export_brain.py


class Critic(nn.Module):
    def __init__(self, sd: dict):
        super().__init__()
        mean = sd["obs_normalizer._mean"][0]
        std = sd["obs_normalizer._std"][0]
        layers, d = [], mean.shape[0]
        for h in HIDDEN:
            layers += [nn.Linear(d, h), nn.ELU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)
        self.mlp.load_state_dict({k[len("mlp."):]: v for k, v in sd.items() if k.startswith("mlp.")})
        self.register_buffer("mean", mean.float().reshape(1, -1))
        self.register_buffer("std", (std.float() + EPS).reshape(1, -1))

    def forward(self, obs: torch.Tensor):
        return self.mlp((obs - self.mean) / self.std)


def export_one(brain: str) -> Path:
    name = brain.removesuffix(".onnx")
    side = json.loads((BRAINS / f"{name}.onnx.json").read_text())
    ckpt = ROOT / Path(side["checkpoint"].replace("\\", "/"))
    if not ckpt.is_file():
        ckpt = Path(side["checkpoint"])
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["critic_state_dict"]
    critic = Critic(sd).eval()
    obs_dim = critic.mean.shape[1]
    out = BRAINS / f"{name}.critic.onnx"
    torch.onnx.export(critic, (torch.zeros(1, obs_dim),), str(out), input_names=["obs"], output_names=["value"],
                      opset_version=OPSET, dynamic_axes=None, dynamo=False)
    model = onnx.load(str(out))
    meta = {"poolympic.name": f"{name}.critic", "poolympic.brain": name, "poolympic.checkpoint": side["checkpoint"],
            "poolympic.fingerprint_sha256": side.get("fingerprint_sha256", ""), "poolympic.obs_dim": str(obs_dim)}
    for k, v in meta.items():
        e = model.metadata_props.add()
        e.key, e.value = k, v
    onnx.checker.check_model(model)
    onnx.save(model, str(out))
    sidecar = {k.replace("poolympic.", ""): v for k, v in meta.items()}
    sidecar["onnx_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    Path(str(out) + ".json").write_text(json.dumps(sidecar, indent=1))

    # verify: ORT graph == the rsl_rl critic math (numpy float32)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    mean, std = sd["obs_normalizer._mean"][0].numpy(), sd["obs_normalizer._std"][0].numpy()
    W = [(sd[f"mlp.{i}.weight"].numpy(), sd[f"mlp.{i}.bias"].numpy()) for i in (0, 2, 4, 6)]
    worst = 0.0
    for o in (mean + rng.normal(0, 1, (300, obs_dim)) * std).astype(np.float32):
        x = (o - mean) / (std + EPS)
        for i, (w, b) in enumerate(W):
            x = x @ w.T + b
            if i < 3:
                x = np.where(x > 0, x, np.expm1(x))
        worst = max(worst, float(abs(sess.run(None, {"obs": o[None]})[0][0, 0] - x[0])))
    print(f"{name}: {ckpt.name} -> {out.name} ({obs_dim} obs), value vs rsl math {worst:.2e}")
    if worst > 1e-3:
        raise SystemExit(f"{name}: critic export mismatch {worst}")
    return out


if __name__ == "__main__":
    for b in sys.argv[1:]:
        export_one(b)
