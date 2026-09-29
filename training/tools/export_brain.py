"""Export an rsl_rl checkpoint (mjlab run) to a contract ONNX brain + sidecar, and verify it.

The rsl_rl actor is  mlp( (obs - mean) / (std + 0.01) )  (EmpiricalNormalization, no clipping). We rebuild exactly
that inside the contract graph (ExportedPolicy with clip_obs=None, std := std + 0.01) and append the contract's
action mapping (ctrl = clip(default + 0.25·a, joint range)).

Usage: uv run python tools/export_brain.py <run_dir or model_XXXX.pt> <name>   -> parity/brains/<name>.onnx(.json)
A 95-input checkpoint (Rung S) is exported as a contract v4 brain (sidecar contract_version 4: v3 obs + skill block).
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poolympic import contract as C  # noqa: E402
from poolympic.policy_export import ExportedPolicy, export, make_mlp, standard_metadata  # noqa: E402

EPS = 1e-2  # rsl_rl EmpiricalNormalization eps (added to std)


def latest_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    pts = sorted(path.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not pts:
        raise FileNotFoundError(f"no model_*.pt in {path}")
    return pts[-1]


def main(src: str, name: str) -> int:
    ckpt = latest_checkpoint(Path(src))
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]
    mean = sd["obs_normalizer._mean"][0].numpy()
    std = sd["obs_normalizer._std"][0].numpy()
    obs_dim = mean.shape[0]
    assert obs_dim in (C.OBS_DIM, C.OBS_DIM_V4), f"checkpoint has {obs_dim} inputs"
    version = C.SKILL_VERSION if obs_dim == C.OBS_DIM_V4 else None
    mlp = make_mlp(obs_dim)
    mlp.load_state_dict({k[len("mlp."):]: v for k, v in sd.items() if k.startswith("mlp.")})
    ath = C.Athlete.bind(mujoco.MjModel.from_xml_path(str(C.SCENE_XML)))
    pol = ExportedPolicy(mlp, mean, std + EPS, ath.default_pos, ath.range_lo, ath.range_hi, clip_obs=None).eval()

    from poolympic.fingerprint import write_python_fingerprint

    _, sha = write_python_fingerprint()
    contract = C.export_contract(fingerprint_sha256=sha)
    meta = standard_metadata(contract, name, version)
    meta["poolympic.checkpoint"] = str(ckpt)
    out = export(pol, ROOT.parent / "parity" / "brains" / f"{name}.onnx", meta)

    # verify: ORT graph == reference rsl_rl actor math (float32), and ctrl == contract mapping
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    obs = (mean + rng.normal(0, 1, (500, obs_dim)) * std).astype(np.float32)
    worst_a = worst_c = 0.0
    W = [(sd[f"mlp.{i}.weight"].numpy(), sd[f"mlp.{i}.bias"].numpy()) for i in (0, 2, 4, 6)]
    for o in obs:
        x = (o - mean) / (std + EPS)
        for i, (w, b) in enumerate(W):
            x = x @ w.T + b
            if i < 3:
                x = np.where(x > 0, x, np.expm1(x))
        ctrl, act = sess.run(None, {"obs": o[None]})
        worst_a = max(worst_a, float(np.abs(act[0] - x).max()))
        worst_c = max(worst_c, float(np.abs(ctrl[0] - C.action_to_ctrl(ath, act[0].astype(np.float64))).max()))
    print(f"{ckpt.name} -> {out.name} (contract v{version or contract['contract_version']}, {obs_dim} obs): "
          f"action vs rsl math {worst_a:.2e}, ctrl vs contract {worst_c:.2e}")
    return 0 if worst_a < 1e-4 and worst_c < 1e-5 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
