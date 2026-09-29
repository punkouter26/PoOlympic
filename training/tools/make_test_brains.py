"""A10/A11 — Fingerprint + contract.json + the two Phase-B test brains.

  zero_brain.onnx    action_raw == 0 -> ctrl == default pose (passive PD hold)
  random_brain.onnx  seeded random weights + random normaliser stats (exercises the whole graph)
  random_brain_v4.onnx  the same with the contract v4 input (95 obs: stance-skill block) — G2/G3/G4 of the v4 builder

Verifies onnxruntime == torch (< 1e-6) on 1000 random observations.
Usage: uv run python tools/make_test_brains.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poolympic import contract as C  # noqa: E402
from poolympic.fingerprint import write_python_fingerprint  # noqa: E402
from poolympic.policy_export import ExportedPolicy, export, make_mlp, standard_metadata  # noqa: E402

OUT = ROOT.parent / "parity" / "brains"


def build(kind: str, ath: C.Athlete, obs_dim: int = C.OBS_DIM) -> ExportedPolicy:
    torch.manual_seed(1234)
    mlp = make_mlp(obs_dim)
    rng = np.random.default_rng(1234)
    if kind == "zero":
        last = mlp[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
        mean, std = np.zeros(obs_dim), np.ones(obs_dim)
    else:
        # PyTorch default init keeps action_raw O(1), like a trained policy
        mean = rng.normal(0, 0.2, obs_dim)
        std = rng.uniform(0.5, 2.0, obs_dim)
    return ExportedPolicy(mlp, mean, std, ath.default_pos, ath.range_lo, ath.range_hi)


def main() -> int:
    fp_path, sha = write_python_fingerprint()
    contract = C.export_contract(fingerprint_sha256=sha)
    import mujoco

    ath = C.Athlete.bind(mujoco.MjModel.from_xml_path(str(C.SCENE_XML)))
    print(f"fingerprint {fp_path.name} sha256={sha[:16]}…  contract.json written")
    ok = True
    for kind, dim, version, name in (("zero", C.OBS_DIM, None, "zero_brain"), ("random", C.OBS_DIM, None, "random_brain"),
                                     ("random", C.OBS_DIM_V4, C.SKILL_VERSION, "random_brain_v4")):
        pol = build(kind, ath, dim).eval()
        path = export(pol, OUT / f"{name}.onnx", standard_metadata(contract, name, version))
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        obs = np.random.default_rng(7).normal(0, 1, (1000, dim)).astype(np.float32)
        worst, worst_contract, act_max = 0.0, 0.0, 0.0
        for o in obs:
            ctrl_o, act_o = sess.run(None, {"obs": o[None]})
            with torch.no_grad():
                ctrl_t, act_t = pol(torch.from_numpy(o[None]))
            worst = max(worst, float(np.abs(ctrl_o - ctrl_t.numpy()).max()), float(np.abs(act_o - act_t.numpy()).max()))
            ref_ctrl = C.action_to_ctrl(ath, act_o[0].astype(np.float64))
            worst_contract = max(worst_contract, float(np.abs(ctrl_o[0] - ref_ctrl).max()))
            act_max = max(act_max, float(np.abs(act_o).max()))
        if kind == "zero":
            ctrl, act = sess.run(None, {"obs": obs[:1]})
            assert np.abs(act).max() == 0.0 and np.allclose(ctrl[0], ath.default_pos, atol=1e-6)
        print(f"{path.name}: ort-vs-torch {worst:.2e} | graph-vs-contract(float64) {worst_contract:.2e} | max|action_raw| {act_max:.2f}")
        ok &= worst < 1e-6 and worst_contract < 1e-6
    print("A11", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
