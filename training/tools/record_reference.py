"""A12 — Record golden CPU-MuJoCo reference trajectories for the parity harness.

Usage: uv run python tools/record_reference.py                 # zero + random brains (Phase B)
       uv run python tools/record_reference.py path/to.onnx name  # any trained brain
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poolympic.reference import PARITY, rollout, write  # noqa: E402


def summarize(ref: dict) -> str:
    f = ref["frames"]
    z = np.array([fr["qpos"][2] for fr in f])
    fell = next((fr["t"] for fr in f if fr["qpos"][2] < 0.55), None)
    tau = np.abs(np.array([fr["actuator_force"] for fr in f])).mean()
    return f"pelvis z {z.min():.3f}..{z.max():.3f} m, fell at {fell if fell is not None else 'never'}, mean|tau| {tau:.1f} Nm"


def record(onnx: Path, name: str) -> bool:
    a = rollout(onnx)
    b = rollout(onnx)
    identical = json.dumps(a) == json.dumps(b)
    path = write(a, name)
    print(f"{path.name}: {len(a['frames'])} frames, {summarize(a)}, rerun bit-identical={identical}")
    return identical


def main(argv: list[str]) -> int:
    if len(argv) == 2:
        return 0 if record(Path(argv[0]), argv[1]) else 1
    ok = all(record(PARITY / "brains" / f"{k}_brain.onnx", k) for k in ("zero", "random"))
    print("A12", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
