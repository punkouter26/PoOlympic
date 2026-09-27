"""Minimal glTF 2.0 binary (.glb) reader for skeleton + skinned-mesh extraction.

Only what the body-derivation pipeline needs: node hierarchy, world transforms,
skins (joints + inverse bind matrices) and skinned vertex attributes.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_COMPONENT_DTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_TYPE_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

# glTF is right-handed, Y-up, character faces +Z, character's left is +X.
# MuJoCo convention used by this project: X forward, Y left, Z up.
# p_mj = GLTF_TO_MJ @ p_gltf  ->  (x, y, z)_mj = (z, x, y)_gltf  (proper rotation, det = +1).
GLTF_TO_MJ = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _quat_xyzw_to_mat(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _node_local(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=float).reshape(4, 4).T
    m = np.eye(4)
    m[:3, :3] = _quat_xyzw_to_mat(node.get("rotation", [0, 0, 0, 1])) @ np.diag(node.get("scale", [1, 1, 1]))
    m[:3, 3] = node.get("translation", [0, 0, 0])
    return m


@dataclass
class SkinnedMesh:
    name: str
    positions: np.ndarray  # (V, 3) glTF mesh space
    joints: np.ndarray  # (V, 4) indices into Skin.joint_nodes
    weights: np.ndarray  # (V, 4)


@dataclass
class Skin:
    joint_nodes: list[int]
    inverse_bind: np.ndarray  # (J, 4, 4)
    meshes: list[SkinnedMesh] = field(default_factory=list)


class Glb:
    def __init__(self, path: str | Path):
        data = Path(path).read_bytes()
        magic, _version, _length = struct.unpack("<4sII", data[:12])
        if magic != b"glTF":
            raise ValueError(f"{path} is not a .glb file")
        json_len = struct.unpack("<I", data[12:16])[0]
        self.gltf = json.loads(data[20 : 20 + json_len])
        bin_start = 20 + json_len
        bin_len = struct.unpack("<I", data[bin_start : bin_start + 4])[0]
        self.bin = data[bin_start + 8 : bin_start + 8 + bin_len]

        self.nodes: list[dict] = self.gltf["nodes"]
        self.parent: dict[int, int] = {}
        for i, n in enumerate(self.nodes):
            for c in n.get("children", []):
                self.parent[c] = i
        self._world: dict[int, np.ndarray] = {}

    def accessor(self, index: int) -> np.ndarray:
        acc = self.gltf["accessors"][index]
        view = self.gltf["bufferViews"][acc["bufferView"]]
        dtype = _COMPONENT_DTYPE[acc["componentType"]]
        width = _TYPE_WIDTH[acc["type"]]
        offset = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = view.get("byteStride")
        item = np.dtype(dtype).itemsize * width
        if stride and stride != item:
            raw = np.frombuffer(self.bin, np.uint8, count=stride * acc["count"], offset=offset)
            raw = raw.reshape(acc["count"], stride)[:, :item].copy()
            arr = raw.view(dtype).reshape(acc["count"], width)
        else:
            arr = np.frombuffer(self.bin, dtype, count=acc["count"] * width, offset=offset).reshape(acc["count"], width)
        arr = arr.astype(np.float64) if dtype == np.float32 else arr.astype(np.int64)
        if acc.get("normalized") and dtype != np.float32:
            arr = arr / np.iinfo(dtype).max
        return arr

    def world(self, node: int) -> np.ndarray:
        if node not in self._world:
            m = _node_local(self.nodes[node])
            if node in self.parent:
                m = self.world(self.parent[node]) @ m
            self._world[node] = m
        return self._world[node]

    def name(self, node: int) -> str:
        return self.nodes[node].get("name", f"node_{node}")

    def skins(self) -> list[Skin]:
        out = []
        for si, s in enumerate(self.gltf.get("skins", [])):
            ibm = self.accessor(s["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
            skin = Skin(joint_nodes=list(s["joints"]), inverse_bind=ibm)
            for ni, n in enumerate(self.nodes):
                if n.get("skin") != si or "mesh" not in n:
                    continue
                mesh = self.gltf["meshes"][n["mesh"]]
                for prim in mesh["primitives"]:
                    a = prim["attributes"]
                    skin.meshes.append(SkinnedMesh(
                        name=mesh.get("name", f"mesh_{n['mesh']}"),
                        positions=self.accessor(a["POSITION"]),
                        joints=self.accessor(a["JOINTS_0"]),
                        weights=self.accessor(a["WEIGHTS_0"]),
                    ))
            out.append(skin)
        return out


def to_mj(points: np.ndarray) -> np.ndarray:
    """Convert (..., 3) glTF points/vectors into the MuJoCo frame."""
    return points @ GLTF_TO_MJ.T


def rot_to_mj(r: np.ndarray) -> np.ndarray:
    """Convert a glTF rotation matrix into the MuJoCo frame (change of basis)."""
    return GLTF_TO_MJ @ r @ GLTF_TO_MJ.T
