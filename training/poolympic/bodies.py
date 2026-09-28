"""Athlete bodies (tasks.md Phase Z). One profile per trained body; a Python process works on one body, chosen with the
environment variable POOLYMPIC_BODY (default "matt"), so training / evaluation / export modules keep their module-level
constants. MATT's profile reproduces the original single-body constants exactly (his artefacts stay byte-identical).

Size scaling (zombie): lengths λ = height ratio, dynamic similarity (Froude): speeds × √λ, times and 1/frequencies × √λ,
torques ∝ mass × length. The zombie is "weaker but relentless" (user decision 2026-09-28): 70 % of the size-scaled torque.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # training/
ASSETS = ROOT / "assets"
DERIVED = ASSETS / "derived"
PARITY = ROOT.parent / "parity"


@dataclass(frozen=True)
class Body:
    name: str
    glb: Path                         # T-pose skinned source (extract_skeleton.py)
    n_joints: int                     # skeleton joints expected in the GLB
    height_range: tuple[float, float]
    total_mass: float                 # kg, distributed with the de Leva table
    length_scale: float               # λ relative to MATT (height ratio)
    strength: float = 1.0             # × the size-scaled actuator torque (the "weaker" knob)
    self_collision: bool = False      # True: every body geom collides with every other (parent/child pairs excepted)
    joint_defaults: dict = field(default_factory=dict)   # default pose overrides (deg) by joint name, both sides

    # ---- paths -------------------------------------------------------------------------------------------------
    @property
    def robot_xml(self) -> Path: return ASSETS / f"{self.name}.xml"
    @property
    def scene_xml(self) -> Path: return ASSETS / f"scene_{self.name}.xml"
    @property
    def skeleton_json(self) -> Path: return DERIVED / f"skeleton_{self.name}.json"
    @property
    def skin_npz(self) -> Path: return DERIVED / f"skin_{self.name}.npz"
    @property
    def body_report(self) -> Path: return DERIVED / ("body_report.json" if self.name == "matt" else f"body_report_{self.name}.json")
    @property
    def contract_json(self) -> Path: return PARITY / ("contract.json" if self.name == "matt" else f"contract_{self.name}.json")
    @property
    def fingerprint_json(self) -> Path:
        return PARITY / ("fingerprint_python.json" if self.name == "matt" else f"fingerprint_python_{self.name}.json")

    # ---- Froude scaling ----------------------------------------------------------------------------------------
    @property
    def speed_scale(self) -> float: return math.sqrt(self.length_scale)
    @property
    def time_scale(self) -> float: return math.sqrt(self.length_scale)
    @property
    def mass_scale(self) -> float: return self.total_mass / MATT_MASS
    @property
    def torque_scale(self) -> float: return self.mass_scale * self.length_scale * self.strength
    @property
    def inertia_scale(self) -> float: return self.mass_scale * self.length_scale ** 2


MATT_MASS = 80.0
MATT_HEIGHT = 1.837          # body mesh height (skeleton_matt.json height_body_mesh_m)
ZOMBIE_HEIGHT = 1.135        # SourceArt/Zombie/convert_report.json

BODIES = {
    "matt": Body("matt", ROOT.parent / "SourceArt" / "test_MATT_Avaturn.glb", 52, (1.80, 1.90), MATT_MASS, 1.0),
    "zombie": Body(
        "zombie", ROOT.parent / "SourceArt" / "Zombie" / "zombie.glb", 22, (1.10, 1.17), 22.0,
        ZOMBIE_HEIGHT / MATT_HEIGHT, strength=0.7, self_collision=True,
        # zombie stance: slight forward lean, crouched legs, wide feet, arms held out forward (elevation ~0 = level,
        # flexion swings the level arm forward), elbows a little bent
        joint_defaults={"abdomen_flex": 15.0, "hip_flex": 20.0, "hip_abd": 5.0, "knee": 35.0, "ankle_dorsi": 15.0,
                        "shoulder_elev": -10.0, "shoulder_flex": 75.0, "elbow": 20.0}),
}


def current() -> Body:
    name = os.environ.get("POOLYMPIC_BODY", "matt").lower()
    if name not in BODIES:
        raise KeyError(f"POOLYMPIC_BODY={name!r}: known bodies {sorted(BODIES)}")
    return BODIES[name]
