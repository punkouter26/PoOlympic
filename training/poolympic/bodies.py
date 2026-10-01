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
    stiffness: float | None = None    # × the size-scaled PD gains (kp, kv); None = `strength` (zombie: weak AND soft)
    self_collision: bool = False      # True: every body geom collides with every other (parent/child pairs excepted)
    joint_defaults: dict = field(default_factory=dict)   # default pose overrides (deg) by joint name, both sides
    rig: str | None = None            # variant of another body: reuse its skeleton/skin and its rules (same size)
    torque_caps: dict | None = None   # per joint (base name) (Nm against, Nm towards the anatomical + direction)

    @property
    def family(self) -> str:
        """The rig this body is built from ('matt' for MATT and his variants): picks size-dependent rules."""
        return self.rig or self.name

    # ---- paths -------------------------------------------------------------------------------------------------
    @property
    def robot_xml(self) -> Path: return ASSETS / f"{self.name}.xml"
    @property
    def scene_xml(self) -> Path: return ASSETS / f"scene_{self.name}.xml"
    @property
    def skeleton_json(self) -> Path: return DERIVED / f"skeleton_{self.family}.json"
    @property
    def skin_npz(self) -> Path: return DERIVED / f"skin_{self.family}.npz"
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
    @property
    def gain_scale(self) -> float:
        """PD stiffness scale: mass × length × stiffness (default: the torque scale)."""
        return self.mass_scale * self.length_scale * (self.strength if self.stiffness is None else self.stiffness)


MATT_MASS = 80.0
MATT_HEIGHT = 1.837          # body mesh height (skeleton_matt.json height_body_mesh_m)
ZOMBIE_HEIGHT = 1.135        # SourceArt/Zombie/convert_report.json
GRANDMA_HEIGHT = 1.600       # SourceArt/Grandma/rig_report.json

BODIES = {
    "matt": Body("matt", ROOT.parent / "SourceArt" / "test_MATT_Avaturn.glb", 52, (1.80, 1.90), MATT_MASS, 1.0),
    "zombie": Body(
        "zombie", ROOT.parent / "SourceArt" / "Zombie" / "zombie.glb", 22, (1.10, 1.17), 22.0,
        ZOMBIE_HEIGHT / MATT_HEIGHT, strength=0.7, self_collision=True,
        # zombie stance: slight forward lean, crouched legs, wide feet, arms held out forward (elevation ~0 = level,
        # flexion swings the level arm forward), elbows a little bent
        joint_defaults={"abdomen_flex": 15.0, "hip_flex": 20.0, "hip_abd": 5.0, "knee": 35.0, "ankle_dorsi": 15.0,
                        "shoulder_elev": -10.0, "shoulder_flex": 75.0, "elbow": 20.0}),
    # GRANDMA (user decisions 2026-09-30: Claude rigs the unrigged scan — SourceArt/Grandma/rig_grandma.py —, 1.60 m,
    # "frail but steady"): stocky 1.60 m woman, 65 kg; 60 % of the size-scaled torque (knee extension cap ≈ 119 Nm);
    # full self-collision; slightly stooped soft-kneed stance (flat feet: shin lean = knee − hip = ankle).
    # stiffness 1.0: frail = weaker torque caps, not floppy servos. With kp × 0.6 (g0_v1) the hip stiffness (127 Nm/rad)
    # was below the gravity gradient of her trunk (~150 Nm/rad): the trunk folded forward in 0.8 s, nothing to learn from.
    "grandma": Body(
        "grandma", ROOT.parent / "SourceArt" / "Grandma" / "grandma.glb", 22, (1.57, 1.63), 65.0,
        GRANDMA_HEIGHT / MATT_HEIGHT, strength=0.6, stiffness=1.0, self_collision=True,
        # stance: pelvis level (hip − knee + ankle = 0), shins 7° forward; COM 35 % of the way heel -> toe like MATT (the
        # first stance, hip 14 / ankle 10, put it at 43 %: 6 cm ahead of the ankles)
        joint_defaults={"abdomen_flex": 10.0, "hip_flex": 17.0, "hip_abd": 3.0, "knee": 24.0, "ankle_dorsi": 7.0}),
}


# MATT-bio (STAGED 2026-09-29, not trained; needs the user's OK before it replaces MATT — DESIGN §2 change, every MATT
# brain retrains): MATT with (1) full self-collision like the zombie (arms can no longer pass through the trunk / thighs;
# checked: no geom pair touches at the T-pose, the default stance or a ±40° running arm swing, so no new excludes) and
# (2) joint-specific, direction-specific torque caps instead of one symmetric cap per group. Values = approximate
# peak isometric torques of a strong adult male (Anderson et al. 2007 J Biomech 40:3105; Harbo et al. 2012 Eur J Appl
# Physiol 112:267), rounded, capped at MATT's current group caps. Tuple = (cap against the + direction, cap towards it);
# + = flexion / abduction / dorsiflexion / inversion / elevation / left-lateral / left-twist (build_mjcf sign convention).
BIO_TORQUE_CAPS = {
    "abdomen_flex": (220.0, 180.0),   # extension (back) is stronger than flexion (abs)
    "abdomen_lat": (150.0, 150.0),
    "abdomen_twist": (80.0, 80.0),    # trunk rotation is weak compared with flexion
    "shoulder_elev": (80.0, 70.0),    # adduction / abduction
    "shoulder_flex": (80.0, 70.0),    # extension / flexion
    "shoulder_twist": (50.0, 50.0),   # internal / external rotation
    "elbow": (55.0, 70.0),            # extension / flexion
    "hip_flex": (280.0, 200.0),       # extension (glutes, hamstrings) / flexion
    "hip_abd": (150.0, 150.0),        # adduction / abduction
    "hip_rot": (80.0, 80.0),          # was 280: hip rotators produce ~60-100 Nm
    "knee": (280.0, 150.0),           # extension (quads) / flexion (hamstrings)
    "ankle_dorsi": (220.0, 60.0),     # plantarflexion (calf) / dorsiflexion (tibialis, was 220)
    "ankle_inv": (45.0, 60.0),        # eversion / inversion (was 220 both ways)
}

BODIES["mattbio"] = Body("mattbio", BODIES["matt"].glb, 52, (1.80, 1.90), MATT_MASS, 1.0, self_collision=True, rig="matt",
                         torque_caps=BIO_TORQUE_CAPS)


def current() -> Body:
    name = os.environ.get("POOLYMPIC_BODY", "matt").lower()
    if name not in BODIES:
        raise KeyError(f"POOLYMPIC_BODY={name!r}: known bodies {sorted(BODIES)}")
    return BODIES[name]
