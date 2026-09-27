# PoOlympics — Build Plan

Design reference: [docs/DESIGN.md](docs/DESIGN.md) (approved 2026-09-27). Run log: `rl_optimization_log.md`.
Legend: `[ ]` todo · `[x]` done · **GATE** = blocking check, work stops until it passes. Each task ends with a git commit.

## Repo layout (target)

```
/                       Unity project (Assets, Packages, ProjectSettings)
  Assets/PoOlympic/     Runtime/ (PolicyRunner, ObservationBuilder, MjCubePool, Shove, BoneBinder)
                        Editor/  (PhysXGuard, fingerprint export, parity tests)
                        Models/  (matt.xml synced from training/, *.onnx)
                        Scenes/  (Testbed_ZeroBrain, Testbed_Rung1, ...)
  SourceArt/            MATT / ZOMBIE / GRANDMA source meshes (Git LFS)
  training/             uv Python project: assets/matt.xml, envs/, train.py, eval_cpu.py,
                        export_onnx.py, record_reference.py, fingerprint.py
  parity/               reference_trajectory_*.json, fingerprint_*.json, gate reports
  docs/DESIGN.md · tasks.md · rl_optimization_log.md
```

---

## Phase 0 — Housekeeping

- [x] **0.1** `git init`; Unity `.gitignore` (Library/, Temp/, Logs/, obj/, UserSettings/, *.csproj, *.slnx); `.gitattributes` with Git LFS for `*.glb *.fbx *.onnx *.png` in SourceArt/Models.
- [x] **0.2** Move the 3 source meshes from project root into `SourceArt/` (outside `Assets/` so Unity doesn't auto-import GRANDMA/ZOMBIE yet).
- [x] **0.3** Create empty `rl_optimization_log.md` with entry template (date · run id · change · hypothesis · result · decision).
- [x] **0.4** Initial commit.

## Phase A — Rig & Physics Body Derivation

- [x] **A1 Version pin.** Determine the MuJoCo version required by the latest `mujoco-warp`; confirm `org.mujoco` Unity package + matching `mujoco.dll` exist for that version and compile on Unity 6000.6. Record the pinned version in DESIGN.md + log. *Accept: one version number used everywhere.*
- [x] **A2 Python env.** `training/` uv project (Python 3.11): `mujoco==<pin>`, `mujoco-warp`, `warp-lang`, PyTorch cu128+, onnx, onnxruntime, tensorboard. *Accept: Warp kernel + a 4096-env `mujoco_warp` humanoid step runs on the 5070 Ti (sm_120) without error; steps/s logged.*
- [x] **A3 Trainer decision.** Try mjlab on Windows. If blocked → minimal PPO (rsl_rl-style) over `mujoco_warp` directly, or WSL2 fallback. *Accept: decision + reason logged.*
- [x] **A4 Skeleton extraction.** `training/tools/extract_skeleton.py`: MATT GLB → joint world positions/orientations in MuJoCo frame (glTF (x,y,z) → MJ (z,x,y)) → `skeleton_matt.json`. *Accept: 52 joints, L/R mirror error < 1 mm, height 1.85 m.*
- [x] **A5 Geom fitting.** Fit capsule radius/length per physics body from skin-weighted vertices; box feet from foot/toe vertex hull. *Accept: geoms visually inside the mesh silhouette (overlay render).*
- [x] **A6 MJCF generator.** `training/tools/build_mjcf.py` → `training/assets/matt.xml` implementing DESIGN §2 exactly (bodies, masses, inertias, joints/ranges, welded parts, passive toes, 23 position actuators with kp/kv/forcerange/armature, contact excludes, solver options, lane-bit defaults, ground plane, 4-cube pool).
- [x] **A7 MJCF validation** (`training/tests/test_model.py`):
  - compiles in CPU `mujoco` and `mujoco_warp`; nq = 32, nv = 31, nu = 23
  - total mass 80.0 ± 0.05 kg; per-body masses match table
  - qpos = 0 → body positions match `skeleton_matt.json` joints < 1 mm (bind-pose invariant)
  - L/R symmetry of masses, ranges, gains
  - passive PD hold at default pose on ground: no explosion, penetration < 5 mm, settles within 1 s
  - *Accept: all tests green.*
- [x] **A8 Visual check.** MuJoCo viewer screenshot of geoms overlaid with MATT mesh (non-colliding visual mesh) at T-pose and default pose → saved to `parity/`. *Accept: user eyeball sign-off.*
- [x] **A9 Contract module.** `training/poolympic/contract.py` — actuator order, default pose, action scale, obs layout (DESIGN §3); exports `contract.json` consumed by C#. Obs builder implemented here once.
- [x] **A10 Fingerprint.** `fingerprint.py` → `parity/fingerprint_python.json` + SHA-256.
- [x] **A11 Zero-brain ONNX.** Export (opset 17, batch 1) two graphs with baked normalization + metadata: `zero_brain.onnx` (ctrl = default pose) and `random_brain.onnx` (random weights, seeded — exercises the full math). *Accept: onnxruntime output == torch output < 1e-6.*
- [x] **A12 CPU reference recorder.** `record_reference.py`: CPU MuJoCo rollout of a given ONNX, 5 s, with scripted disturbances (1 shove + 1 cube drop) → `parity/reference_trajectory_zero.json`, `…_random.json` (schema DESIGN §4). *Accept: re-running with same seed is bit-identical.*

## Phase B — Early Engine Ingestion & Zero-Brain Parity (CRITICAL)

In-editor authoring runs through the Unity CLI (`unity command …`, com.unity.pipeline). Unattended Play-mode runs use deterministic stepping (`ParityTools.StepUntilTick` / `ArmDeterministicStepping`).

- [x] **B1 Packages.** Add `org.mujoco` (pinned, git URL / local package) + native `mujoco.dll`, `com.unity.ai.inference`, `com.unity.cloud.gltfast`. *Accept: empty scene with one MjScene + MjBody steps without errors.*
- [x] **B2 PhysX isolation.** `Physics.simulationMode = Script` (never simulated); `PhysXGuard` editor validator (menu, play-mode enter, build preprocess) fails on any `Rigidbody`/`Collider`/`CharacterController`/`Joint`/`ArticulationBody` in athlete scenes; EditMode test. *Accept: test injects a BoxCollider → fails; clean scene → passes.*
- [x] **B3 MJCF import.** Import `matt.xml` via plug-in importer → `MATT_Physics.prefab` (MjBody/MjGeom/MjHingeJoint/MjActuator hierarchy). Script syncs `training/assets/matt.xml` → `Assets/PoOlympic/Models/` with hash check.
- [x] **B4 GATE G0 — Fingerprint.** C# dumps Unity's compiled `mjModel` → `parity/fingerprint_unity.json`; compare script vs Python. *Accept: ints exact, floats ≤ 1e-6 rel.*
- [x] **B5 Visual binding.** Import MATT GLB (glTFast) → skinned mesh; `BoneBinder` maps bones → MjBody (merged bones follow parents). *Accept: at qpos = 0 every mapped bone within 1 cm / 1° of its MjBody.*
- [x] **B6 Testbed scene** `Testbed_ZeroBrain.unity` authored in-editor via MCP (not procedurally): MuJoCo ground plane, MATT, 9:16 portrait camera (Game view 1080×1920), light, basic HUD (FPS, sim time, tick, lane state).
- [x] **B7 Cube pool + shove.** (Testbed uses the training-identical 4-cube pool so G4/G5 compare the same model; the 16-cube pool lands with the 8-lane scene in C6.) 16 MjBody + MjFreeJoint + MjGeom box cubes authored in-scene, parked resting at `x = 50 + 2i`; `MjCubePool.Fire(pos, vel)` writes qpos/qvel; `Shove.Apply(dv)` adds Δqvel to pelvis free joint. *Accept: no `Instantiate`/`Destroy` in runtime code (grep test); fired cube strikes MATT and collides only via MuJoCo.*
- [x] **B8 Timing.** `Time.fixedDeltaTime = 0.005`; tick counter → policy every 4th step; PolicyRunner execution order before MjScene (or own the step loop if the plug-in can't guarantee order). *Accept: 1000 ticks logged → exactly 4 mj_step per policy call, tick order per DESIGN §3.*
- [x] **B9 PolicyRunner + ObservationBuilder (C#).** Load ONNX via Inference Engine (CPU backend), validate metadata vs fingerprint hash, build obs from `mjData` qpos/qvel per `contract.json`, write `ctrl`.
- [x] **B10 Parity harness.** Unity test runner suite reading `parity/reference_trajectory_*.json`, writing `parity/gate_report_<name>.json`:
  - **GATE G2** obs builder < 1e-5
  - **GATE G3** policy replay < 1e-4 (random_brain)
  - **GATE G4** open-loop ctrl replay, qpos drift < 1e-3 over 1 s (both references)
- [x] **B11 Zero-brain milestone.** Play `Testbed_ZeroBrain`: MATT (untrained, PD-held default pose) reproduces the Python reference: sags, is shoved at 1.0 s, falls at ~2.1 s, cube lands at 2.0 s+; screenshot + gate report committed.

## Phase C — Phased Training & Early Verification Loop

- [x] **C1 Env implementation.** Warp env per DESIGN §3/§5: obs, actions, rewards, terminations, DR, shove + cube pool per env, per-lane trait randomization, command sampling, TensorBoard, checkpoints. *Accept: 10-min smoke run, reward rising, no NaNs.*
- [x] **C2 CPU evaluator (G1).** `eval_cpu.py`: rung bars, 10 seeds, CPU MuJoCo, JSON report + optional video.
- [x] **C3 Rung 0 — train** stand + shove recovery to bar. **GATE G1.** Log all runs.
- [x] **C4 Rung 0 — Unity spot-check.** Export ONNX + reference; G0, G2–G5 in `Testbed_ZeroBrain` with Rung 0 brain.
- [ ] **C5 Rung 1 — train** forward velocity to bar. **GATE G1.**
  - r1_v1 (warm start from Rung 0) FAILED — collapsed into standing still (warm-start obs normalizer + narrow tracking kernel; see log). r1_v2 = from scratch + coarse tracking term + walking-speed curriculum start: running since 16:31 (`runs/matt_rung1/2026-09-27_16-31-44_r1_v2`); it 350 track_lin 86 % of max, per-step vel error ≈ 0.14 m/s at 0.3–1.0 m/s.
- [ ] **C6 Rung 1 — Early Verification Gate (HALT POINT).**
  - export `rung1.onnx` (opset 17, batch 1) + `reference_trajectory_rung1.json` (5 s, CPU MuJoCo)
  - `Testbed_Rung1.unity`: **GATES G0, G2, G3, G4, G5, G6** (8 lanes + cube pool)
  - any divergence → stop training; fix solver config / decimation / friction / gain mapping; re-run gates; log root cause
  - [x] infrastructure: `scene_meet8.xml` (8 lane-isolated athletes `L<k>_`, 1.22 m lanes, 16-cube pool) · lane-safe `PolicyRunner` (per-lane reset, lane origin, cube-slot remap, per-lane recording) · `Testbed_Rung1.unity` (MenuItem *Build Testbed_Rung1*) · `tools/make_g6.py` (lane plan + solo refs + CPU meet pre-check) · `tools/compare_g6.py` (G0 per lane, G6)
  - [x] G0 meet8 PASS (8/8 lanes, each == solo athlete except bits/origin) · G6 harness dry run PASS (interim brain) · solo G5 re-verified (zero/random/rung0, contract v2 brains)
  - [ ] with the G1-passing Rung 1 brain: `make_g6.py` → G2–G4 (auto-discovered refs) → G5 solo → `MeetTestbed.ConfigureG6` + play 250 ticks → `compare_g6.py g6`
- [ ] **C7 Rung 2 — train** omnidirectional + yaw to bar. **GATE G1.**
- [ ] **C8 Rung 2 — Unity spot-check.** G3, G5, G6 with rung2 brain; commands switchable from HUD.

## Phase D — Engine Polish & Game Loop (Step 5, detailed plan written after C6 passes)

- [ ] D1 9:16 tracking broadcast cameras · D2 HUD anchors (TL title · TC FPS/telemetry · TR menu/behaviour selector · BL reset/shove toggle · BR version) · D3 native-API interaction (fire cubes from pool, force on MjBody, behaviour switching) · D4 auto-reset (fall/stall/finish → reset `mjData`) · D5 Sprint Series events 1, 5, 8, 9, 10, 11, 12, 19, 22 · D6 full suite re-validation + perf pass.

## Backlog (later rungs & platforms)

- [ ] Android: arm64-v8a MuJoCo build via NDK + ARM↔desktop trajectory parity + mobile perf pass
- [ ] R3 get-up · R4 ramp/rubble/stairs · R5 jumps/hurdles
- [ ] R6 bodies: rig GRANDMA (AccuRig/Mixamo/Blender), clean + rescale ZOMBIE, derive MJCFs, train variants
- [ ] R7 optional motion-prior polish · R8+ remaining skill events
- [ ] Game layer: betting slip & odds from lane stats, PBP ticker, records, gauntlets
- [ ] Licensing review (Avaturn, Hunyuan3D) before any commercial release
