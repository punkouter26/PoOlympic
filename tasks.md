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
- [x] **C5 Rung 1 — train** forward velocity to bar. **GATE G1.**
  - r1_v1 (warm start) failed — standing collapse. r1_v2 (from scratch + coarse tracking) → **Rung 1 brain = `r1_v2_it2000` (`parity/brains/rung1.onnx`)**: G1 10/10 + margin 30/30, 0 falls, lane drift ≤ 0.42 m, vel RMS ≤ 0.121 m/s. G1 uses the contract's lane-keeping steering (user decision 2026-09-27); later checkpoints (2500/2999) regress on slow-speed lane drift.
- [x] **C6 Rung 1 — Early Verification Gate (HALT POINT).** PASSED 2026-09-27 — brain `rung1.onnx` (r1_v2_it2000).
  - export `rung1.onnx` (opset 17, batch 1) + `reference_trajectory_rung1.json` (5 s, CPU MuJoCo)
  - `Testbed_Rung1.unity`: **GATES G0, G2, G3, G4, G5, G6** (8 lanes + cube pool)
  - any divergence → stop training; fix solver config / decimation / friction / gain mapping; re-run gates; log root cause
  - [x] infrastructure: `scene_meet8.xml` (8 lane-isolated athletes `L<k>_`, 1.22 m lanes, 16-cube pool) · lane-safe `PolicyRunner` (per-lane reset, lane origin, cube-slot remap, per-lane recording) · `Testbed_Rung1.unity` (MenuItem *Build Testbed_Rung1*) · `tools/make_g6.py` (lane plan + solo refs + CPU meet pre-check) · `tools/compare_g6.py` (G0 per lane, G6)
  - [x] G0 meet8 PASS (8/8 lanes, each == solo athlete except bits/origin) · G6 harness dry run PASS (interim brain) · solo G5 re-verified (zero/random/rung0, contract v2 brains)
  - [x] `make_g6.py rung1` → plan + 8 solo refs; CPU meet pre-check PASS (5 s drift ≤ 2e-7)
  - [x] Unity: G0 meet8 8/8 · G2 ≤ 4e-16 · G3 ≤ 2.4e-6 · G4 (1 s) ≤ 1.8e-6 on all 8 lane refs · **G5/G6: all 8 lanes of Testbed_Rung1 vs their solo CPU runs PASS — qpos drift ≤ 2e-6 @ 1 s, ≤ 1.6e-5 @ 5 s, torque ratio 1.0000, same (no-)fall outcomes** · SteeringTests PASS · EditMode 18/18 · capture `parity/c6/rung1_tick252.png`
- [ ] **C7 Rung 2 — train** omnidirectional + yaw to bar. **GATE G1.**
- [ ] **C8 Rung 2 — Unity spot-check.** G3, G5, G6 with rung2 brain; commands switchable from HUD.

## Phase D — Engine Polish & Game Loop (detailed plan, written 2026-09-27 after C6 passed)

Rules: every surface an athlete can touch is a MuJoCo geom generated into the MJCF (training, CPU gates and Unity share it); stadium / dressing art from Blender is render-only (no colliders — PhysXGuard). Events run on control ticks with seeded RNG (reproducible attempts). Runs in parallel with C7/C8 (GPU trains, editor builds).

- [ ] **D1 Event 1 — Iron Pedestal (vertical slice, 1 biped)**
  - [x] physics: `scene_pedestal.xml` — 1 m × 1 m × 0.5 m block, top at z = 0, ground at −0.5 (athlete pose / obs / fall rule unchanged)
  - [x] CPU check: rung-0 brain survives 40/40 seeds on the pedestal (20 s, 0.5 m/s gusts + cubes); foot overhang ≤ 0.10 m
  - [x] Unity: `IronPedestalEvent` (countdown → 20 s live → result → auto reset; gusts + cube drops on seeded tick schedule; out = fall rule / foot below pedestal top / non-foot contact), `EventHud` (D2 anchors), `Event_IronPedestal.unity` via *PoOlympic › Events › Build Event 1*; play-through SURVIVED 20.00 s · EditMode 18/18
  - [ ] G0 for the pedestal scene (Python vs Unity fingerprint) + a G5-style closed-loop parity run of one attempt
  - [ ] Iron Pedestal fine-tune (foot-on-pedestal term, trained on the pedestal geometry) for edge margin
- [x] **D2 Stadium (Blender MCP, Olympic realistic)** — `SourceArt/Stadium/stadium.blend` → `Assets/PoOlympic/Art/Stadium/Stadium.glb` (1.5 MB, ~28k tris, render-only): World-Athletics 400 m oval (84.39 m straights, R 36.5 m, 8 × 1.22 m lanes = meet layout), 20 m start extension, finish / 100 m lines; two-tier bowl with front-view crowd textures; roof canopy + floodlight ring; LED boards; 2 scoreboards. **Venues for every event category** with `VENUE_*` anchors (glTF nodes + `venues.json`): HomeStraight (8, 19, 9, 10, 22, hurdles) · CentreStage (1, 5) · Agility (11 slalom, 12 turntable) · Terrain (14–17, parkour) · Jumps (runways + pit) · Mats (27 get-up, crawl, flip) · Skills (carry, bench, kick + goal) · BackStraight. `EventScenes.PlaceStadium(venue, groundY)` snaps a venue onto the MuJoCo origin (verified: home anchor at Unity (−57.81, −0.50, −41.38)). Iron Pedestal scene now plays inside the stadium (`parity/d1/stadium_live.png`).
  - [ ] confirm the full 30-event list (only ~20 are named in DESIGN.md) and add any missing venue types
- [ ] **D3 Broadcast** — 9:16 tracking cameras per event, HUD polish (UI Toolkit), result cards
- [ ] **D4 8-lane Iron Pedestal heat** — per-lane traits (strength / latency / obs noise) → survival odds; pedestal per lane in the meet MJCF
- [ ] **D5 remaining Sprint Series events** 5 Gust Gauntlet · 8 30m Dash · 19 Terminal Velocity (rung1) · 9, 10, 11, 12, 22 (rung2, after C8)
- [ ] **D6** full suite re-validation + perf pass

## Backlog (later rungs & platforms)

- [ ] Android: arm64-v8a MuJoCo build via NDK + ARM↔desktop trajectory parity + mobile perf pass
- [ ] R3 get-up · R4 ramp/rubble/stairs · R5 jumps/hurdles
- [ ] R6 bodies: rig GRANDMA (AccuRig/Mixamo/Blender), clean + rescale ZOMBIE, derive MJCFs, train variants
- [ ] R7 optional motion-prior polish · R8+ remaining skill events
- [ ] Game layer: betting slip & odds from lane stats, PBP ticker, records, gauntlets
- [ ] Licensing review (Avaturn, Hunyuan3D) before any commercial release
