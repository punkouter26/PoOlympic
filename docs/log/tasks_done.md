# Finished task sections

Moved from the root tasks.md on 2026-10-14: every item done.

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
- [x] **C7 Rung 2 — train** omnidirectional + yaw to bar. **GATE G1.** — `rung2.onnx` = **r2_v8 it 600**: G1 **10/10** (0 falls, turntable 2.64/2.58 s, brake 1.16 m, 20 m backward); 20-seed gate 18/20; margin 23/30 official protocol / 28/30 steady-state variant — the remaining misses are sprints from standstill (acceleration ~1.5 m/s², top speed ≈ 3.8 m/s < the 4.0 envelope top; see log, open questions for the user). Path: r2_v1 (yaw decay) → r2_v2 (yaw rewards) → symmetry aug + mirror loss → **contract v3** clock (yaw weight 1.2) → widened envelope → lateral-accel command cap (r2_v6 10/10) → sharper speed kernel (r2_v7) → sprint-focus sampling (r2_v8).
- [x] **C8 Rung 2 — Unity spot-check.** G3, G5, G6 with rung2 brain; commands switchable from HUD. — G2 4.4e-16 · G3 ≤ 4.3e-6 · G4 ≤ 9.5e-7 on 8 lane refs (stand / back / crab ± / spin ± / sprint 3.5 / walk-turn); **G6 PASS 8/8** in Testbed_Rung1 (1 s drift ≤ 4.6e-6); Testbed_Rung1 HUD presets Stop / Walk / Sprint / Back / Crab ◀▶ / Spin. **Phase C complete.**

## Phase K — KIM: fourth athlete body (added 2026-10-10)

User (2026-10-10): "i have a new unique player to train · train her · give her unique characteristics · add her to the
menu", then the goal "train kim to achieve all behaviors 100% as others" = Rung 0, Rung 2, crawl, get-up, each at its
gate. Characteristics picked by Claude ("light and quick"): **1.55 m, 54 kg**, λ = 0.844, strength 0.9 of the size-scaled
torque (knee cap 144 Nm: the best strength-to-weight after MATT), servo stiffness 1.0, full self-collision, upright
athletic stance, quick short steps (air-time reward up to 0.45 s × √λ instead of 0.6). The scan has a long trunk on
short legs (hip joints at 0.44 of her height).

- [x] **K1 Asset** — `Kim.glb` (user; skinned, 21 bones, 1.062 m, one walk action) → `SourceArt/Kim/Kim_source.glb`;
  `SourceArt/Kim/convert_kim.py` (headless Blender): rig symmetrised (it was 13-20 mm off left / right), hip joints
  16.4 cm apart, legs posed vertical and baked as rest (feet hip-width: only the thigh pair touches at rest), Chest
  split into Spine1 + Spine2, Mixamo names, 1.550 m, soles on z = 0 → `SourceArt/Kim/kim.glb` (22 bones, 32k verts,
  0 unweighted, mirror error 0) + `convert_report.json`.
- [x] **K2 Body pipeline** — `bodies.BODIES["kim"]`; extract_skeleton A4 PASS; build_mjcf (pelvis fitted to the hip
  slab like GRANDMA; chest fitted to Spine2 + shoulder skin) → `kim.xml` / `scene_kim.xml` (54.0 kg, root z 0.723 m),
  fingerprint, `contract_kim.json`; `check_self_collision.py kim` PASS (118 of 135 pairs collide, 16 parent-child,
  thigh pair off = overlap at rest; no contact in the T-pose, the stance or the arm / leg swing).
- [x] **K3 Tasks** — `tasks/kim_env.py`: GRANDMA's recipes on her body (`grandma_env.G` = the process body for KIM),
  without the stoop / careful steps. Tasks `PoOlympic-Kim-Rung0-Stand`, `-Rung1-Run`, `-Rung2-Omni-Base` / `-Cap` /
  `-Bands`, `-Crawl2` / `-Crawl3` (hands-and-knees pose at 73°, pelvis 0.335 m), `-Getup` / `-Getup-Prone` / `-Getup-Roll`.
  C1 PASS on rung0; every task builds and steps (16 envs).
- [x] **K4 Rung 0** — k0_v1 (from scratch, 4096 envs, 1500 its, 53 min): gates 9/10 · **10/10 · 10/10** at it 500 / 1000 /
  1499, 0 falls; 30 other seeds 27/30 (GRANDMA 24, ZOMBIE 25, MATT 29) → `kim_rung0.onnx` = it 1499. Review
  `parity/tb/k0_v1/review.html`.
- [x] **K5 Rung 1 → Rung 2** — k1_v1 (warm start k0_v1, 2000 its): G1 rung 1 **10/10 at it 1000 / 1500 / 1999**. k2_v1
  (base recipe, 2000 its): 0 → 2 → 3 → 7/10, one turntable direction and the right-hand side-steps slow; k2_v2
  (`-Bands`: SymmetricRunner + sharp kernel + command bands, entropy 0.001, 1200 its): 5 → 8 → 8 → 9/10; k2_v3
  (`-Bands2`: 30 % fast side-steps, 1000 its): **10/10 at all five gates, 0 falls**; 30 other seeds 28 / **29** / 28 of 30 at
  it 200 / 600 / 999 (the miss: a 3.45 m/s sprint with a turn). Event heats (6 per event, her 2 lanes): finishes every
  race in Events 5, 9, 10, 11, 12, 13, 19, 22; it 600 has no fall and no DQ → `kim_rung2.onnx` = k2_v3 it 600. Reviews
  `parity/tb/k1_v1`, `parity/tb/k2_v3`; `tools/gate_fails.py` (which checks / segments fail).
- [x] **K6 Crawl** — kcrawl_v1 (`-Crawl2`, from Rung 0, 1500 its): flat until it ~600, on all fours from 900, 30 m 5/5 in
  32.4-33.3 s at it 1499 but 5 m off the lane; kcrawl_v2 (`-Crawl3`, 1000 its): **5/5 at every gate, 28.1-29.1 s, lane
  0.15 m at it 999** (MATT 25.7-26.7 s, GRANDMA 27.1-27.9 s, zombie 31.7 s) → `crawl_kim.onnx`. Review `parity/tb/kcrawl_v2`.
- [x] **K7 Get-up** — kgetup_v1 (ladder from Rung 0; stopped by hand at it ~1030, ladder complete): supine **10/10 at it
  1000**, up in 0.74-0.78 s; kgetup_prone_v1 (`-Getup-Prone`, 600 its, press-up recipe — no roll ladder needed): **supine
  10/10 + face down 10/10 at it 599; 30 seeds: 30/30 (0.70-0.82 s) and 30/30 (1.00-1.06 s), 0 falls after**; joint speeds
  0.8 % / 0.0 % of substeps above 18 rad/s (allowance 5 %) → `getup_kim.onnx` (no event uses a get-up brain yet). Films
  `parity/tb/kgetup_prone_v1/`.
- [x] **K8 Unity / menu** (2026-10-11) — `compose_mixed.ROSTER` + kim (prefix `K<k>_`): roster verify 46/46, crowd-contact
  audit 32/32 scenes (new lineup `mzgkmzgk`); `Testbed_Kim.unity`: 16 bones bound at 0.00 mm · **G0 0 mismatches · G2**
  ≤ 2.2e-16 · **G3** ≤ 6.6e-7 · **G4** (1 s) ≤ 1.7e-5 on `kim_rung0` / `kim_rung2` (G5 not run); critics + confidence for
  her three event brains (AUC 0.67 / 0.67 / 0.70); odds refitted with a KIM weight (40 heats per event with her, 6
  features); the 11 roster event scenes rebuilt (32 athletes each), re-dressed and baked; main menu rebuilt (fourth
  roster card, orange K chip, tighter card text). Played in the editor: Terminal Velocity (K8 third, 3.6 m/s), Iron
  Pedestal, 30m All Fours (K8 29.28 s, K1 29.40 s), Slalom; no errors besides one "Unknown error … LightingData.asset" per
  scene, present since the fresh clone. Before / after: `parity/ui/kim_menu_before_after.html`.
- **2026-10-11 midday (goal check: 30-seed margins + MATT's extra skills):** k0_v2 (`-Rung0-Tight`: stay-near-the-start
  reward x3, kernel x0.6; from k0_v1 it 1499, 500 its): 30 seeds 28 · **30** · 28 of 30 at it 200 / 400 / 499, 0 falls →
  `kim_rung0.onnx` = k0_v2 it 400 (critic + confidence refit, AUC 0.70; parity reference re-recorded, Unity gates not
  re-run). k2_v4 (`-Bands3`: 30 % top-speed sprint band; from k2_v3 it 600, 600 its): 28/30 at it 200 / 400 / 599 — no
  gain, `kim_rung2.onnx` stays k2_v3 it 600 (29/30). Written, smoke-tested and queued, not run yet: `-Rung2-Flight`
  (MATT's r2f_v3 terms, Froude-scaled) and `-RungS-Stance` (rs_v7 recipe on her Rung 2 base; `stance_env` now follows the
  process body). Queue stopped by the user to play in Unity.
- Open: G5 closed-loop gate for her brains; her joint speeds in the crawl; the phone build has not been made with her;
  her height / mass / strength are Claude's picks, to be confirmed by the user.