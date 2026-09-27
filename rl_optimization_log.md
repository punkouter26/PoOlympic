# RL Optimization Log

One entry per run or decision. Newest at the bottom.

<!-- Template
## YYYY-MM-DD · <run-id or DECISION>
- **Change:** what was changed (config diff, code, asset)
- **Hypothesis:** why
- **Result:** metrics (TensorBoard tags, eval_cpu report path, gate report path)
- **Decision:** keep / revert / next step
-->

## 2026-09-27 · DECISION A1 — MuJoCo version pin = 3.11.0
- **Change:** `mujoco==3.11.0`, `mujoco-warp==3.11.0`, Unity `org.mujoco` from git tag `3.11.0` + `mujoco-3.11.0-windows-x86_64.zip` → `bin/mujoco.dll`.
- **Hypothesis:** mjlab 1.6.0 (latest) pins `mujoco~=3.11.0`; the Unity plug-in exists at tag 3.11.0 with the matching Windows DLL. Latest MuJoCo is 3.14.0 but gains nothing we need for Rungs 0–2.
- **Decision:** one version everywhere. Revisit only when mjlab moves its pin.

## 2026-09-27 · DECISION A2 — Python env verified
- **Result:** Python 3.11, torch 2.14.0+cu130 (sm_120 OK), warp-lang 1.17.0 (arch 120), mujoco-warp 3.11.0.
  `mujoco_warp` reference humanoid (nq 28, nu 21): 1.08 M env-steps/s @ 1024 worlds, **3.23 M env-steps/s @ 4096 worlds** (CUDA graph).
- **Note:** `nconmax`/`njmax` are per-world in mujoco-warp 3.11 (`put_data(..., nconmax=64, njmax=256)`).

## 2026-09-27 · DECISION A3 — Trainer = mjlab 1.6.0 (native Windows, no WSL)
- **Result:** `Mjlab-Velocity-Flat-Unitree-G1`, 4096 envs, 30 iters: ~1.0 s/iter (~100 k policy-steps/s), vel-xy error 0.045→0.030, yaw error 0.107→0.072. TensorBoard logger works (`--agent.logger tensorboard`).
- **Decision:** build PoOlympic tasks on mjlab managers (G1 velocity task = template for Rungs 1–2).
- **Parity constraint:** athlete must use MuJoCo-native `<position>` actuators (mjlab builtin actuator), never a torch-side PD, so Unity's `mj_step` computes identical forces.

## 2026-09-27 · FINDINGS — Unity plug-in 3.11.0 constraints (read from source)
- `MjScene.FixedUpdate` = `preUpdateEvent` → `mj_step` → `postUpdateEvent`. **PolicyRunner subscribes to `preUpdateEvent`** and writes ctrl every 4th tick → exact tick order without script-execution-order hacks. Do NOT use `ctrlCallback` (switches to `mj_step1/2`).
- Timestep comes from Unity `Time.fixedDeltaTime`; gravity from Unity `Physics.gravity` (must be (0, −9.81, 0)).
- `MjOptionStruct` exposes integrator/cone/jacobian/solver/iterations/tolerance but **not `ls_iterations`/`ls_tolerance`** → MJCF must leave them at MuJoCo defaults (50 / 0.01). DESIGN §2 amended (was ls_iterations 10). Training may lower it in Warp for speed only if G1 still passes.
- Components store values as C# `float` and regenerate MJCF text → generator writes every float with ≤ 7 significant digits so the round-trip is lossless; G0 verifies.
- Supported: position actuator `kp`/`kv`/`forcerange`/`ctrlrange`, joint `armature`/`stiffness`/`damping`/`springref`/`range`, geom `contype`/`conaffinity`/`condim`/`friction`/`solref`/`solimp`/`priority`, explicit `<inertial>` (mass + diaginertia), `<exclude>`, body `gravcomp`.

## 2026-09-27 · A4–A8 — MATT physics body derived
- **Change:** `tools/extract_skeleton.py` (52 joints, bind-consistency 1.3e-7, L/R mirror 0.13 mm, body height 1.837 m, 1.857 m w/ hair) → `tools/build_mjcf.py` → `assets/matt.xml` (robot) + `assets/scene_matt.xml` (ground + 4-cube pool + `default` keyframe).
- **Result:** robot nq 32 / nv 31 / nu 23, 16 bodies, 80.000 kg. Capsules fitted from skin-weighted vertices (left side, mirrored). Uniform-density geoms would weigh 100.4 kg → inertias rescaled to de Leva masses. Default standing root z = 0.9549 m. `tests/test_model.py` 10/10 green (incl. anatomical sign semantics for every joint, PD hold settles < 0.8 s, penetration < 5 mm, only feet touch ground). Overlay renders in `parity/a8_*.png`.
- **Fixes vs DESIGN:** torso mass is 13.06 kg (Spine + Spine1 segments; table said 6.53 — total 80 kg was already correct). Bone offsets in Unity are constant-per-bone, not identity (Mixamo bones carry bind rotations).
- **Decisions:** thigh_l↔thigh_r contact excluded (capsules overlap 6 mm at rest); shin/foot/toe still collide across legs. Collision bits: upper body `contype=1<<lane, conaffinity=0`, legs `1<<(lane+8)` both, ground/cubes `0xFFFF` → no self-collision except leg↔leg, no lane↔lane. Actuators have no ctrlrange (clip in ONNX). Contact friction is max(geom pair) → cube friction 0.8 is effectively 1.0 against athlete/ground (accepted).

## 2026-09-27 · A9–A12 — Contract, fingerprint, test brains, golden references
- **Change:** `poolympic/contract.py` (84-dim obs, 23 actions, phase clock 1.8 Hz frozen when |cmd| < 0.1) → `parity/contract.json`; `poolympic/fingerprint.py` → `parity/fingerprint_python.json` (sha256 in `.sha256`, embedded in every ONNX + contract); `poolympic/policy_export.py` (opset 17, batch 1, outputs `ctrl` + `action_raw`, obs normalisation + clip baked in); `tools/make_test_brains.py` → `parity/brains/{zero,random}_brain.onnx`; `tools/record_reference.py` → `parity/reference_trajectory_{zero,random}.json`.
- **Result:** ORT vs torch ≤ 2.2e-7; graph ctrl vs float64 contract mapping ≤ 6.3e-8. References are bit-identical on rerun. `pytest` 16/16.
- **Observation:** the PD-held statue (no policy) is NOT statically stable: COM is 3 cm ahead of the ankles (inside support [-0.09, +0.19] m) but the knees sag 20°→28° under load and it tips backward at 2.4 s undisturbed; with the parity script (0.5 m/s shove at 1.0 s) it falls at 2.12 s. Expected for fixed PD gains; it's a strong, deterministic dynamic signal for G4/B11. Rung 0 policy must handle it.
- **Decision:** G0 is a tolerant field-by-field compare (Unity floats are float32) — the ONNX↔fingerprint link is the exact SHA-256 of `fingerprint_python.json`, and Unity must pass G0 against that file before it runs any brain.

## 2026-09-27 · A8 — User sign-off
- **Result:** overlay renders (`parity/a8_*.png`) approved by user. Phase A complete.

## 2026-09-27 · B1–B2 — Unity packages + PhysX isolation
- **Change:** `Assets/Plugins/MuJoCo/mujoco.dll` (3.11.0, release zip sha256 a9cf1ec4…); `org.mujoco` 3.11.0 **embedded** in `Packages/org.mujoco` with one patch (`GetInstanceID` → `GetEntityId`, obsolete-as-error on Unity 6000.6; see `POOLYMPIC_PATCHES.md`); `com.unity.ai.inference` 2.6.1; `com.unity.cloud.gltfast` 6.20.0. `mj_version()` in-editor = 3011000.
- **Editor tooling decision:** in-editor authoring uses the **Unity CLI** (`unity command …`, via the project's `com.unity.pipeline` package) — the official Unity plug-in the brief asked for. The separate MCP-for-Unity bridge (UnityMCP, 127.0.0.1:8080) was never installed in this project and is not needed.
- **PhysX isolation:** `Physics.simulationMode = Script`, `autoSyncTransforms = false`; `PhysXGuard` (menu, Play-mode entry, build preprocess) rejects Rigidbody/Collider/CharacterController/Joint/ArticulationBody/2D equivalents in `Assets/PoOlympic/Scenes`. EditMode tests 3/3.
- **Parity finding:** Unity 6 stores Fixed Timestep as a rational (count / 141 120 000). Setting 0.005 through the float API gave 705 599 ticks = 0.004999993 s, which the MuJoCo plug-in copies into `opt.timestep` (would fail G0 and perturb dynamics). Fixed by writing `m_Count = 705600` → `Time.fixedDeltaTime == 0.005f`, MJCF text "0.005". Runtime will additionally assert `opt.timestep == contract.timestep` in `MjScene.postInitEvent`.

## 2026-09-27 · B3–B4 — MJCF import + GATE G0 PASS
- **Change:** `AthleteImport` (menu *PoOlympic › Import Athlete Scene*) copies `training/assets/scene_matt.xml` → `Assets/PoOlympic/Models/` (SHA-256 checked, stored in `ModelProvenance`) and imports it with the plug-in importer into `Scenes/Testbed_ZeroBrain.unity` (20 bodies, 22 geoms, 25 hinges, 5 free joints, 23 actuators). `ModelFingerprint.cs` dumps Unity's compiled mjModel in the Python schema; `ParityTools` (menu *PoOlympic › Parity › G0*) writes `parity/fingerprint_unity.json`.
- **Result:** **G0 PASS — 0 mismatches**, worst float difference 1.4e-7 (float32 floor).
- **Root causes found and fixed on the way (each would have silently perturbed physics):**
  1. Plug-in `MultiCCD` default = disable, MuJoCo default = enable → MJCF now pins `<flag multiccd="disable"/>`.
  2. `MjImporterWithAssets.ImportFile` round-trips through `mj_saveLastXML` (6 significant digits, radians → float degrees): mass 13.05674→13.0567, ranges off 2e-6 rad → import with `ImportString` on the original text.
  3. Capsule `fromto` rebuilt in float32 `FromToRotation` → 3.2e-6 rad axis error on near-vertical thighs → generator emits capsules as `pos`/`quat`/`size(r, half)` computed once in float64.
  4. Plug-in suffixes every MuJoCo name with `_<n>`; the C# dump strips it (generated names never end in `_<digits>`).
  5. Edit-mode `MjScene` singleton must not be saved in the scene (it throws when found without Awake) → parity tool uses a temporary instance.
- **Compare semantics:** quaternions sign-canonicalised; capsule/cylinder frames compared as unsigned axis (symmetric shapes); sphere orientation ignored.
- Model sha256 (scene_matt.xml) changed with fixes 1 and 3 → brains, contract and references regenerated (outcomes unchanged: zero-brain falls at 2.12 s).

## 2026-09-27 · B8–B10 — Runtime policy loop + GATES G2 / G3 / G4 / G5 PASS (zero + random brains)
- **Change:** runtime `Contract`, `AthleteBinding` (name-resolved indices), `ObservationBuilder` (line-for-line port of `contract.build_obs`), `PolicyBrain` (Inference Engine CPU, batch 1), `Disturbance`, `PolicyRunner` (hooked on `MjScene.postInitEvent` / `preUpdateEvent`, records `parity/unity_run_<name>.json`). Editor `ParityHarness` + EditMode `ParityGateTests` (G2–G4); `tools/compare_closed_loop.py` (G5). References now carry the Python joint address table; contract carries the default state keyed by joint name; brains ship a `.onnx.json` sidecar (Inference Engine exposes no ONNX metadata) that PolicyRunner validates against the contract fingerprint.
- **Result (edit mode, Unity-compiled model + native lib):**
  | ref | G2 obs | G3 action / ctrl | G4 qpos 1 s | G4 qpos 5 s |
  |---|---|---|---|---|
  | zero | 4.4e-16 | 0 / 0 | 3.9e-7 | 3.3e-3 (post-fall chaos) |
  | random | 4.4e-16 | 1.6e-7 / 1.2e-7 | 4.0e-7 | 8.2e-4 |
- **Result (Play mode, real MjScene.FixedUpdate loop, G5):** zero — fall 2.12 s vs 2.12 s, pelvis-height RMS 1.8e-5 m, torque ratio 0.9997, 1 s drift 3.9e-7 (identical to G4 → the play loop adds no error). random — PASS, action diff 7.6e-6, 5 s drift 8.1e-4. Reports: `parity/gate_report_*.json`, `parity/g5_report_*.json`.
- **Critical parity findings:**
  1. `MjActuator.OnSyncState` writes its float32 `Control` into `mjData.ctrl` after every `mj_step` → PolicyRunner re-asserts the cached float64 ctrl before **every** substep (not only on control ticks). Without this, substeps 2–4 would run float32-truncated/stale ctrl.
  2. The Editor does not tick the Play-mode player loop while unfocused (Windows blocks focus stealing) → unattended parity runs use `ParityTools.ArmDeterministicStepping` (pause + `EditorApplication.Step()` from `EditorApplication.update`, `Time.captureDeltaTime = 0.02`). Physics remains one `mj_step` per FixedUpdate; ticks = FixedUpdates / 4 verified.
- **Decision (B8):** timing verified by construction + G5 (tick = fixed steps / 4, disturbance ticks land identically).

## 2026-09-27 · B5–B7, B11 — Visual binding, testbed, cube pool; PHASE B COMPLETE
- **B5:** MATT.glb imported with glTFast (faces Unity +Z, left −X); plug-in places MuJoCo +X/+Y on Unity +X/+Z → visual root yaw +90°. `BoneBinder` drives the 16 pivot bones from their MjBody (offsets captured at qpos 0); merged bones follow rigidly. Bind error 0.001 mm / 0.000°.
- **B6:** `Testbed_ZeroBrain.unity` authored in-editor via Unity CLI: `BroadcastCamera` (9:16 letterbox, side-on pelvis tracking), directional light, `TrackVisual` (render-only plane, collider stripped — physics ground is the MuJoCo plane), `TestbedHud` (FPS, sim time, tick, pelvis height, lane state; Reset / Shove / Drop cube).
- **B7:** `MjCubePool` (pooled `cube<i>_free` teleported via qpos/qvel at the next control tick through `PolicyRunner.Request`), `Shove` (Δqvel on the root), `RequestReset`. Tests: `FiredPoolCubeHitsAthleteThroughMuJoCo` (8 m/s cube → contact with athlete geom, pelvis pushed back), `NoInstantiateOrDestroyInRuntimeCode`, `NoPhysXApiInRuntimeCode`. EditMode suite 8/8.
- **B11:** zero-brain Play-mode run reproduces the Python reference (G5 PASS, fall at 2.12 s both); captures `parity/b11/tick_{25,60,110,160}.png` + `b11_sheet.png`.
- **Phase B verdict:** G0, G2, G3, G4, G5 all PASS in Unity for zero and random brains. The inference→control pipeline, joint indexing, axis directions, rest pose, masses and limits match the training model. Cleared to start Phase C.

## 2026-09-27 · C1–C2 — mjlab task + CPU evaluator
- **Change:** `poolympic/tasks` (entry point `mjlab.tasks` → `uv run train PoOlympic-Matt-Rung0-Stand`). Robot = `matt.xml` with its own `<position>` actuators via `XmlActuatorCfg` (delay 0–4 substeps, redrawn per episode — "latency" trait); 4 pooled cube entities (`assets/cube.xml`); ground = mjlab plane with the scene's exact contact bits (0xFFFF) + friction via `CollisionCfg`. Observations = 9 custom terms reproducing `contract.build_obs` in contract order; actions `preserve_order=True` in contract actuator order with per-joint `clip` = joint range (identical to the ONNX graph). `AthleteCommand` = mjlab velocity command + contract phase clock (advanced once per step in `compute`). Rung 0: zero command; rewards upright(torso) / height / still lin+ang / near-origin / posture / torso ang-vel / action-rate / joint limits / torques / joint-vel>18 / terminated −200; terminations pelvis<0.55, torso tilt>60°, non-foot–ground contact sensor; events shove Δv ±0.6 m/s every 3–5 s, cube drop from 1.5 m above shoulder every 3–5 s; DR mass ×[0.85,1.15], foot friction ×[0.8,1.2], kp/kv ×[0.85,1.15], strength (forcerange) ×[0.85,1.15]. Restitution DR not applied (not needed for rigid contact; revisit if G5 needs it). Obs noise on actor only.
- **Checks (`tools/check_task.py`):** compiled options identical to scene_matt.xml (timestep, iterations 20, ls 50, implicitfast, pyramidal, Newton, multiccd disabled, tolerances); ground/athlete/cube contact bits + friction identical; **training obs vs contract.build_obs max |Δ| = 2.4e-7** over 16 envs × 30 random steps.
- **C2:** `poolympic/evaluate.py` + `tools/eval_cpu.py` (G1): CPU MuJoCo, contract tick order, Rung 0 protocol (20 s, independent 3–5 s shove and cube-drop schedules, fall/recovery/foot-box/joint-velocity metrics, 10 seeds). Zero brain: 0/10 (tips over at 2.34 s) — expected.
- **Smoke train:** 4096 envs, ~2.0 s/iteration (~50 k policy-steps/s), episode length rising 75 → 110 ticks in 25 iterations.

## 2026-09-27 · r0_v1 — ABORTED at iter ~260: action wiring scrambled in training (G1 caught it)
- **Symptom:** training episode length 955/1000 ticks by iter 256, but the exported it-200 brain fell on 10/10 CPU seeds at 4–9 s (slow sink, `pelvis_low`).
- **Root cause:** mjlab's `XmlActuator` pairs per-joint targets (joint-tree order) with ctrl slots in actuator *declaration* order. `matt.xml` declared actuators limb-grouped (abdomen, L arm, L leg, R arm, R leg) ≠ tree order (abdomen, L arm, R arm, L leg, R leg) → 10/23 channels cross-wired in training (e.g. action "hip_flex_l" drove shoulder_elev_r). Observations were verified (2.4e-7) but the action→ctrl path was only printed, not asserted.
- **Fix:** `build_mjcf.py` declares actuators in joint-tree (DFS) order; `test_model.py::test_actuators_declared_in_joint_order`; `check_task.py` now asserts per-channel wiring (action[i] moves exactly contract actuator i's ctrl) → 0 mismatches. Contract actuator order changed accordingly (brains/refs/contract regenerated; Unity re-sync + G0/G2–G4 rerun at C4).
- **Lesson:** every link of the obs→policy→ctrl chain gets an assertion, not a print.

## 2026-09-27 · r0_v2 — Rung 0 GATE G1 PASS (it 1000) + C4 Unity spot-check PASS
- **Run:** `runs/matt_rung0/2026-09-27_15-26-01_r0_v2` (4096 envs, fixed wiring). Training ep-length 981/1000 @ it 256, 1000/1000 from it 500.
- **G1 (CPU MuJoCo, seeds 1000–1009):** it 200: 5/10 (1 fall, 4 pedestal drift) · it 500: 9/10 (drift 0.503 m) · **it 1000: 10/10 PASS** — 0 falls, all hits recovered (worst 1.04 s, peak tilt 10.5°, typically 2–5°), max foot excursion 0.48 m (bar 0.5), joint-vel violations 0. Margin set (30 extra seeds): 29/30, 0 falls; the one miss is pedestal drift 0.575 m.
- **Evaluator fixes:** recovery time now = hit → last tick ≥ 10° before the next hit (was: first 0.5 s window, which could hide a late spike); peak tilt reported.
- **Brain:** `parity/brains/r0_v2_it1000.onnx` (export verified vs rsl_rl math 7.8e-7).
- **C4 (Unity, new actuator order):** re-import → **G0 PASS (0 mismatches)**; EditMode 8/8; rung0 reference: **G2 2.2e-16, G3 5.8e-7 / 1.5e-7, G4 1 s 4.0e-7, 5 s 8.7e-4**; Play-mode **G5 PASS**: stands through shove + cube, 5 s qpos drift 1.2e-5, torque ratio 1.0000001. Captures `parity/c4/`.
- **Follow-ups:** pedestal-drift margin is thin (0.48 m worst of the official seeds) — evaluate later checkpoints of r0_v2 (training continues to 3000); if drift persists, strengthen the stay-on-pedestal term in a fine-tune. Cube renderers got a URP material (plug-in default rendered magenta).

## 2026-09-27 · r0_v2 stopped at it ~1210 · Rung 0 brain = `r0_v2_it1000`
- it 1200 also G1 10/10 (0 falls in 40 seeds) but pedestal-drift margin not improving (worst 0.57 m on the 30-seed margin set; bar 0.5 m). Cause is reward design (weak `near_origin`, pelvis-based), not training time → stop and free the GPU.
- **Follow-up (Iron Pedestal event):** fine-tune with a foot-based pedestal term (+ termination when a foot leaves the 1 m box) before Phase D.
- Contract v2: stride frequency now speed-dependent `gait_hz = 0.8 + 0.2·(|v_xy| + 0.5|wz|)` (was fixed 1.8 Hz — too fast for walking). C# `Contract.GaitHz` mirrors it. Rung 0 brains unaffected (command 0 ⇒ phase ≡ 0) but must be re-exported under contract v2.

## 2026-09-27 · r1_v1 — Rung 1 started (warm start from r0_v2 it 1000)
- **Task `PoOlympic-Matt-Rung1-Run`:** heading-hold commands (stiffness 0.5, |wz| ≤ 0.5, random heading targets), vx curriculum (0.5–2.0) → (0.5–3.0) @ it 800 → (0.5–4.0) @ it 1800, 10 % standing envs; shoves ±0.5 m/s every 3–5 s, cube drops every 5–8 s; rewards track_lin (2.0, std 0.5), track_ang (1.0), upright, height (std 0.15), variable posture (0.2/0.5/0.8), phase↔stance contact (0.5), feet air time (0.5, 0.1–0.6 s), foot slip (−0.1) + Rung 0 penalties. `feet_ground` subtree contact sensor (foot + toe) with air-time tracking.
- **Checks:** `check_task.py rung1` — obs vs contract 3.5e-7, phase clock vs `contract.advance_phase` 2.1e-7 with moving commands, wiring 0 mismatches.
- **G1 evaluator (rung 1):** 30 m dash at U(0.5, 3.0) m/s, heading hold with mjlab's heading definition (yaw of pelvis x-axis) and the same P-controller, 0.3 m/s shoves every 3–5 s; bars: finish, no fall, forward-velocity RMS error < 0.15 m/s (after 2 s warm-up), lateral drift < 0.5 m, joint-vel violations ≤ 5 %.
- **Warm start:** checkpoint copied to `runs/matt_rung1/r0_init/model_0.pt` with `iter = 0` and env_state cleared so the command curriculum starts at stage 0.

## 2026-09-27 · r1_v1 — stopped at it ~300: stand-still local optimum
- it 200 dash eval: MATT stands in place (distance ≈ 0, velocity RMS error = commanded speed) with full-length episodes. Warm start from the standing policy + shrunken action std + tracking reward with no gradient at 1.5–2 m/s errors.
- **r1_v2:** train from scratch; command curriculum (0.3–1.0) → (0.5–2.0) @ it 600 → (0.5–3.0) @ it 1200 → (0.5–4.0) @ it 2000.

## 2026-09-27 · r1_v1 — FAILED (standing collapse) · r1_v2 prepared
- **Symptom:** by it 150 episodes hit the 1000-tick cap with zero falls while per-step velocity error ≈ 1.15 m/s (mjlab's `error_vel_xy` metric sums over the episode ÷ max resample steps → 2.3 ≈ 2× the per-step error) — i.e. ≈ the mean command. G1 on it 200 (`r1_v1_it200`, 4 seeds): distance travelled −0.3…0.1 m, 0/4.
- **Root causes:**
  1. *Warm-start normalizer:* the Rung 0 obs normalizer had var = 0 on command and gait-phase dims (constant in Rung 0) and count ≈ 1e8. At the start of Rung 1 the network saw command ×150 and phase ×100; with that count the running stats adapt only slowly (it 200: vx mean 0.18 vs true ≈ 1.1). Pelvis lin-vel std was also standing-scale (0.17 m/s → running speeds normalized ≈ 12).
  2. *Narrow tracking kernel + low exploration:* track_lin std 0.5 gives exp(−6) at a 1.25 m/s error — no gradient out of the standing basin; warm-start action std was 0.10–0.29.
- **Fix (r1_v2):** train **from scratch** (fresh normalizer, init std 0.8 — as Rung 0); add `track_lin_coarse` (w 1.0, std 1.0); curriculum starts at walking speeds vx 0.3–1.0 → 0.5–2.0 @ it 600 → 3.0 @ 1200 → 4.0 @ 2000. `check_task.py rung1` PASS.
- **Lesson:** never warm-start across a change in observation distribution without resetting (or re-seeding) the normalizer; watch the tracking error per step, not just episode length.

## 2026-09-27 · C6 infrastructure — 8-lane meet, G0 meet8 PASS, G6 harness dry run PASS
- **Change:** `build_mjcf.compose_meet` → `assets/scene_meet8.xml` + `meet8_layout.json`: 8 copies of the training athlete, every name prefixed `L<k>_`, lane bits (upper body `1<<k` / legs `1<<(k+8)`), pelvis at y = (3.5 − k)·1.22 m, 16 pool cubes (2 scripted slots per lane). Solo `matt.xml`/`scene_matt.xml` byte-identical. `poolympic/meet.py` (lane mapping root→`L<k>_root`, cube i→slot, xyz += origin; per-lane reset; CPU meet rollout recording in the Unity run format); `poolympic/closed_loop.py` (G5 compare, lane-aware; athlete drift separated from pooled-cube drift, which only counts after the cube is fired).
- **Unity:** `PolicyRunner` is lane-safe — `ResetToDefault` touches only its own athlete (joints under its pelvis; zero qvel + warm start) instead of `mj_resetData` on the whole scene (with 8 runners each lane's reset would have wiped the previous lanes); lane origin (double) + cube-slot remap for scripted disturbances (`Disturbance.InLane`, mirror of `meet.Lane.disturbance`); records only its own joints + cube slots with lane meta; `RequestReset` also parks its cube slots (`park` disturbance). `BoneBinder.Capture(prefix)`, `AthleteImport` parametrised by source MJCF, `MjCubePool` per-lane overloads, HUD shows all lanes. `MeetTestbed` builds `Testbed_Rung1.unity` and loads a G6 plan.
- **Results:** CPU meet vs solo: 1 s drift ≈ 1e-15 on all lanes (lane isolation exact up to rounding). **G0 meet8 PASS** 8/8 lanes, 0 mismatches; each lane == solo athlete except bits/origin. **G6 dry run** (plan `g6dev`, interim brain `r1_v1_it200`, lanes vx 0…3 + one yawing lane, staggered shove + cube drop per lane): PASS 8/8, 1 s drift 3.7e-7…1.1e-6 (float32 floor, same as solo). Solo G5 re-verified after the refactor: zero (fall 2.12 s both, drift 3.9e-7 — unchanged), random, rung0 PASS.
- **Also:** r0_v2_it1000 / zero / random brains re-exported under contract v2 (sidecars were v1 → the runner would refuse them); references regenerated, frame-for-frame identical (Rung 0: command 0 ⇒ phase ≡ 0). The previously committed `unity_run_{zero,random}_play` predated the actuator reorder — re-recorded. `MUJOCO_LOG.TXT` warnings "NaN QPOS at DOF 0" come from the MuJoCo plug-in's own EditMode tests (8 of its 723 tests fail on Unity 6 — upstream, not our code); log files now gitignored.
- **r1_v2 launched** 16:31 (`runs/matt_rung1/2026-09-27_16-31-44_r1_v2`, from scratch). It 350: episode length 983/1000, track_lin 1.72 / 2.0 (r1_v1: 0.25), per-step velocity error ≈ 0.14 m/s at vx 0.3–1.0, pelvis_low terminations 0.17/episode. Curriculum → 2.0 m/s @ 600, 3.0 @ 1200, 4.0 @ 2000; first G1 check planned ≥ it 1500.

## 2026-09-27 · r1_v2 it 1500 — G1 FAIL on lane drift only; cause = heading bias + weak heading controller
- **G1 (official 10 seeds, evaluator as specified: heading hold k = 0.5):** 10/10 finish, 0 falls, vel RMS 0.047–0.089 m/s (bar 0.15), joint-vel 0 — but lateral drift 1.0–3.3 m (bar 0.5) → 0/10.
- **Diagnosis:** not shoves (no-shove runs still drift ~2 m at 0.7–1.0 m/s; 0.26 m at 1.8 m/s). Decomposition at 0.7 / 1.0 m/s: mean heading −5.5° (→ −2.7 m over 30 m); sideways velocity in the heading frame ≈ 0.01 m/s (→ < 0.7 m). The policy walks straight along its pelvis heading, but with k = 0.5 a 5.5° error commands only 0.05 rad/s, which the policy doesn't respond to (track_ang kernel std √0.5 is almost flat for small yaw-rate errors). Heading-only hold also can't undo lateral displacement from shoves.
- **Deploy-side controller experiment (same brain, no retraining):** heading gain 2.0 → 4/10 (max 0.81 m); heading gain 2.0 + lane keeping (heading target = atan(−0.3·y_lane)) → **10/10 official seeds (max drift 0.26 m)**, margin 30 seeds 26/30, 0 falls; misses all at 2.7–3.0 m/s, marginal (drift 0.55 m or vel RMS 0.151). |wz| stays within the trained ±0.5.
- **Open decision (user):** adopt the lane-keeping steering controller as the standard (evaluator + Unity runner) and re-gate on a later checkpoint (it ≥ 2000), vs. fine-tune with a tighter track_ang kernel to keep the original heading-only protocol.

## 2026-09-27 · Lane-keeping steering adopted · r1_v2 it 2000 = Rung 1 brain (GATE G1 PASS)
- **Decision (user):** race steering = lane keeping, shared by evaluator and Unity: `contract.steer_yaw_rate` — heading target atan(−0.3·lane offset), wz = clip(2.0·heading error, ±0.5). Constants exported in `contract.json["steering"]`; C# `Contract.SteerYawRate` + `PolicyRunner.laneKeeping` (off for parity runs, which keep fixed commands); EditMode `SteeringTests` vs Python vectors.
- **G1 (lane keeping):** it 1500 10/10, margin 26/30 (misses at 2.7–3.0 m/s) · **it 2000 10/10, margin 30/30**, 0 falls, lateral ≤ 0.42 m, vel RMS ≤ 0.121 · it 2500 8/10 · it 2999 6/10 — the late misses are all lateral drift 0.62–0.68 m on the slowest (0.5–0.7 m/s, 43–58 s) dashes; speed tracking stays ≤ 0.07. Regression starts once the curriculum widens to 4 m/s (it 2000). Selected **r1_v2_it2000 → `parity/brains/rung1.onnx`**.
- **G6 prep:** `make_g6.py rung1`: all lanes run (x after 5 s: 2.1 m @ 0.5 … 12.2 m @ 3.0 m/s, no falls); CPU meet vs solo PASS, 5 s drift ≤ 2e-7.
- **Evaluator fix:** rung-1 verdict produced numpy bools (JSON crash once a dash finished).

## 2026-09-27 · C6 — Rung 1 Early Verification Gate PASSED (HALT POINT cleared)
- **Brain:** `rung1.onnx` (r1_v2 it 2000). **Unity EditMode 18/18** (G2–G4 on 8 lane references, steering C#==Python, PhysX guard, runtime rules).
- **G2** ≤ 4.4e-16 · **G3** action ≤ 2.4e-6, ctrl ≤ 5.4e-7 · **G4** 1 s ≤ 1.8e-6 (5 s open-loop replay of a runner diverges, 0.02–0.99 m — expected, not gated) · **G0 meet8** 8/8 (model unchanged since).
- **G6 (Testbed_Rung1, 8 lanes running 0–3 m/s + yaw lane, staggered shoves + cube drops, 250 ticks):** every lane vs its solo CPU-MuJoCo run PASS — drift 1 s 5.4e-7…2.0e-6, 5 s 5.5e-7…1.6e-5, height RMS ≤ 7.8e-8 m, torque ratio 1.0000, no falls either side. Unity's multi-athlete scene reproduces training physics to float32 precision while running.
- **Next:** C7 Rung 2 (omnidirectional + yaw). Before Phase D: Iron Pedestal fine-tune (Rung 0 foot-based pedestal term); consider a Rung 1 fine-tune on slow-speed lane drift if later rungs regress it.

## 2026-09-27 · r2_v1 — Rung 2 started (warm start from rung1 = r1_v2 it 2000, normalizer re-seeded)
- **Task `PoOlympic-Matt-Rung2-Omni`:** direct (vx, vy, wz) commands, 15 % zero-command stops, resample every 3–8 s (brakes / reversals); curriculum vx (−1,3) vy ±0.5 wz ±1 → @500 vx (−1.5,3.5) vy ±1 wz ±2 → @1000 vx (−1.5,4) wz ±2.5 (spec says ωz ±2, but the 360° < 3 s bar needs ≥ 2.1 rad/s — trained to 2.5, turntable test at 2.2). Rewards = Rung 1 + track_ang w 1.5 std 0.5 (Rung 1's std √0.5 was too flat → heading bias) + track_ang_coarse w 0.5 std 1.5. 2000 iterations. `check_task.py rung2` PASS (obs 3.0e-7, phase 2.0e-7, wiring 0).
- **Warm start (`tools/warm_start.py`):** command dims of actor + critic normalizers re-seeded to the stage-0 command distribution (Rung 1 had vy var = 0 → ×100 input), count 1e6 (was 2e8), action std floored at 0.3, iter 0, env_state dropped.
- **Steering fix:** `steer_yaw_rate` takes the vx command sign — running backwards the lane-correction heading flips (otherwise it steers away from the lane). C# + SteeringTests updated.
- **G1 rung 2 (`eval_cpu.py --rung 2`):** per seed: tracking over the event envelope (5 × 5 s: sprint/back vx −1.5…4 |wz| ≤ 0.5, crab vy ±1, turn wz ±2, stop; shoves 0.3 m/s; RMS lin < 0.2 m/s, yaw < 0.3 rad/s after 1.5 s), turntable 360° < 3 s at wz 2.2 with drift < 0.3 m, brake from 3 m/s < 2 m, 20 m backward at −1.5 m/s, no falls, joint-vel ≤ 5 %. Baseline rung1 brain: 0/2 (falls on crab / turntable).
