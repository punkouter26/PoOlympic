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

## 2026-09-27 · D1 Iron Pedestal slice · r2_v1 G1 at it 1500
- **Pedestal physics:** `scene_pedestal.xml` = solo scene with the ground plane at z = −0.5 and a static 1 × 1 × 0.5 m box whose top is z = 0 (so default pose, pelvis-height obs and fall rule are unchanged; stepping off = real drop). Rung 0 brain (`r0_v2_it1000`) on it, Rung 0 protocol, 40 seeds: 40/40 survive, max foot-corner excursion 0.60 m (≤ 0.10 m overhang past the 0.5 m edge, still supported).
- **Unity:** `IronPedestalEvent` + `EventHud` + `EventScenes.BuildIronPedestal` → `Event_IronPedestal.unity`; deterministic play-through: SURVIVED 20.00 s (4 gusts, 5 cubes). EditMode 18/18 (PhysX guard covers the new scene).
- **r2_v1 it 1500 G1 (rung 2): 0/10.** No falls; brake 1.12 m ✓; backward 20 m ✓; linear tracking ✓ except crab on 2 seeds (0.21/0.26). Fails: yaw tracking (0.2–0.4 while running, ≤ 1.0 on fast turns; bar 0.3) and turntable 4.0–5.3 s (bar 3 s). Yaw reward fell after the ±2 / ±2.5 curriculum stages (track_ang 0.46 → 0.16) — reward/clock issue, r2_v2 needed (see proposal).

## 2026-09-27 · r2_v1 finished — G1 FAIL (yaw); best yaw checkpoint = it 700
- G1 rung 2: it 1500 0/10, it 1999 0/10 (no falls; brake / backward / linear tracking pass; yaw tracking + 360° turntable fail).
- Yaw RMS error probe (5 s holds; cmds wz ±1 in place, 2.2 in place, 1.0 m/s + 1.5, 2.0 m/s + 0.3, crab 1.0): it 300 0.45 · 500 0.42 · **700 0.29** (2.2 rad/s spin err 0.44 ≈ 1.8 rad/s achieved) · 1000 0.40 · 1200 0.46 · 1500 0.47 · 1999 0.50. Yaw tracking peaks right after wz ±2 enters (it 500) and then decays → reward balance, not training time.
- **Proposal r2_v2 (awaiting user OK):** warm start from r2_v1 it 700; track_ang w 2.0 (std 0.5) + track_ang_coarse w 1.0 (std 1.0); posture relaxed while |wz| > 0.5; curriculum keeps wz ±2 → ±2.5 later. Fallback if spins stay slow: gait clock weights |wz| fully (contract v3).

## 2026-09-27 · D4 — Iron Pedestal 8-runner heat (last one standing)
- **Physics:** `scene_pedestal8.xml` from `compose_meet(origins=venue_lane_origins(1, 3), pedestal_h=0.5)` — lane origins read from the stadium layout (venues.json), so the 8 physics pedestals coincide with the stadium's. G0 pedestal8 8/8 PASS (meet8 still PASS).
- **Traits** (contract.json `trait_ranges`, `obs_noise`): strength scales the lane's actuator force limits, latency delays ctrl by 0–4 substeps, sensor noise scales the training obs noise. Python (`events/iron_pedestal.py`) and C# (`PolicyRunner.SetTraits`) implement the same semantics; nominal traits leave every parity gate unchanged (EditMode 18/18).
- **Tuning (CPU, rung-0 brain):** escalation 0.5 + 0.1/round, cube every 2nd → heats 13–17 s (too short); nominal-trait sanity run at a constant 0.5 m/s still loses 2–3/8 runners in 25 s (3 s rounds with cubes are harsher than G1's 3–5 s). Chosen: 0.3 + 0.05/round, cube every 3rd → heats 27–41 s (mean 34.8), first elimination 17–26 s. Over 6 heats, place correlates with latency (+0.29) and strength (+0.25, noisy), not noise (−0.06): outcomes stay uncertain (good for betting) but traits matter.
- **Unity heat:** SURVIVED by L1 after 31.6 s / 11 rounds, first out 19.9 s — same shape as the CPU runs.

## 2026-09-27 23:08 · r2_v2 started (user: "spend next 8 hours training as needed")
- Warm start: r2_v1 **it 700** (best yaw), `warm_start.py` (command stats re-seeded to stage 0, count 1e6, action std floor 0.25, iter 0).
- Rewards vs r2_v1: track_ang w 1.5 → **2.0** (std 0.5); track_ang_coarse w 0.5 → **1.0**, std 1.5 → **1.0**; posture 0.5 → **0.25**. Curriculum unchanged (wz ±1 → ±2 @500 → ±2.5 @1000). 2000 iterations. `check_task.py rung2` PASS.
- Unattended gate `tools/watch_eval.py` → `parity/watch_r2_v2.jsonl`: every 250 its export + G1 rung 2 (10 seeds) + yaw probe.
- C8 prep: `make_g6.py <brain> <name> rung2` lane commands (stand, back −1.5, crab ±0.75, spin ±1.5, sprint 3.5, walk-turn 1.0/1.0); Testbed_Rung1 HUD command presets (Stop / Walk / Sprint / Back / Crab ◀▶ / Spin, lane keeping on for the straight-line ones).

## 2026-09-28 00:10–00:40 · r2_v2 result · symmetry augmentation · pedestal fine-tune running
- **r2_v2 finished (2000 its, 1.47 s/it).** Unattended watcher never fired (waited for model_250; checkpoints exist only every 100) — fixed (`--its`, multiple-of-100 check). G1 rung 2 (turntable at 2.2): it 500 0/10 · 1000 4/10 · 1500 4/10 · 1800 3/10 · 1999 3/10; yaw-probe mean 0.21 → 0.19 (r2_v1 best 0.29); no falls, brake 1.1 m, backward pass.
- **Turntable protocol:** DESIGN lists ωz ≤ 2 and "360° < 3 s" — impossible together (3.14 s at 2 rad/s). Event 12 is scored on rotational speed → drill now commands the max trained rate **2.5 rad/s**; bars (< 3 s, < 0.3 m drift) unchanged. With it: it 1500 **6/10** (turntable 2.70 s left / 2.86 s right, drift ≤ 0.10 m, all seeds pass), it 1800 6/10, it 1000 4/10, it 1999 3/10.
- **Remaining misses are asymmetric:** right turns weaker (spin −2.2: 3.2–3.7 s vs +2.2: 2.96 s; sprint 3–3.7 m/s with wz −0.33..−0.36: yaw RMS 0.36–0.39). → **left/right symmetry augmentation** (`poolympic/tasks/symmetry.py`, rsl_rl Symmetry extension via `SymmetricRunner`): every PPO mini-batch + its mirror. Mirror map verified physically (`tests/test_symmetry.py`: obs of the mirrored MuJoCo state = mirror(obs) < 1e-5; forward kinematics of the mirrored joint map puts every body on its partner's reflection < 2 mm — validates the joint-axis sign conventions).
- **r2_v3 prepared:** task `PoOlympic-Matt-Rung2-Omni-Sym` (r2_v2 rewards, full envelope from it 0), warm start r2_v2 it 1500, 1500 its.
- **Pedestal fine-tune `ped_v1` running** (task `PoOlympic-Matt-Pedestal`, warm start r0_v2 it 1000): termination when a foot/toe box centre leaves the 1 × 1 m square (= stepping off), `feet_centred` reward (w 1.0, std 0.3), gusts ±0.8 m/s every 2–4 s (heat escalates to ~1 m/s). Baseline for comparison (`tools/eval_pedestal.py`, 10 identical heats): r0_v2_it1000 mean heat 33.8 s, mean survival 28.2 s, first-out 20.0 s; Rung 0 G1 10/10 (foot excursion 0.48 m).

## 2026-09-28 00:55 · ped_v1 — FAILED (no improvement); r2_v3 running
- **ped_v1 (1000 its):** identical 10-heat comparison — baseline r0_v2_it1000 mean survival 28.2 s / heat 33.8 s; ped_v1 it 300 27.3 / 34.1; it 600 28.0 / 33.8; it 999 24.5 / 30.8 and Rung 0 G1 9/10 (foot excursion 0.59 m). No gain, late regression.
- **Why:** ~90 % of training episodes ended by `off_pedestal`, episode length flat at ~500 ticks (10 s) from it 100 on — gusts up to ±0.8 m/s per axis (≈1.1 m/s) every 2–4 s from iteration 0 are beyond what the policy can recover from inside a 1 m box, so there is no learning signal (a failure curriculum, not a hard task).
- **Decision:** keep r0_v2_it1000 as the Iron Pedestal brain. If GPU time remains: ped_v2 with a gust-magnitude curriculum that tracks success (0.3 → 1.0 m/s, like the heat itself).
- **r2_v3** (symmetry augmentation, from r2_v2 it 1500) started 00:48, 1.82 s/it, 1500 its; initial mirror loss 0.36.

## 2026-09-28 00:56 · r2_v3 stopped at it ~330 (mirror bug) → r2_v3b
- r2_v3 it 300 gate: 0/10, turntable 3.42 / 3.22 s, yaw-probe mean 0.31 (worse than its r2_v2 it 1500 start) and the logged mirror loss ROSE 0.36 → 1.15 — the policy was getting less symmetric under symmetry augmentation.
- **Bug:** the mirror map shifted the gait phase by half a stride unconditionally, but the contract freezes the clock at 0 while |cmd| < 0.1 — a mirrored standing athlete is still at phase 0. Every mirrored standing sample (15 % zero-command envs) carried the impossible phase 0.5. The physical test drew random phase + random commands, so it never hit the frozen-clock case.
- **Fix:** flip the phase only when |cmd| ≥ threshold; test now includes 30 % standing states (3/3 pass). r2_v3b relaunched from the same init (r2_v2 it 1500), 1.49 s/it, gate at 300/600/900/1200/1499.

## 2026-09-28 01:08 · r2_v3b stopped (it ~315) → r2_v3c with explicit mirror loss
- r2_v3b it 300: 0/10, turntable 3.06 / 3.46 s, yaw-probe mean 0.23; mirror loss still rising 0.38 → 0.70. Augmentation alone can't symmetrise an asymmetric starting policy: mirrored samples reuse the original action's old log-prob, their PPO ratios fall outside the clip range and contribute ~no gradient.
- **r2_v3c:** augmentation + rsl_rl mirror loss (coeff 0.5); plain warm start from r2_v2 it 1500 (normalizer + action noise kept — same rung, same command distribution; the re-seed/noise floor had degraded the start). Mirror loss now falls 0.39 → 0.13 in 60 its. 2.16 s/it, 1500 its, gates at 300/600/900/1200/1499.

## 2026-09-28 01:33 · contract v3 (gait clock yaw weight 0.5 → 1.2) · r2_v4
- **r2_v3c** (mirror loss) symmetrised the policy (mirror loss 0.39 → 0.03 by it 600) but converged to the slower side: turntable 3.18 / 3.32 s, G1 0/10 at 300 and 600.
- **Diagnosis:** at a 2.5 rad/s pivot the v2 clock ran 1.05 Hz → ~60° per step at MATT's ±40° hip-rotation limit. Test with the clock swapped in at inference only (no retraining): r2_v3c it 600 3.18/3.32 s (1.05 Hz) → 2.96/3.06 (1.20 Hz) → **2.88/2.96 s (1.40 Hz)**; r2_v2 it 1500 left 2.70 → 2.54 s. The clock is the limiter.
- **Contract v3:** `GAIT_HZ_YAW_WEIGHT = 1.2` (speed = |v_xy| + 1.2·|wz| → 1.4 Hz at 2.5 rad/s; walking/running barely change; standing identical). Single constant read by contract.py, the training clock (mdp.AthleteCommand) and C# `Contract.GaitHz` (new `gait_hz_yaw_weight` field). check_task rung2 PASS (clock 3.0e-7). Rung 0 / test brains re-exported under v3 — references frame-for-frame identical (clock frozen at zero command). EditMode 18/18; pytest pass. `rung1.onnx` (v2) is retired: PolicyRunner refuses it under v3; the Rung 2 brain covers Rung 1 events.
- **r2_v4:** symmetric runner (augmentation + mirror loss 0.5), warm start r2_v3c it 600 (plain), contract v3 clock, 1500 its at 1.69 s/it, gates 300/600/900/1200/1499.

## 2026-09-28 02:05 · r2_v4 it 600 = best so far (6/10, symmetric turntable) · C8 pipeline dry run PASS
- r2_v4 gates: it 300 0/10 (turn 3.04/3.14) · **it 600 6/10**, turntable 2.76 / 2.82 s (all seeds pass, symmetric), yaw probe spin err 0.29, mirror loss 0.0097.
- it 600 misses by segment type: stop 2/4 (lin RMS 0.28–0.38 — holding zero velocity after shoves has regressed), crab 3/12 (all ~0.9 m/s, lin 0.20–0.29), turn 2/15 (1.1–1.5 m/s with 1.35–2.0 rad/s, yaw 0.33–0.37), sprint 1/19 (3.7 m/s + wz −0.33, lin 0.255).
- **C8 dry run** with r2_v4_it600 (`make_g6.py … rung2`: stand / back −1.5 / crab ±0.75 / spin ±1.5 / sprint 3.5 / walk-turn): CPU meet pre-check PASS; **Unity G6 PASS 8/8** (1 s drift ≤ 1.4e-6; the crab-right and spin lanes diverge chaotically after contacts by 5–9 cm at 5 s in BOTH the CPU meet and Unity — identical numbers, i.e. Unity reproduces the multi-athlete CPU run; height RMS ≤ 5.6e-4 m, torque ±0.4 %). Capture `parity/c8/c8dev_tick252.png`.

## 2026-09-28 02:29 · r2_v4 finished — **7/10 at it 1499** · r2_v5b (widened training envelope)
- r2_v4 gates (turntable at 2.5 rad/s): 300 0/10 · 600 6/10 (2.76/2.82 s) · 900 0/10 (3.04/3.10 s) · 1200 5/10 (2.82/2.78 s) · **1499 7/10** (turntable 2.64/2.68 s, yaw-probe mean 0.22; misses: tracking lin 2, yaw 2). The turntable sits at the 3 s edge and flips between checkpoints; the 2.5 rad/s drill command is also the edge of the training range.
- **r2_v5b:** task `PoOlympic-Matt-Rung2-Omni-Sym2` — training envelope ~20 % wider than the G1 envelope in the hard directions (wz ±3.0, vy ±1.2; vx unchanged −1.5…4), zero-command stops 0.15 → 0.20; symmetric runner; warm start r2_v4 it 1499 (a first r2_v5 from it 600 was restarted after 3 min when 1499's gate came in better). 2000 its at 1.80 s/it, gates 300…1999.
- 02:51 r2_v5b gates: it 300 6/10 (turn 2.82/2.82 s), it 600 6/10 (2.80/2.82) — turntable now stable; misses: fast sprint + small turn (yaw), crab ~0.9–1 m/s (lin). r2_v4 it 1499's only 3 misses: sprint 3.0/3.7 m/s with wz −0.36/−0.33 (yaw RMS 0.31/0.35), crab −0.94 (lin 0.257).
- Prepared r2_v6 option: `AthleteCommandCfg.max_lateral_accel` — |wz| ≤ a_max/|v| at resampling (task `…-Sym3`, a_max = 4 m/s²: every G1 command stays inside; 4 m/s × 3 rad/s-style combos clipped). Verified on 10 240 sampled commands: max |v|·|wz| = 4.000.
- 03:13 r2_v5b: it 900 6/10 (turntable 2.58/2.60 s, yaw mean 0.19) · it 1200 5/10 — plateau at 5–7/10. Diagnosis on r2_v4 it 1499 (5 s holds): yaw RMS of fast sprint-turns 0.29–0.36 raw vs **0.19–0.23 stride-averaged** (0.5 s) → the yaw misses are mostly per-stride pelvis oscillation; linear misses are real: 3.69 m/s + |wz| 0.33 lin 0.34 raw / **0.26 stride-avg** (speed shortfall while turning), crab ±0.94 m/s 0.21–0.24 / 0.20–0.22. G1 keeps the raw metric (bar unchanged).

## 2026-09-28 03:41 · r2_v5b final · r2_v6 (feasibility-capped commands)
- r2_v5b gates: 300 6 · 600 6 · 900 6 · 1200 5 · **1500 7/10** (turntable 2.66/2.68 s, yaw-probe mean 0.18 — best yet) · 1800 6/10 (2.52/2.52 s). Every checkpoint from 300 on passes the turntable; remaining misses = fast sprint + turn and ~1 m/s crab.
- **r2_v6:** task `…-Sym3` (r2_v5 envelope + `max_lateral_accel` 4 m/s²), warm start r2_v5b it 1500, 2000 its at 1.73 s/it, gates 300…1999.

## 2026-09-28 04:25 · **r2_v6 it 1200: GATE G1 rung 2 = 10/10** · margin 22/30 → r2_v7 prepared
- r2_v6 gates: 300 7/10 · 600 8/10 · 900 9/10 · **1200 10/10** (turntable 2.44/2.44 s, brake 1.16 m, backward 20 m, 0 falls, yaw mean 0.17). The lateral-acceleration cap removed every yaw failure (from it 300 on).
- **Margin set (30 seeds): 22/30.** 10 of 11 failing segments are sprints at 3.2–3.8 m/s (lin RMS 0.20–0.48), one crab 0.97 m/s (0.204). Steady-state probe: mean speed undershoots by ~4–5 % at every speed (2.5 → 2.40, 3.0 → 2.87, 3.5 → 3.35, 4.0 → 3.78 m/s) plus per-stride oscillation; peak torque at the ankle cap (220 Nm). The fine tracking kernel (std 0.5) pays 91 % at a 0.15 m/s shortfall — no incentive to close it; Rung 1's bar stopped at 3 m/s so it didn't show there.
- **r2_v7** (task `…-Sym4`): r2_v6 + track_lin std 0.5 → 0.3 (same shortfall costs ~22 %). Warm start from r2_v6's best checkpoint.

## 2026-09-28 04:50 · C8 PASS with rung2.onnx (r2_v6 it 1200) — Phase C complete
- `rung2.onnx` exported; `make_g6.py rung2 rung2`: CPU meet pre-check PASS.
- Unity: G2 4.4e-16, G3 ≤ 4.3e-6, G4 (1 s) ≤ 9.5e-7 on all 8 lane references (ParityHarness direct — the EditMode test list is fixed at compile time and did not yet include the new rung2 refs). **G6 8/8 PASS**, 1 s drift ≤ 4.6e-6; crab-right / spin lanes diverge chaotically by 5 s (13 / 27 cm) exactly like the CPU meet (height RMS ≤ 1.9e-3 m, torque ±0.2 %). Capture `parity/c8/rung2_tick252.png`.
- r2_v7 (track_lin std 0.3, from r2_v6 it 1200) training to lift the margin set; if it beats rung2.onnx on the 30-seed margin set it replaces it (C8 re-run ~15 min).

## 2026-09-28 04:55 · r2_v7 stopped (plateau) → r2_v8 sprint focus
- r2_v7 (track_lin std 0.3): 20-seed gates it 300 17/20, it 600 17/20 (yaw mean 0.13–0.14 best yet, only lin misses). Speed bias fixed in the normal range (2.5 → 2.47, 3.0 → 2.93 m/s; RMS 0.06 / 0.10) but 4.0 → 3.76 (RMS 0.39). Margin set unchanged 22/30: misses are 3.2–3.8 m/s sprints with a mild turn (|wz| 0.2–0.5, lin 0.21–0.63) + one stop (0.202). A straight 3.5 m/s sprint passes (0.18) — the athlete slows when turning at speed.
- **r2_v8:** `…-Sym5` = r2_v7 + `sprint_fraction` 0.3 (vx 2.5–4.0, |wz| ≤ 0.6; verified 34 % of commands in that band, cap intact). Warm start r2_v7 it 600. 1500 its at 1.57 s/it; 20-seed gates.

## 2026-09-28 05:21 · r2_v8 it 600 beats rung2.onnx everywhere · failing sprints = acceleration time
- r2_v8 20-seed gates: it 300 18/20, it 600 18/20 (yaw mean 0.136). Official G1 (seeds 1000–1009) it 600: **10/10 PASS**. Margin set (official protocol): 23/30.
- **Every margin failure is a sprint started from ~standstill** (Δvx 2.7–3.8 m/s, lin 0.20–0.55); the one 3.7 m/s sprint that started already moving (Δv 0.29) tracks at 0.10. Standstill → speed test: r2_v8 reaches ±0.2 m/s of 3.0 / 3.5 / 4.0 after 1.96 / 2.46 / 3.32 s (~1.5 m/s² average); steady-state RMS after 2.5 s: 3.0 → 0.056, 3.5 → 0.116, 4.0 → 0.259 (top speed ≈ 3.8 m/s; ankle at its 220 Nm cap). The drill's fixed 1.5 s settle window (my evaluator choice — the spec only says "vel err < 0.2 m/s") therefore scores acceleration, not tracking.
- **Diagnostic steady-state variant** (`rung2_episode(steady_state=True)`, settle = max(1.5, 0.5 + |Δv|/1.5 s); NOT the G1 bar): rung2.onnx 9/10 official, 25/30 margin · **r2_v8 it 600 10/10, 28/30** (2 marginal yaw).
- Open questions for the user: (1) keep the fixed 1.5 s settle as the G1 bar, or adopt a Δv-aware settle for tracking? (2) MATT's top speed with elite-athlete torque caps is ~3.8 m/s, below the 4.0 m/s top of the Rung 2 envelope.

## 2026-09-28 06:00 · rung2.onnx := r2_v8 it 600 · C8 re-run PASS · ped_v2 running
- r2_v8 gates (20 seeds): 300 / 600 / 900 / 1200 / 1499 all **18/20** (only lin misses; turntable 2.44–2.64 s; yaw mean 0.136–0.162) — converged. Chosen it 600 (best yaw, fully evaluated: official 10/10, margin 23/30, steady-state 28/30).
- **C8 with the new rung2.onnx:** CPU meet pre-check PASS; Unity G2 4.4e-16, G3 ≤ 3.8e-6, G4 ≤ 1.0e-6 on 8 lane refs; **G6 8/8 PASS** (1 s drift ≤ 4.0e-6; crab / spin lanes diverge chaotically by 5 s as in the CPU meet).
- **ped_v2** (adaptive heat-shaped gusts) started 05:44 from r0_v2 it 1000 (plain warm start); gust level already 0.60 m/s by it 30.

## 2026-09-28 06:35 · ped_v2 — no gain (curriculum oscillated) · overnight summary
- **ped_v2** (1200 its): identical 10-heat comparison — it 400 mean survival 25.1 s (G1 9/10), it 800 26.6 s (8/10), it 1199 26.4 s (10/10) vs baseline r0_v2_it1000 28.2 s. The adaptive level oscillated bang-bang 0.3 ↔ 1.2 m/s: it steps ±0.02 on every reset *event* (many per iteration, arriving in synchronized waves), so difficulty never settled. If revisited: update once per iteration from a smoothed success rate. The baseline already sits near the physical limit of a 1 m block under ~0.7 m/s gusts. **Event 1 keeps r0_v2_it1000.**
- **Overnight outcome:** Phase C complete. Rung 2 brain `rung2.onnx` = r2_v8 it 600 (G1 10/10; Unity G2–G6 PASS). Contract v3. Physically verified L/R symmetry augmentation + mirror loss, lateral-acceleration command cap, sprint-focus sampling, unattended checkpoint gating. Two pedestal fine-tunes failed (documented).
- **Decisions for the user:** (1) Rung 2 tracking drill: keep the fixed 1.5 s settle (official, margin 23/30) or adopt the Δv-aware steady-state settle (margin 28/30)? (2) MATT's top speed ≈ 3.8 m/s vs the Rung 2 envelope top of 4.0 m/s (Terminal Velocity "stretch ≥ 4 m/s") — accept, or raise actuator limits (a body change → retrain)? (3) Turntable drill commands 2.5 rad/s (the spec's ωz ≤ 2 cannot meet 360° < 3 s).

## 2026-09-28 · Rung 2 rulings applied (user) · rung2.onnx under the final protocol
- User decisions: (1) tracking judged at steady state — settle max(1.5, 0.5 + |Δv|/1.5) s, now the official drill; (2) Rung 2 envelope tops out at MATT's ≈ 3.8 m/s (was 4.0); (3) turntable drill at the max trained rate 2.5 rad/s. DESIGN.md §1 updated.
- **rung2.onnx (r2_v8 it 600): G1 10/10 PASS; margin set 29/30** — the one miss is a sprint segment with yaw RMS 0.30 (bar < 0.3), turntable 2.64 s, brake 1.22 m.

## 2026-09-28 · Events 8 / 19 / 22 playable (8 runners, Rung 2 brain)
- **Physics:** `scene_track8.xml` = compose_meet with lane origins from the home-straight venue (1.22 m lanes; the back straight is identical in the runner's frame). **Bug found by the CPU sim:** the idle cube pool was parked on y = 0, x = 50–80 — lane 4's running line; the Terminal Velocity runner in lane 4 fell at ~50 m in every race. Track scene parks the pool 30 m aside (`park_offset`). G0 track8 8/8 PASS.
- **Rules (Python `events/track.py` ≡ Unity `TrackRaceEvent`):** dash 30 m at cmd 3.8 m/s (rank finish time); terminal 84.39 m at cmd 4.0 (rank peak 1 s speed); brake: run-in 3.0 m/s, each runner brakes at its seeded "nerve" distance U(1.4, 2.2) m before the line — with one fixed trigger all 8 stopped within ±1 cm (7 DQ), so the brake point is the contest; toe past the line = DQ, rank by gap. Lane keeping via `PolicyRunner.laneKeeping`.
- **CPU:** dash 9.22–9.34 s (tight fields); terminal peaks 3.99–4.06 m/s; brake gaps +0.02…+0.68 m, 1–4 DQs per heat. **Unity play-throughs:** dash L3 9.30 s (field 9.30–9.42); terminal L5 4.05 m/s; brake L7 14 cm short, gaps 14–59 cm, 1 DQ. Captures `parity/d5/`. Hub: 4 playable events. pytest 22/22, EditMode 26/26.

## 2026-09-28 · Events 9 (Inverted Sprint) and 12 (360 Turntable) playable (8 athletes, Rung 2 brain)
- **9 Inverted Sprint** = track framework mode `inverted`: 20 m at cmd vx −1.5 m/s (Rung 2 backward limit), lane keeping (backward-aware), pelvis > 0.61 m off the lane line = DQ, rank by finish time. Physics = `scene_track8.xml` unchanged; Unity turns the stadium 180° about venue lane 5 (`PlaceStadium(extraYawDeg)`), so MuJoCo lane k runs in venue lane 8 − k. CPU (4 heats): all finish, 14.68–15.14 s, no DQ. **Unity play-through:** L1 14.66 s, field 14.66–15.04 s (`parity/d5/inverted_*.png`).
- **12 360 Turntable:** `scene_turntable8.xml` (compose_meet on the venue's 2 × 4 spin-spot grid; **G0 turntable8 8/8**, 0 mismatches vs the Python fingerprint of each lane). Rules (`events/turntable.py` ≡ `TurntableEvent` + `TurntableHud`): GO → pure yaw command 3.0 rad/s (training envelope top) in a seeded per-heat direction until 3 full turns, then zero command; score = time + 2 s per m of max pelvis drift; fall = out, drift > 1.1 m ring = DQ. Spin probe before choosing: no falls at any rate from 2.0 to 4.5 rad/s (8 lanes × 2 seeds), drift 2–21 cm — over-commanding carries no risk, so the event stays at the trained top rate. CPU heats: times 6.52–6.78 s, drift 3–12 cm, winner varies with seed. **Unity play-through:** S3 wins 6.64 (6.54 s, 5 cm), field 6.54–6.68 s / 5–9 cm (`parity/d5/turntable_*.png`).
- `EventScenes.BuildMeetScene` = shared 8-athlete scene setup (track races + turntable). Hub: 6 playable events. pytest 24/24 (new `tests/test_events.py`), EditMode PoOlympic 26/26.

## 2026-09-28 · Event 10 (Crab Shuffle) playable
- **Physics:** `scene_crab8.xml` — athletes turned 90° to the course (`venue_lane_origins(10, 3, extra_yaw_deg=90)`: they face the course's left and side-step to their right = −y), so lanes are 1.22 m apart along x; the 9 steel rails of the venue (4 cm bars, 0.3 m up, on every lane line, 20 m) are real MuJoCo boxes (`compose_meet(props=…)`, all collision bits). G0 crab8 8/8 per lane; rails compiled in Unity ≡ layout to 1e-14 m (`ParityTools.DumpProps`, since the lane fingerprint does not cover world props).
- **Steering:** new `PolicyRunner.steer` hook (runtime delegate, called every control tick before the observation, after laneKeeping — tick-exact like the Python loops). Crab law (`events/crab.py crab_command` ≡ `CrabShuffleEvent.Steer`): vy = −1.2, wz = clip(2·wrap(−yaw), ±0.5), vx = clip(−1.0·x_offset, ±0.3).
- **Probe:** 8 lanes × 2 seeds at vy 0.8…1.8 m/s: zero leg crossings, zero rail touches, zero falls, drift ≤ 0.16 m; the real side-step speed saturates at ~1.5 m/s. Detection verified by aiming every lane 0.6 m off its line: 13–17 rail touches each and all fall — the rails are genuine hazards. Event runs at the training-envelope top 1.2 m/s (over-commanding adds no risk).
- **Rules:** 20 m; +1 s per leg crossing (left foot passes right of the right foot along the pelvis' left axis) and per new rail contact; fall = out; rank by time + penalties. CPU heats 18.2–18.5 s. **Unity play-through:** L6 18.10 s, field 18.10–18.48 s, drift ≤ 0.14 m, no faults (`parity/d5/crab_mid.png`).
- Generic `StandingsHud` + `IStandingsBoard` (crab + turntable; TurntableHud removed). pytest 26/26; EditMode PoOlympic 26/26. Note: the MuJoCo plug-in's own EditMode tests are flaky under repetition (`MjSceneGenerationTests`: a different one of SceneRecreatedAfterDeletion / MjcfHierarchyMirrors… fails on identical reruns — static naming/scene state inside the plug-in, not our code).

## 2026-09-28 · Event 11 (Slalom Sprint) playable · slalom venue poles moved to the lane centre lines
- **Venue change:** the stadium's slalom poles stood 0.35 m either side of each lane line, alternating — running straight down the lane already passed them "alternately", so there was nothing to weave. `build_venues.py` now puts the 7 poles on every lane's centre line (x = 3, 7, … 27 m, classic slalom) and records them in venues.json (`events.11.poles`). Rebuilt in Blender + re-exported `Stadium.glb` (same export as before: Stadium collection, UVs on, no camera/light); verified: venues.json identical to HEAD apart from `poles`, GLB identical in structure (1547 nodes / 1307 meshes / 29 materials / 2952 accessors) and only the 56 `E11_Gate_*` nodes changed bounds.
- **Physics:** `scene_slalom8.xml` = compose_meet on venue 11 + 56 pole boxes (4 cm × 1.2 m, all collision bits). G0 slalom8 8/8; Unity poles ≡ layout to 1e-15 m.
- **Steering tuning (CPU):** heading steering onto a racing line y*(x) = A·cos(π(x − 3)/4): at 2.5–3 m/s with a 2.5 rad/s yaw limit almost every runner fell, most with zero pole contact (the alternating yaw commands, v·wz up to 7.5 m/s², are far outside the brain's training — the 4 m/s² lateral-acceleration cap). Strafe weaving (heading held, vy follows the line) was worse (slow lateral tracking → 70–190 clips per heat, falls). Final: vx 2.2 m/s, |wz| ≤ 4/v, heading gain 2, lateral gain 1, look-ahead 0.8 m → 16/16 finish at A = 0.5.
- **Racing line = the contest:** at 2.2 m/s, A 0.35 → ~11 pole clips per runner, 0.40 → ~5, 0.45 → ~2–3, 0.50 → ~1, 0.55 → none but ~1 in 2 falls (all near x ≈ 12 m). Each runner draws A ∈ U(0.40, 0.56); +0.5 s per clip, +2 s per pole on the wrong side, fall = out. CPU heats: winners 15.9–16.1 s on clean wide lines, fields to 21.4 s, 0–2 falls per heat.
- **Unity play-through:** L8 wins 16.06 s (A 0.51, clean); tight lines A 0.44–0.46 clipped 2–6 poles; no falls (`parity/d5/slalom_*.png`). Hub: 8 playable events. pytest 28/28, EditMode PoOlympic 26/26.

## 2026-09-28 · Event 5 (Gust Gauntlet) playable — spring-mounted shaker platforms, Rung 2 homing
- **Physics:** `scene_shaker8.xml` = compose_meet(shaker=…) on the venue's 2 × 4 grid: every lane stands on its own 1.6 m × 0.1 m, 150 kg platform body `L<k>_shaker` with x/y slide joints (stiffness 36 300 N/m, damping 1 160 N·s/m → 2.0 Hz, ζ ≈ 0.2 with the athlete aboard; ±0.25 m travel); top at z = 0, ground at −0.1. No actuator and no per-substep plumbing: a floor shake is a velocity kick on the slide joints through the existing tick-exact disturbance path (new Disturbance kind `kick` = Δqvel on a 1-dof joint), then the spring/damper physics plays out identically on CPU and Unity. The platform bodies/joints are lane-prefixed, so **G0 shaker8 8/8** covers their mass, stiffness, damping and range. `_Lane` / AthleteJudge count the platform as support (stepping onto the ground = STEPPED OFF).
- **Probe (8 lanes, one disturbance at 1 s):** floor kicks alone are easy for both brains (even 3 m/s → recovery ~0.65 s, identical across lanes). Wind bursts: Rung 0 stays within ~0.2 m but never walks back to the spot (it cannot be commanded); Rung 2 with a zero command drifts up to 0.65 m. **Rung 2 + homing** (walk command back to the spot, gain 1.5, ≤ 0.6 m/s, 8 cm deadband, heading held) recovers a 0.6 m/s burst in 1.2–2.0 s; 0.9 m/s → 2 of 8 step off; 1.2 m/s → 6 of 8. The event therefore uses the Rung 2 brain (catalogue brain rung0 → rung2).
- **Rules (`events/gauntlet.py` ≡ `GustGauntletEvent`):** 10 rounds every 4 s from 1 s: lateral burst (same size for all, own seeded side ±90° ± 30°) 0.5 + 0.05 m/s per round; every 2nd round also a 2 m/s platform jolt. Recovery per round = until within 0.15 m and < 0.2 m/s for 0.5 s (cap 4 s); fall / stepped off = out (remaining rounds count 4 s). Rank: still in by total recovery, then eliminated (later = better). CPU heats: 0–1 survivors, first out 2.4–9.8 s, most eliminated in rounds 6–9, winner varies by seed.
- **Unity play-through:** S4 wins (13.2 s total), S8 second (13.8 s), eliminations at 13.7 / 25.5–25.8 / 33.6–33.7 s — same pattern as the CPU heats. The moving platforms are drawn by their MuJoCo geoms (stadium hazard material), the stadium's static shaker pads hidden (`parity/d5/gauntlet_*.png`). Hub: 9 playable events. pytest 30/30, EditMode PoOlympic 26/26.

## 2026-09-28 · Phase Z — ZOMBIE body + tasks (user: own 1.1 m body, style rewards, "weaker but relentless", Rung 0 + 2)
- **Asset (Z1):** `SourceArt/Zombie/convert_zombie.py` (Blender). Blender's FBX import leaves the mesh object at −90° X to its armature: a 60° forearm bend threw vertices 0.87 m; zeroing the mesh rotation re-aligns skin and bones exactly (bend → 0.26 m). The file's bind pose is a hunched, splayed stance; its "0_T-Pose" action is a clean T-pose but still has a 36° knee crouch → legs straightened (thigh 17–18°, knee 34–36°) so qpos = 0 has straight legs like MATT (anatomical joint ranges); twist/share/face bones merged into their limbs; 22 bones renamed to Mixamo names. 25k verts all weighted, 1.135 m, L/R 5 mm.
- **Body (Z2):** `poolympic/bodies.py` profiles, selected per process with `POOLYMPIC_BODY`; extract_skeleton / build_mjcf / contract / fingerprint body-aware with **MATT's artefacts byte-identical** (skeleton, all MJCFs + meets, body report, contract, fingerprint). Zombie: λ = 1.135/1.837 = 0.618, 22 kg de Leva, torque ∝ m·l × 0.7 (hip/knee cap 33 Nm, ankle 26, shoulder 9.5, elbow 8.3; kp 36/24/7/6, kv ∝ √λ), armature ∝ m·l², no finger bones (forearm capsule to the far hand skin), heel from sole skin (AccuRig weights the heel to the calf), **full self-collision** (all geoms on the lane's leg bit; only parent/child + inner thighs excluded; no touching pairs at rest). Default stance (contract): trunk 15°, hips 20°/abd 5°, knees 35°, ankles 15°, arms out in front (elev −10°, flex 75°), elbows 20° → pelvis 0.562 m. Gait clock Froude-scaled: 1.018 Hz + 0.324 Hz per m/s. Zero brain collapses (like MATT's), no initial self-penetration.
- **Tasks (Z3/Z4):** `tasks/zombie_env.py` = MATT's recipe (Rung 2 = the final r2_v8 recipe) with Froude scaling: fall 0.55 × 0.562/0.955 = 0.324 m, pushes/commands × √λ, times × √λ, yaw rates ÷ √λ, cube drop × λ, torque penalty ÷ torque_scale², contact buffers up for self-collision. **Personality rewards:** torso hunch 20° (replaces "upright"), arms held forward (shoulder/elbow posture std 0.3 at all speeds), wide stance 0.22 m, low shuffling feet (foot box ≤ 5 cm, Rung 1+). check_task zombie_rung0/1/2 C1 PASS. G1 gates per body (evaluate.py): zombie rung 0 fall 0.324 m, recovery 1.18 s judged against its own settled lean, shove 0.39 m/s, cube from 0.93 m, foot box 0.31 m; rung 2 lin tol 0.157, yaw 0.382, turntable 3.18 rad/s < 2.36 s, brake 2.36 m/s < 1.24 m, backward 1.18 m/s over 20 m; top speed 2.99 m/s (pure scaling — to be set from what the weakened zombie actually reaches). MATT's bars unchanged (pytest 30/30).
- **Housekeeping (user):** obsolete TensorBoard runs deleted (1.2 GB → 346 MB; kept r0_v2, r1_v2, r2_v8 for warm starts); Unity closed for training; TensorBoard on `training/runs` (:6006); MuJoCo viewer (`play … --viewer native --device cpu`: on the GPU it doubled the iteration time 2.15 → 5.1 s).
- **z0_v1** launched 14:33: `PoOlympic-Zombie-Rung0-Stand`, from scratch, 4096 envs, 1500 its (~2.7 s/it with self-collision). It 100: episode 371/1000 ticks; it 184: 565, reward 27.
- **Viewer cost (user asked):** no viewer ~2.8 s/it; CPU viewer alongside ~4.4 s/it (+55 %, 2–3 cores); GPU viewer ~5.1 s/it. Policy: viewer only for a short look at each gate checkpoint, open freely after a run. z0_v1 it 260: episode length 972/1000.
- **z0_v1 it 600 (G1 rung 0, zombie bars): 8/10** — 0 falls, every hit recovered (worst 0.26 s ≤ 1.18 s), peak lean change ≤ 13.7°; the 2 misses are foot excursion 0.315 / 0.328 m vs the 0.309 m box. Stance (quiet 5 s): torso lean 19.0° (target 20), foot width 0.239 m (0.22), arms at default (elev −10 / flex 74 / elbow 20), knees 27° (default 35), pelvis 0.567 m — the personality holds. Viewer shown 3 min on it 600.
- **z0_v1 it 900: G1 rung 0 10/10 PASS** (0 falls, worst recovery 0.24 s, foot excursion 0.185–0.296 m ≤ 0.309) · margin set 25/30 (0 falls; 4 foot excursions 0.314–0.515 m, 1 hit not re-settled before the log ends). MATT's r0_v2_it1000 for comparison: 10/10, margin 29/30. Checkpoints 1200 / 1499 to be gated the same way.
- **z0_v1 final:** it 1200 G1 9/10 (margin 25/30) · it 1499 **10/10** (margin 25/30). Tie-break on Event 1 (all-zombie pedestal8 heats, 6 identical seeds): it 900 mean survival 27.5 s, **it 1499 29.1 s** (MATT's r0_v2_it1000 in all-MATT heats: 28.2 s) → **`parity/brains/zombie_rung0.onnx` = z0_v1 it 1499** (fingerprint = zombie scene).
- **Mixed meets (Z7, Python):** `tools/compose_mixed.py` composes an event scene with a body per lane from the per-body robot MJCFs (lane bits shifted per lane, per-body keyframes, excludes, actuators). `--verify`: an all-MATT lineup compiles to exactly `scene_pedestal8.xml` (per-lane fingerprints, keyframe, nq/nu). `scene_pedestal8_mzmzmzmz.xml` (MATT lanes 1/3/5/7, zombie 2/4/6/8) + all-zombie `scene_pedestal8_zombie.xml`. `iron_pedestal.run_heat(brains={body: onnx})`: per-lane brain, defaults and fall line (MATT 0.55 m, zombie 0.324 m). First mixed heats (MATT r0_v2_it1000 vs zombie it 900, 4 seeds): 29–35 s, **MATT 2 wins, zombie 2 wins**.
- **z1_v1** launched 15:47: zombie Rung 1 from scratch, 4096 envs, 3000 its (~2.5 s/it).
- **Unity (Z7 for Event 1):** Event_IronPedestal_Heat.unity rebuilt as the mixed meet (per-lane body contract / brain / visual, labels M1…Z8; zombie visual binds within tolerance). **G0 pedestal8 mixed 8/8** (0 mismatches per lane, incl. the zombie's self-collision bits and actuators). Play-through: M7 wins at 32.0 s; zombies out 28.2–32.0 s, MATTs 20.1–31.3 s (`parity/d5/pedestal_mixed_*.png`). Menu: ZOMBIE roster card, each event shows its scene's lineup (Event 1: 4× MATT + 4× ZOMBIE). EditMode PoOlympic 26/26. Unity open ~25 min during z1_v1: 2.5 → 3.5 s/it.
- **Mixed meets for every event (prep, Python):** compose_mixed.py now covers pedestal8 / track8 / turntable8 / crab8 (rails, 90° facing) / shaker8 (platforms) / slalom8 (poles); `--verify`: all-MATT lineups compile to exactly the existing 6 scenes (per-lane fingerprints, world props, keyframe, sizes). Events share `iron_pedestal.make_lanes`: per lane body, brain, default pose, fall line and **gait clock** (the zombie's stride clock differs); every event takes `scene / layout_path / brains`. MATT regression: 7 event runs (dash, brake, inverted, turntable, crab, slalom, gauntlet) byte-identical before/after; the mixed pedestal heat reproduces; pytest 30/30. Open for the zombie's Rung 2 events: per-body commands (speeds, slalom line/speed, crab speed, turntable rate) — to be set from what zombie Rung 2 can do.

## 2026-09-28 evening · zombie Rung 1 (z1_v1) → Rung 2 (z2_v1)
- **z1_v1** (from scratch, 3000 its, 2 h 09 min). Zombie G1 rung 1 (30 m dash, lane keeping; bars vel RMS < 0.118 m/s, lateral < 0.309 m): **it 1500 3/10** — 10/10 finished, 0 falls, vel RMS 0.066–0.143, lateral 0.26–0.39 m (misses: lateral just over 0.309, 2 on vel) · it 2000 0/10 (lateral 0.46–0.63) · it 2500 0/10 (lateral up to 28 m, one DNF) · it 2999 0/10 (8–33 m: steering ignored at speed). Training metric error_vel_yaw: zombie 1.84 → 0.74 (it 1000) → 1.73 (it 2990) as the speed curriculum widens; MATT's r1_v2 over the same run 0.78 → 0.61 → 0.97. Same failure MATT's Rung 1 had (lane drift from heading tracking traded for speed), much stronger on the zombie. Rung 1 is the stepping stone: Rung 2 trains yaw directly (heading command off, track_ang w 2.0 + coarse, symmetric runner).
- **z2_v1** launched 18:05: `PoOlympic-Zombie-Rung2-Omni` (MATT's final r2_v8 recipe, scaled), warm start **z1_v1 it 1500** (best lane keeping) via warm_start.py (command stats → stage 0: vx −0.786…2.358, vy ±0.393, wz ±1.272; standing 0.20; action std floor 0.25), 3000 its, 2.7 s/it; episode length 900/1000 at it 37.
- **z2_v1 stopped at it ~430 (18:17):** MATT's *final* r2 recipe applied cold (symmetric runner, sprint fraction 0.3 at 2.0–3.1 m/s, lateral-accel cap, track_lin std 0.236) drove the warm-started zombie towards standing still: track_lin 1.52 (z1 end) → 0.16–0.31, velocity error 2.2 → 2.9 and rising, shuffle (feet low) 0.31 → 0.48, pelvis_low terminations 4 % → 27 %. Not a capability limit: the Rung 1 brains run 2.1–2.5 m/s (it 1500) and 2.9 m/s (it 2999) when commanded. MATT reached that recipe only after r2_v1…v7 on a brain that already had the envelope.
- **z2_v2** (18:22): new task `PoOlympic-Zombie-Rung2-Omni-Base` = MATT's r2_v1 recipe scaled (direct yaw commands, track_ang w 2.0 / coarse 1.0, 15 % stops, RUNG2_STAGES curriculum; no symmetry / accel cap / sprint focus / sharp kernel), same warm start (z1 it 1500), 2000 its. It 80: track_lin 1.13, velocity error 1.19 (MATT r2_v1 it 50: 1.08), episode 969/1000 — learning. The final recipe (`…-Rung2-Omni`) stays for a later fine-tune, as for MATT.
- check_task wiring sweep now runs without terminations (1 of 4 runs had reported a 1-channel "mismatch" when the single zombie fell mid-sweep and the reset moved ctrl); zombie base + MATT rung2 PASS.
- **z2_v2 gates** (zombie bars; 0 falls throughout, backward 20 m and brake ≈ 0.8 m < 1.24 m always pass): it 500 turntable 2.76–2.98 s, lin 29/50 segments, yaw 42/50 · it 1000 2.36–2.90 s, 33/50, 34/50 · it 1500 2.48–2.60 s, 37/50, 30/50 · it 1999 2.28–2.78 s, **39/50**, 22/50 → all 0/10 (turntable bar 2.36 s at 3.18 rad/s, lin < 0.157, yaw < 0.382). Linear tracking improves while yaw tracking degrades as the curriculum's wz reaches ±3.18 — MATT's r2_v1…v5 pattern, which the lateral-acceleration cap fixed.
- **z2_v3** (19:40): fine-tune = MATT's final recipe (`PoOlympic-Zombie-Rung2-Omni`: symmetric runner + mirror loss, full widened envelope from the start vx −1.18…3.14 / vy ±0.94 / wz ±3.82, lateral-acceleration cap 4 m/s², sprint fraction 0.3, track_lin std 0.236), warm start z2_v2 it 1500 (balanced), normalizer kept (count 1e8), action std floor 0.2, 2000 its; gates every 300.
- **z2_v3 stopped at it ~870:** the final recipe fixed yaw immediately (it 300/600: turntable 2.00–2.08 s < 2.36, yaw segments 50/50, 0 falls) but wrecked linear tracking again — median lin RMS sprint 0.15 → 0.68, turn 0.13 → 0.41, crab 0.13 → 0.32 (vs z2_v2 it 1500); lin segments 8/50, 7/50. Same signature as z2_v1: on the zombie, the sharp track_lin kernel (std 0.236) + 30 % sprints (2.0–3.1 m/s) hurt.
- **z2_v4** (~20:35): `PoOlympic-Zombie-Rung2-Omni-Sym3` = MATT's r2_v6 recipe (his first G1 10/10): symmetric runner, full widened envelope, lateral-acceleration cap, track_lin std 0.5, no sprint focus. Warm start z2_v2 it 1500 (same init as v3), 2000 its, gates every 300.
- **Stadium dressing (while training, Blender):** `SourceArt/Stadium/build_dressing.py` — crowd in home colours (the random rainbow shirts read as static from the broadcast camera; team blocks tiled into diagonal stripes), rings + POOLYMPICS on navy scoreboards and LED boards, facade rings/ribbon/4 gates, roof flag ring, cauldron, venue materials + stair nosings + start blocks, plaza/park/trees/skyline. 415 anchors identical, venues.json unchanged; GLB 4.0 → 6.5 MB. Unity: sky + re-check of the event cameras at the next editor session.
- **z2_v4 result (2000 its):** speed tracking improved (it 1500 43/50 segments) but yaw collapsed (it 300 19/50 → it 1999 9/50), turntable 2.5–3.6 s. Diagnosis (it 900, 1 s moving average): at straight speeds most yaw "error" is the zombie's gait wobble (0.25–0.45 rad/s, averages to 0.07–0.17), but hard turns reach only ~70 % of the commanded −1.5…−1.8 rad/s (bias +0.3–0.5) and crab drifts ~0.33 rad/s — consistent with the 70 % strength body.
- **Kept (user: "keep the best rung 2"): `zombie_rung2.onnx` = z2_v2 it 1500** — ranked over all 12 gated checkpoints (falls, then drills, then speed+yaw segments): 0 falls, drills 30/30 (brake, backward 20 m, joint speed), speed 37/50, yaw 30/50, turntable 2.54 s (bar 2.36). **Not a G1 pass (0/10 seeds)**: best available. Open ruling for later: zombie turn-rate envelope / step-averaged yaw scoring.

## 2026-09-28 late · Z7 parity gates on the zombie + mixed 8-lane meet (G0, G2–G6 PASS)
- **Tooling per body:** `reference.rollout(body=…)` (scene, gait clock, fingerprint of that body; meta `body`; standard shove × √λ = 0.393 m/s for the zombie, also in C# `Disturbance.StandardParityScript(speedScale)`), `closed_loop.g5_compare` uses the reference body's fall line (zombie 0.324 m) and contract, `meet.rollout_meet(scene_xml, layout_json, brains)` runs mixed meets (per-lane body, brain, defaults, gait clock via `contract.advance_phase_clock`). `compose_mixed.py` gained `meet8` (the Testbed_Rung1 layout; all-MATT compose == scene_meet8.xml). `compare_g6.py g0` checks each lane against its own body's solo scene. **MATT artefacts unchanged:** references, G6 plans and meet runs regenerate byte-identical; pytest 30/30.
- **Unity:** `ParityHarness` picks the testbed + contract from the reference body; `ZombieTestbed` builds `Testbed_Zombie.unity` (scene_zombie.xml, zombie contract + visual, bind 0.00 mm); `MeetTestbed.Build(source, layout, scene)` + `BuildMixed` → `Testbed_Mixed.unity`; `ConfigureG6` handles per-lane body/brain (sidecar fingerprint checked against the lane contract).
- **Results:** G0 zombie solo 0 mismatches · G2 ≤ 4.4e-16 · G3 ≤ 3.8e-6 · G4 1 s ≤ 1.3e-6 (zombie_rung0, zombie_rung2, 8 mixed lane refs; open-loop 5 s drift 0.1–0.8 on the zombie vs 0.01–0.02 MATT, not gated — self-collision contacts make the open-loop replay more sensitive) · G5 zombie_rung0 5 s drift 7e-7, torque 1.0000, cadence equal · G0 meet8_mzmzmzmz 8/8 · **G6 mixed 8/8** (MATT: stand, crab +, spin +, sprint 3.5; zombie: backward, crab −, spin −, walking turn at MATT's commands × √λ), 1 s drift ≤ 4e-6, height RMS ≤ 4e-4 m, torque 1.000–1.002.
- **Note — mixed-meet solver coupling:** the CPU meet diverges from the solo runs at ~1e-9 from tick ~160–200 in every lane, and the chaotic crab / spin lanes grow that to 3–6 cm by 5 s (MATT-only meet8: ≤ 1.6e-5). Lanes cannot collide (disjoint collision bits, checked); the coupling is the shared constraint solve (the zombie's self-collision adds contacts to the global solver's termination). All G5/G6 criteria hold with wide margin.

## 2026-09-29 night · Z6 zombie Rung 2 (z2_v5, z2_v6) + Event 13 flight prep (user asleep, 8 h of training as needed)
- **Unity closed** 00:05 for training (reopen when this block says training is over). TensorBoard already on :6006 (`training/runs`). Obsolete runs removed: `zombie_rung2/…_z2_v1` (stopped at it 430, unused) + `z2_init`.
- **Diagnosis before training:** z2_v3 it600 (sharp lin kernel + sprints) tracks yaw (median 0.10) but not speed (sprint lin 0.68, turn 0.41, backward drill 12.8 m); z2_v4 it1500 (Sym3, wide lin kernel, same init) tracks speed (0.10-0.14) but not yaw (0.56-0.77) → reward balance, not capability.
- **New report metric (not a pass criterion):** `yaw_rms_stride` = yaw error averaged over one stride of the body's gait clock (per-tick RMS includes the pelvis' natural ±5°/step rotation). Stride-averaged medians — v3 it600: turn 0.05, sprint 0.05, crab 0.03 (max 0.19); v4 it1500: 0.23 / 0.20 / **crab 0.58**; zombie_rung2 (v2 it1500): 0.15 / 0.18 / 0.13 (max 0.99, one hard turn). → v4's crab failure is a real sustained spin, not wobble; v3 steers precisely once the stride is averaged out. Data for the open "step-averaged yaw scoring" ruling.
- **z2_v5** (00:09): new task `PoOlympic-Zombie-Rung2-Omni-Sym3Yaw` (Sym3 + track_ang 2 → 3, std 0.5 → 0.35·√λ-scaled), warm start v4 it1500. It 300: 0/10, lin fails 4, yaw 10, turntable 2.97 s, crab stride-yaw 0.74 (worse than v4) → **stopped at it ~370**.
- **z2_v6** (00:26): same task, warm start **v3 it600** (the precise-yaw line) — recover speed with the wide linear kernel while keeping v3's steering. Gates at 200/400/600/900/1200/1499.
- **Event 13 prep:** `tools/flight_probe.py` — rung2.onnx at 3.5 m/s: airborne 18.6 %, 3.7 flights/s, mean 43 ms (max 60); 4.0 m/s: 52 ms. New task `PoOlympic-Matt-Rung2-Flight` (r2_v8 recipe + `flight_phase` reward w 1.0 above 2.2 m/s, 50 % sprint band; separate brain for Event 13), warm start `matt_rung2/r2f_init` = r2_v8 it600. Target ~80-100 ms flights at 3.5 m/s, no falls. Queued after the zombie run.
- Lesson: 4 CPU evals in parallel with training dropped it to ~3 its/min — only the sequential watcher runs alongside training.
- **z2_v6 stopped at it ~430:** yaw + turntable pass on every seed (1.98 s, yaw probe 0.18), lin + backward fail on every seed — `flight_probe` shows why: **it no longer runs** (cmd 2.0 → 0.14 m/s, ≥ 2.5 → drifts backwards). The v3 (sprint-focus) line had lost the running gait; "precise yaw" = barely moving. z2_v4 it1500 runs 2.82 m/s at the 3.14 envelope top.
- **Training-side diagnosis:** on the v4 line the yaw kernels paid ~nothing (track_ang 0.1-0.2 of 2-3; training yaw error ~4 vs MATT r2_v8 ~1.0 at track_ang 0.7) — misses are large, and mjlab's track_angular_velocity also counts roll/pitch rates (the rocking shuffle).
- **z2_v7** (YawGrad: yaw-only kernel 1.0·√λ-scaled + per-tick L1 −0.5, from v4 it1500): it200 yaw fails 2/10, turntable 1.97 s — but running collapsed (2.0 → 0.65 m/s). **z2_v8** (same terms on the stride-filtered yaw rate, EMA τ 0.5 s): same slide in training (track_lin 1.16 → 0.81 by it150) → stopped. **z2_v9** (v8 + lateral-accel cap 4 → 3 m/s²; the drill needs ≤ 3.0): it200 **first seed pass (1/10)**, runs 2.0 → 1.79 m/s — by it400 drifted to standing again (0.66 m/s). Every yaw-pushing variant trades speed away: the zombie style rewards (low feet, hunch, arms, wide stance) are all easier slow.
- **z2_v10** (v9 + track_lin 2 → 3, coarse 1 → 1.5; warm start v9 it200): **runs and turns** — it200/600: 2.0 → 2.0 m/s, 2.8 → 2.65 m/s, turntable 1.9 s, backward 20 m, brake 0.9 m, no falls; official 1/10 → 0/10 (lin fails 5-7, yaw 8-10). Segment detail (it200): speed medians 0.11 everywhere; the yaw fails are **sprint wobble** — per-tick 0.41-0.44 on segments whose stride-averaged error is 0.07-0.15. Stopped at it ~890 (plateau).
- **z2_v11** (02:00): v10 + yaw-wobble penalty (per-tick yaw rate − stride-filtered, squared, w −1.0), warm start v10 it600.
- **z2_v11 results** (Wobble: v10 + (yaw rate − stride-filtered)² penalty, w −1.0, from v10 it600): it100 1/10 · it200 2/10 (yaw fails 2) · it300 1/10 · **it400 8/10** (lin fails 2, yaw 0, turntable 2.21 s, yaw probe 0.25) · it600 3/10 · it800 4/10 · runs 2.0 → 2.06 m/s, 2.8 → 2.74. **it400 margin set (30 fresh seeds): 21/30 official, 23/30 stride-averaged, 0 falls**, lin median 0.104, yaw 0.243 (stride 0.086); 10 of 150 segments fail — almost all 2.4-2.85 m/s sprints missing the 0.157 m/s speed bar by 0.003-0.04. (Kept zombie_rung2.onnx = v2 it1500: 0/10.) Checkpoint-to-checkpoint swings (8 → 3 → 4) → consolidation run.
- **z2_v12** (03:44): Wobble task from v11 it400 with smaller PPO steps (desired_kl 0.01 → 0.004), entropy 0.005 → 0.001, action-std floor 0.05, save every 50, 600 its.
- **Event 13 flight brain (MATT):** r2f_v1 (`PoOlympic-Matt-Rung2-Flight`, per-step airborne reward w 1.0 above 2.2 m/s, 50 % sprint band, from r2_v8 it600): G1 rung2 it200 9/10, it400-999 10/10 (it800 8/10); flights at 3.0/3.5/4.0 m/s: 24/43/52 ms (rung2.onnx) → **59/68/72 ms** (it999), airborne 31 % at 3.5 — many short hops (4.4-4.8 flights/s). r2f_v2 (+ touchdown reward for the flight length above 40 ms, cap 200 ms, w 10; per-step 0.5): overshot into bounding — it200 190 ms flights, airborne 65 %, 3.5 m/s → 2.6 m/s, G1 3/10 → stopped. **r2f_v3** (cap 120 ms, w 4; from r2f_v1 it999): **it100 G1 10/10, flights 114/114/108 ms, speed 2.94/3.45/3.85**, airborne 43-47 %; it200/400 9/10 with the same flights → stopped at it ~430 (stable).
- **Zombie Rung 2 decision (04:30): `zombie_rung2.onnx` = z2_v11 it400** — margin 21/30 (stride 23/30), 0 falls; z2_v12 (consolidation: kl 0.004, entropy 0.001) 2-3/10, z2_v13 (track_lin std 0.35) 10-seed 3-6/10, margins 17/30 (it300) and 14/30 (it599). Old brain kept as zombie_r2_v2_it1500.onnx. Solo reference + `make_g6.py mixed g6mixed` regenerated: CPU G6 8/8. **Unity G2-G6 with the new brain: pending (Unity closed).**
- **Event 13 CPU race** (dash framework patched to 50 m @ 3.5 m/s, 8 traited runners): r2f_v3 it100 all finish, 15.72-16.00 s, no falls (rung2.onnx 15.42-15.58 s). r2f_v3 it100 30-seed rung2 margin 23/30 (rung2.onnx 29/30) — event-specific brain, rung2.onnx stays for the other events.
- **R3 get-up (MATT, Event 27), `tools/getup_probe.py`** (supine start, "up" = pelvis > 0.85 m & torso < 20° held 1 s):
  - getup_v1 (Rung 0 + supine resets, no fall terminations, linear height / uprightness + standing-tall rewards; warm start r0_v2 it1000, std floor 0.5): lay still to it 600, action std 0.5 → 0.04 — Rung 0's still_lin / still_ang / near_origin / posture pay ~3.6 per step for lying motionless.
  - getup_v2 (those four removed, get-up terms up, 50 % prone starts, entropy 0.01): still flat at it 200 (std 0.5 → 0.18).
  - getup_v3 (+ fading upward torso force, U(0, max) × body weight per episode, max 60 % → 0 over 1500 its): **sits up** (uprightness 1.95 / 2) by it 300, then settles sitting.
  - getup_v4 (from v3 it500: uprightness w 2 → 0.5, height progress 3 → 6, entropy 0.02): pelvis to ~0.5 m **on all fours**, torso flat.
  - getup_v5 (05:19, from v4 it600): height and uprightness only pay together (rise = height × upright², w 8), standing-tall 3 → 5.
  - getup_v5 result: back to **sitting** (rise ≈ 0.17 = pelvis low × torso vertical, std 0.5 → 0.25 by it 600) — sitting out-scores all fours on the product, and the feet-under transition to a squat (≈ 0.43) is not found. **Stopped at it ~650; R3 open.** Next step: a reverse curriculum (episodes also starting in squat / kneel / half-kneel poses — squat → stand first, then kneel → squat, sit → kneel) or a motion prior (R7). The fading torso assist + product reward are kept in the task code (Getup3-5).
- **z2_v14** (05:38): Wobble + 15 % sprint-band commands (vx 1.97-3.14, |wz| ≤ 0.76), from v11 it400 — targets the margin misses (2.4-2.85 m/s sprints 0.003-0.04 over the speed bar).
- z2_v14 result: 10-seed 3/3/4/4/3/2 at it 100-599 (lin fails 3-6, yaw 2-6) — no better than v11 it400. **Training over 06:04.** Obsolete runs removed (z2_v5-v8, z2_v12, getup_v1-v2, r2f_v2 + unused inits; runs 1.6 → 1.4 GB).
- **Unity re-opened 06:08 (training over).** New zombie_rung2.onnx re-validated: EditMode 36/36 (G2-G4 on the regenerated zombie_rung2 + g6mixed refs), **G6 mixed 8/8** in Testbed_Mixed (1 s drift ≤ 7.8e-6, same outcomes).

## 2026-09-29 · Event 8 → "30m All Fours" (user: race on all fours, survive falls; both bodies; natural falls only)
- **Tasks** (`poolympic/tasks/crawl_env.py`, on the get-up task: no terminations, lying resets, fading assist): torso ~80° tilt, pelvis in a crawl band (MATT 0.5 m, × λ), hands on the ground (contact sensor on the forearm subtrees), standing up penalised, vx 0.3-1.5 m/s (× √λ) + lane-keeping yaw.
- **crawl_v1** (MATT, from getup_v4 it600): perfect *static* all-fours pose by it200-400 (100 % on all fours, 0 m progress) — posture terms paid ~3.5 / step, the exp speed kernel ~0. **crawl_v2** (+ forward-progress reward speed/command, posture halved, speed curriculum): crawled **backwards** (−5 m). Cause: every "forward" was the pelvis x axis (mjlab body-frame speed kernel, contract yaw, yaw about the pelvis z axis) — on all fours it points at the ground and flips towards the feet once the hips are above the shoulders. **crawl_v3**: speed / progress / turn rate on `mdp.crawl_heading` (horizontal projection of pelvis x + z axes) and the world vertical — it200 crawls forward; **it800 = `crawl_matt.onnx`: 30 m at 1.2 m/s in 25.7-26.7 s (5/5), all fours 100 %, hands 75 %, lane 0.12 m** (it1200 same speed, lane 0.51 m).
- **Zombie:** zcrawl_v1 (warm start from MATT's crawl) **belly-slid** forward (progress 2.45 / 3, pelvis below the band, hands ~0 — MATT's actions are offsets from MATT's default pose). zcrawl_v2 (progress × (z / crawl height)², linear climb-to-height term, hands w 1, warm start zombie Rung 0): on all fours, crawls, but **ignored turn commands** (lane keeping asked +0.5 rad/s, it turned ~−0.1; track_yaw flat 0.48) → 11-14 m off its lane. zcrawl_v3 (turn tracking ×3 + L1 turn error, from v2 it700): it300 lane 0.13 m; **it800 = `crawl_zombie.onnx`: 30 m at 0.94 m/s (= 1.2 × √λ) in 31.7-31.8 s (5/5), all fours 100 %, hands 80 %, lane 0.05 m.**
- **Rules** (`poolympic/events/all_fours.py` ≡ Unity `TrackRaceEvent` mode AllFours): face-down start on the lane line (head to the finish), 3 s countdown to get on all fours, GO 1.2 m/s (MATT units; zombies × √λ like `PolicyRunner.BodyCommand`), lane keeping on the crawl heading (`Contract.CrawlSteerYawRate`), 30 m, rank by time; tumbles counted, never eliminate; standing up > 1 s = DQ. CPU mixed heats (MZMZMZMZ): all finish, MATT 25.7-26.2 s, zombies 31.6-32.0 s, 0 tumbles.
- **Unity:** `Event_30mAllFours.unity` (Event_30mDash deleted), hub + main menu rebuilt (catalogue, roster cards "Rung 0/2 · Crawl"), stadium label "08 30M ALL FOURS" (Blender, export_stadium), venues.json name. Mixed play-through via the menu: winner L7 (MATT) 25.52 s, heat over at 32.0 s. EditMode 36/36. Annotated UI before/after: `parity/ui/menu_allfours_before_after.png`; TensorBoard review charts: `parity/tensorboard/`.
- Note: size-scaled commands mean zombies always trail MATTs in mixed heats (0.94 vs 1.2 m/s) — a design choice to revisit (same absolute speed would sit at the zombie's crawl envelope top, 1.18 m/s).

## 2026-09-29 · Non-training block: Event 13 playable, D3 broadcast layer, betting / records / gauntlets, licensing (no training)
- **Event 13 Steeplechase Jog** (`track.py` mode `steeple` ≡ `TrackRaceEvent.Mode.Steeplechase`): 50 m at 3.5 m/s,
  MATT = flight brain `r2f_v3_it100.onnx`, zombie = `zombie_rung2.onnx`. First rule tried — +0.5 s per "grounded" stride
  (double support ≥ 20 ms after a 5 m take-off zone) — never fires: no brain ever has double support at running pace.
  And on finish time alone the plain Rung 2 brain wins (15.4-15.6 s vs 15.7-16.0 s: short hops are faster). Final rule:
  **ground time = finish time − hang time** (flights = both feet off ≥ 20 ms, counted per physics substep, `FootGait`).
  CPU 8-MATT heats: ground 8.9-9.3 s (air 6.6-6.9 s, 54-61 flights, longest 130-150 ms); rung2.onnx would score ~13.7 s,
  zombies ~19.8 s. Unity heats (4 seeds): winners 9.01-9.10 s ground, air 6.7-6.8 s — matches. pytest 31/31.
- **CPU commands now body-scaled in `_Lane.control`** (= Unity `PolicyRunner.BodyCommand`, identity for MATT): before, only
  the crawl race scaled zombie commands on the CPU, so mixed CPU heats of events 5, 9-12, 19, 22 ran zombies at MATT's
  speeds. MATT runs byte-identical; the mixed all-fours heat reproduces the logged times (MATT 25.9-26.2 s, zombie 31.6-32.0).
- **Odds from traits** (`tools/fit_odds.py`): 600 CPU heats (60 per event, MZMZMZMZ lineups from `compose_mixed.py`
  for turntable8 / crab8 / shaker8 / slalom8 too), Plackett-Luce fit on the full order (L2 0.5, L-BFGS). Weights
  [zombie, strength−1, latency, noise] + favourite win rate (in-sample, uniform 12.5 %): E01 ≈ 0 (23 %), E05 zombie +3.9
  (37 %), E08 zombie −6.4 / strength +6.8 (63 %), E09 zombie −10.7 (83 %), E10 −5.4 (38 %), E11 −2.3 (23 %), E12 zombie
  **+7.2** (65 %: the small body spins tighter), E13 −5.2 (12 %: MATTs within 0.3 s), E19 −6.8 (60 %), E22 ≈ 0 (8 %).
  Unity shrinks P 10 % towards uniform and clamps odds to 1.01-50 (raw fit gave 9000.00 for MATTs on the turntable).
- **D3 broadcast** (`BroadcastHud`, `BroadcastDirector`, `IBroadcastBoard`, `Commentary`, `Wallet`, `Records`, `Gauntlet`):
  see tasks.md D3. Lead-change calls needed a 4 s cooldown (neck-and-neck sprinters flipped the lead every second).
  Winner close-up first too tight (3.2 m) → 5.5 m ahead / 3.2 m aside. Played: Steeplechase (bet, result card, new
  record), mixed Turntable (zombies 1-4 as the odds said), 3-event gauntlet 13 → 12 → 05 (points carried, final podium on
  the menu). EditMode PoOlympic 36/36.
- **Licensing** → `docs/LICENSING.md` (Avaturn conditional, Hunyuan3D outputs barred in EU/UK/KR, Olympic marks, notices).
- **D1 pedestal parity:** G0 `scene_pedestal.xml` vs Unity Event_IronPedestal 0 mismatches; G5 one attempt (Rung 0, standard
  shove at tick 50 + cube at tick 100, 5 s): CPU reference `pedestal_rung0` (rerun bit-identical) vs Unity: drift 1.1e-6,
  torque 1.0000001 → PASS.
- **D6:** G0 on all event scene sets PASS (the shaker8 "== solo athlete" check now drops the lane-owned platform, which G0
  still compares Python vs Unity). Desktop perf: mj_step 0.33-0.40 ms for 8 athletes, 194 draw calls → no stadium merge.
- **Android (Pixel 9 Pro, Android 17):** parity APK = pedestal G5 attempt + `DeviceProbe`; ARM run vs the desktop CPU
  reference: drift 1.09e-6 @ 5 s — the ARM libmujoco.so + IL2CPP inference agree with x86 to the same 1e-6 as the
  editor. 8-athlete heat 57.6 fps (p95 16.9 ms, p99 33 ms), mj_step 0.30 ms. InputSystem warning not reproduced.
- **Event 23 The Trench Crawl (no training):** `tools/trench_probe.py` sweep of the ceiling underside with the Event 8
  crawl brains (16 m): 0.62 m 19/24 MATT finish, 22.6-57.8 s, 26 tumbles; 0.66 m 13/24; 0.70 m 23/24 but 17-58 s; **0.72 m
  32/32, 15.2-19.4 s**; 0.74 m 13.8-16.9 s; 0.78 m 13.9-14.4 s (no obstacle). Zombies unaffected (16.8-17.1 s) → 0.72 m makes
  mixed heats a contest (in the open 30 m crawl the zombie never wins). Posts on every lane line (0.61 m from lane centres)
  hooked 7/8 MATTs; posts 0.4 m outside the block still stopped lane 1 (drifts outwards) → ceiling overhangs 0.8 m, posts
  1.2 m outside: 95/96 finishes in 12 heats. Odds fit (60 heats): zombie +1.83, strength +4.3. Unity draws the physical
  ceiling translucent (TrenchGlass) so the broadcast cameras see the crawlers; `build_venues.py` updated to match (the
  current stadium GLB still has the old 0.60 m render-only trench, hidden in the event scene).
- TrackRaceEvent live standings: finishers by finish time (they stop past the line and were sorted by position).
- **Contract v4 stance-skills proposal** → `docs/CONTRACT_V4_STANCE_PROPOSAL.md` (11-dim skill block appended to the obs,
  one Rung S brain warm-started from rung2 recommended; 4 open decisions for the user). Game APK rebuilt with events 13 +
  23, D3 broadcast layer and gauntlets, installed on the Pixel 9 Pro (216 MB; menu `parity/android/menu_2026-09-29.png`).

## 2026-09-29 · User decisions
- Contract v4 stance skills: one shared Rung S brain (warm start rung2, zero-init new inputs), MATT first, proposed ranges,
  welded wrists / forearm-tip reach target.
- Zombie Rung 2: yaw tracking judged on the stride-averaged yaw rate (`yaw_rms_stride`). zombie_rung2 margin 21/30 → 23/30;
  the remaining misses are sprint speed precision (training).

## 2026-09-29 · Contract v4 + Rung S scaffolding (no training)
- v4 = v3 + 11-value skill block after the 84 obs (pelvis height, lift foot l/r, march Hz + knee lift, torso yaw/pitch,
  hand xyz + arm); v3 brains/references byte-identical (contract.json gains an additive skill_block). March cadence > 0
  drives the gait clock at zero velocity. Unity: random_brain_v4 G2 2e-16-level, G3/G4 PASS, closed-loop G5 drift 4.3e-6.
- Rung S task: skill mode per resample (20 % locomotion, 16 % each skill); stance skills stand still (is_standing_env);
  rewards w 3 per skill (squat std 5 cm, flamingo clearance 10 cm, march knee profile std 6 cm, torso std 9°, reach
  10 cm + 3 cm); height/posture/upright/phase-contact/pelvis-fall made skill-aware. Mirror symmetry extended and
  physics-checked. C1: obs 3e-7, clock 2e-7, torch vs numpy skill measures < 1e-6.
- Warm start rs_init = r2_v8 it600 + zero input columns (actor(obs95) == actor(obs84) exactly); 2-iteration smoke run
  loads and trains (removed). G1 drills (evaluate_stance.py) run; the random v4 brain fails all (as it should).

## 2026-09-29 · Brain confidence (broadcast feature 5, no training)
- `tools/export_critic.py`: the PPO critic of each event brain → `parity/brains/<brain>.critic.onnx` (84 obs → value;
  critic obs normalizer = actor's 84 terms, no privileged inputs). ORT vs rsl_rl math ≤ 1e-5. Actor brains untouched.
- `tools/fit_confidence.py`: CPU rollouts on the mixed track8 scene (8 seeds × 150 s per group, envelope / sprint / crawl
  commands + random shoves 0.3-2.4 m/s × √λ) → logistic P(no fall within 2 s) per brain → `Models/confidence_model.json`.
  Value alone ranks danger for standing / crawl brains (AUC r0 0.72, zombie_rung0 0.66, crawl 0.64-0.66) but not for the
  running brains (rung2 0.49, zombie_rung2 0.52: V scales with the command). Adding the value drop against its 1 s EMA and
  |cmd| fixes it: **AUC rung2 0.76, zombie_rung2 0.62, r2f_v3 0.84, r0 0.72, zombie_rung0 0.69, crawl_matt 0.68,
  crawl_zombie 0.64** (report `parity/confidence/report.json`, reliability tables inside). Constant features are dropped
  from the fit (the crawl's fixed 1.2 m/s command made the IRLS weights blow up).
- Unity: PolicyRunner runs the critic every 5th control tick (10 Hz, staggered per lane) on the brain's own observation;
  never feeds ctrl. EditMode parity gates (G2-G4) unchanged: PASS.


## 2026-09-29 late · AUDIT + recipe v5 STAGED (no training run) — throughput, bio-realism, MATT-bio body
- **Throughput** (`tools/bench_env.py`: stepping only, r2_v8 drives the envs; `parity/bench/bench.jsonl`). Rung 2 recipe
  at 4096 envs = 80k env-steps/s (the logged 1.8 s collection per iteration). MuJoCo is only ~36 % of a step; the rest
  is eager torch managers (rewards 16 %, commands 11 %, events 10 %, observations 9 %), which cost the same per step at
  any env count. Solver already exits after 2.3 iterations; peak contacts 18 / world, peak nefc 117-132 (njmax was 500).
  Knobs: 8192 envs 109k (1.35x) · 1 pool cube 105k (1.31x: the 3 parked cubes = 12 resting contacts + 18 dofs per world)
  · buffers alone ~1.07x · ls_iterations 10 no gain (dropped: solver options stay == Unity) · 12288 envs only +9 % over
  8192 · all together 118-134k (1.5-1.7x; runs were noisy: 3 Unity editors, Blender and a parallel session's APK build
  were active). The first 43k baseline was a cold run — ignore it.
- **Bio-realism audit** (`tools/bio_probe.py`, CPU, `parity/bio/`): rung2.onnx is already human-range — per-foot GRF p95
  1.4 BW walking / 1.9 running (max 2.6), Hill torque-velocity violations 0.03 %, torque-cap saturation ~0, no joint over
  18 rad/s, no arm-through-body clipping in the locomotion drills. r2f_v3 (flight brain) lands harder (max 3.4 BW) and
  clips 1.6 cm arm-into-hip on 29 % of standing frames. Clipping shows up in falls / odd poses (r0 brain mid-fall 9 cm).
- **MATT full self-collision check:** no geom pair touches at the T-pose, the default stance or a ±40° running arm swing;
  only arms pressed past −85° elevation reach the hips (−95°, the joint limit, sinks the forearm 6 cm into the pelvis
  today) → no new excludes needed.
- **Staged body `mattbio`** (`POOLYMPIC_BODY=mattbio`; `bodies.BIO_TORQUE_CAPS`, `assets/mattbio.xml`,
  `scene_mattbio.xml`, `parity/contract_mattbio.json`, fingerprint): MATT + full self-collision + joint- and
  direction-specific torque caps (e.g. ankle dorsiflexion 220 → 60 Nm, inversion/eversion 220 → 60/45, hip rotation
  280 → 80, knee flexion 280 → 150, trunk twist 200 → 80). matt.xml / scene_matt.xml byte-identical. **The current brains
  pass G1 on it with no retraining: rung2.onnx 10/10 (turntable 2.6 s, brake 1.25 m, 0 falls), r0_v2_it1000 10/10**;
  only ankle torques would bind (dorsi 1.6-1.9 % of running frames, inversion 8 % while spinning). Not adopted: DESIGN
  §2 change → needs the user's OK, then Unity re-import + G0/G2-G6 + event-scene regeneration.
- **Recipe v5 (staged tasks):** `matt_env.fast_sim` (8192 envs, 1 pool cube, nconmax 64 / njmax 256 — training copy only,
  not for lying tasks), `matt_ppo_v5_cfg` (8 mini-batches = same 24.6k mini-batch, checkpoints every 50),
  `add_bio_rewards` (mechanical power Σ|τq̇| w −2e-4 ≈ −0.13 per step on the Rung 2 mix vs torques −0.24; Hill
  envelope and > 3 BW foot-impact guards). `PoOlympic-Matt-RungS-Stance` now = v5 + `posture_skill_idle` (joints a
  skill does not use hold the default pose) — C1 PASS (obs 3.1e-7, clock 3.1e-7, skill measures ≤ 5.6e-7, wiring 0),
  97.5k steps/s at 8192. New: `PoOlympic-Matt-Rung2-V5`, `PoOlympic-MattBio-{Rung0-Stand,Rung2-Omni,RungS-Stance}`.
- **Not done / no gain:** pruning the 35 registered tasks saves ~0 s (registration is free; the 11.8 s is the mjlab
  import) → kept for log reproducibility. Launch checks: `tools/preflight.ps1` (GPU temp, competing apps, power, TensorBoard,
  run sizes; reports only). At audit time the idle GPU sat at 77-84 °C and two TensorBoard processes were running.
- **Launch (when approved):** `pwsh tools/preflight.ps1` → close Unity/Blender →
  `uv run train PoOlympic-Matt-RungS-Stance --log-root runs --agent.resume True --agent.load-run rs_init
  --agent.load-checkpoint model_0.pt --agent.run-name rs_v1` (no `--env.scene.num-envs 4096`: v5 defaults to 8192).

## 2026-09-30 · DECISION — crowd contact in every event scene + 6 events laid out for collisions (no training)
- **Change:** user: "more interaction and collisions between the 8 players … do all 6". Event scenes get per-lane crowd
  bits (`build_mjcf.crowd_bits`: lane k owns bit 16+k, collides with every other lane's; own parts unchanged); the G6
  testbed `scene_meet8` stays lane-isolated. Venues (`build_venues.py` crowd constants, Blender pass `build_crowd.py`,
  Stadium.glb re-exported): 1 one iron beam, 5 one shaker floor (8 × mass/stiffness/damping → still 2.0 Hz, ζ 0.2),
  8 crawl lanes 1.1 m (own scene `crawl8`; track8 now from venue 22), 11 1.4 m lanes, 12 ring, 19 green break line.
  Rules: 11 mirror slalom (neighbours weave in mirror image) at 1.6 m/s; 12 DQ radius 0.75 m; 19 lane break at 15 m
  (field squeezes to SQUEEZE 0.5 of its distance to lane 1, merge ≤ 12°); 8 DNFs rank by distance covered.
- **Tuning (CPU heats, all-MATT unless noted; "pair-s" = seconds of athlete↔athlete contact summed over pairs):**
  - 01 beam pitch 0.65 / 0.75 / 0.9 m: heats 35-39 / 32-46 / 32-44 s (old pedestals 27-35 s), pair-s 49-90 / 7-36 / 2-10
    → **0.7 m**. Mixed MATT/zombie: 32-41 s, pair-s 0.4-17.
  - 05 floor grid 0.9 / 0.8 / 1.0×1.2 m: survivors 0-2 / 1-3 / 1-3 (old 0-1), pair-s 5-12 / 12-23 / 2-3 → **0.8 m**.
    Mixed: 2-5 survivors, pair-s 5-13.
  - 08 crawl pitch 1.22 / 1.15 / 1.1 / 1.0 / 0.85 / 0.7 / 0.6 m: below 1.1 m crawlers interlock (2-5 DNF per heat,
    pair-s 80-160); 1.1 m: 3 of 4 heats 8/8 finish (pair-s ~30), 1 tangle → **1.1 m** + DNF ranked by distance.
  - 11 every fall follows a contact: 2.2 m/s floors the Rung 2 brain after any bump (mirror at 1.22-2.0 m lanes: 1-6
    finish); no-contact control (3 m lanes) 8/8. **1.4 m lanes at 1.6 m/s**: 6-8 finish (mean ~7), pair-s 4-7, pole
    clips 0-21 (neighbours knock each other into poles); 1.8 m/s: 4-7 finish. Mixed lineups barely touch (zombie narrower).
  - 12 ring pitch 0.65 / 0.75 / 0.8 / 0.9 m: max drift 0.2-1.7 / 0.17-1.13 / 0.14-0.78 / 0.07-0.39 m (old 0.03-0.12), no
    falls → **0.75 m, DQ at 0.75 m**. Mixed: drift 0.03-0.6.
  - 19 one shared inside line (GAP 0.5-0.65 m to runners alongside, look-ahead 3 m): rear-end pile-ups, 2-6 of 8 down,
    slower zombies run over (1-6 falls mixed) → rejected. **Squeeze 0.5**: 1-3 of 8 MATTs down, pair-s ~4, mixed 0 falls;
    0.45: 2-5 down.
- **Verification:** `tools/check_crowd_contacts.py` — every event scene, all-MATT + mixed: 16,184/16,184 cross-athlete
  part pairs pass the filter and produce a real `mj_collision` contact, no support / self-collision change (negative
  control scene_meet8: 0/16,184); `tests/test_crowd_contacts.py` 17/17, `tests/test_events.py` 10/10;
  `compose_mixed --verify` all PASS; Unity EditMode `CrowdContactTests` 11/11 built event scenes.
- **Decision:** keep. The brains never saw another athlete in training; a "crowd" training run (2-4 colliding athletes
  per env) is the next step if contact falls feel too frequent (19, 11).

## 2026-09-30 00:00-00:30 · 8 h training block — INTERRUPTED after 26 min (session ended, background jobs died)
- Done first ("do all"): GPU-sync / per-step lookup fixes in our task code (Rung 2 v5 105k → 137k steps/s, Rung S 97k →
  111k at 8192 envs; 1.7x the old 4096-env setup), `tools/{watch_gate,plain_init,train_queue.sh}`, reverse-curriculum
  get-up task (`tasks/getup_env.py`, PoOlympic-MattBio-Getup-Rev: squat → kneel → sit → lying), mattbio crawl task
  (crawl_matt.onnx on mattbio: 5/5 but 29.7-43.9 s and 13.4 m lane drift vs 0.12 m), mattbio flight task, rsl_rl
  git-diff UTF-8 crash patched, 28 obsolete runs moved to training/runs_archive (reversible).
- **rs_v1** (PoOlympic-MattBio-RungS-Stance from rs_init, 8192 envs): 00:04-00:30, stopped at it 213 when the Claude
  session ended (no traceback; checkpoints 0-200 kept). 4.3-7 s/it: GPU in SW thermal slowdown at its 87 °C target
  (~40 W of 110 W) and a parallel fit_odds.py (12 workers) pegged the CPU for the first 15 min. It 20 → 213: episode
  length 498 → 996, skill rewards flamingo 0.024 → 0.33, torso 0.006 → 0.15, march 0.05 → 0.11, squat 0.02 → 0.05,
  **reach ~0 (not learning)**.
- The queue (r2bio_v1, r0bio_v1, r2fbio_v1, crawl_bio_v1, getup_rev_v1) never started. The it-200 gate hit two watcher
  bugs: Rung S report JSON (numpy int64) and Rung 2 drills fed 84 obs to a 95-obs brain.
- mattbio → MATT promotion still waiting for poolympic-c1 (it owns build_mjcf / event scenes / Unity event code).

## 2026-09-30 09:45-11:xx · Rung S rs_v2 → rs_v3 (MATT, mattbio body)
- **Watcher fixes:** `eval_cpu.py` JSON default for numpy scalars; `evaluate.Sim` zero-pads the obs to the brain's
  input width (v4 brains in the Rung 2 drills = zero skill block).
- **rs_v2** (from rs_v1 model_200, + `skill_reach_coarse` σ 0.50 m, w/2): rs_v1's reach rewards (σ 0.10 / 0.03) paid
  ~0 from a resting hand 0.5-1.2 m away → reach stuck at 0 for 213 its. With the coarse term: reach_coarse 0 → 0.21,
  precise reach 0 → 0.020, torso 0.14 → 0.33; squat 0.06 → 0.08, march 0.10 flat, flamingo 0.34 flat; action std
  rising 0.47 → 0.54 (entropy 12 → 14). G1 S it400 / it600: torso 6/10, others 0/10 (median squat err 0.29 / 0.34 m,
  flamingo stance slip 0.40 / 0.07 m, march no knee lift, reach 0.30 / 0.25 m); Rung 2 G1 9/10 → **7/10**. Stopped at
  it 600 (10:37). Throughput 77k → 14k env-steps/s while RealityScan + Maestro ran on the laptop (09:55-10:25).
- **rs_v3** (from rs_v2 model_600, 600 its, entropy_coef 0.005 → 0.0025 via CLI): squat and march had reach's flaw
  (σ 0.05 / 0.06 m vs 0.3 m errors) → `skill_squat_coarse` / `skill_march_coarse` σ 0.25 (w/2); flamingo reward × 
  exp(−|stance-foot xy speed| / 0.05 m/s) (rs_v2 was paid for contact while sliding). Gates at 800 / 1000 / 1199.
- Review page with TensorBoard charts: `parity/tb/rs_v2/review.html` (tool `training/tools/tb_shots.py`).
- **rs_v3 it800 gate (11:14):** G1 S torso 7/10, others 0/10; Rung 2 8/10. Median drill: squat err 0.265 m (v2 0.337),
  reach 0.248, torso 0.093 rad, flamingo slip 0.141 m / 46 touch ticks (v2 it600 0.07 / 9: the slip gate did not
  help the drill), march **no knee lift**. The march coarse kernel (σ 0.25) pays ~73 % for standing still (lift 0.15-0.30
  → err² ≈ 0.02) → no reason to march. Stopped at it 800.
- **rs_v4** (from rs_v3 model_800, 600 its, entropy 0.0025): `skill_march_coarse` → `skill_march_lift` (swing-knee rise /
  commanded lift while that knee's profile is > 30 % up; standing pays 0), w/2. Gates at 1000 / 1200 / 1399.
- **rs_v4 died at it ~966 (12:04)** with its Claude session (no traceback; gates never ran). Gate on the last checkpoint,
  **it950**: G1 S torso 5/10, squat / flamingo / march / reach 0/10; Rung 2 8/10, 0 falls. `skill_march_lift` flat at
  ~0.015 the whole run; drill **mean knee peak 0.0 m on all 10 seeds** (cadence 0 Hz), squat worst err 0.18-0.31 m,
  flamingo 154 touch ticks / 0.18 m slip. Diagnosis: every leg skill is paid in task space only (pelvis height, knee /
  foot rise) — nothing points the policy toward the pose until it is already close.

## 2026-09-30 13:37 · rs_v5 — joint-space leg guides (MATT, mattbio body)
- **Change:** `skill_mdp.leg_pose_target` = plausible hip / knee / ankle angles for the commanded skill (hand-set
  geometry, no mocap): squat from an FK table (flat feet: shin lean = knee − hip = ankle, capped at 24° for the 25° ankle
  limit → 0.4 m = hip 83° / knee 107° / ankle 24°), march = swing thigh rotated so the knee rises by the profile, shin at
  its default lean, flamingo = lifted leg hip 0.6 / knee 1.2 rad (foot ≈ 0.14 m up). Rewards (w/2 each):
  `skill_leg_progress` = 1 − RMS err / RMS err of the default pose (standing pays 0, target 1, linear between) and
  `skill_leg_pose` exp(−mean err² / 0.25²). C1 PASS.
- **Run:** from rs_v4 model_950, 600 its (→ 1549), entropy 0.0025, 8192 envs; 8.8 s/it (GPU 84 °C idle before the start
  → throttled). Gates (watch_gate S,2) at 1150 / 1350 / 1549. Obsolete intermediate rs_v1-v4 checkpoints (not warm-start
  sources) moved to `training/runs_archive/mattbio_stance/` (reversible).
- **rs_v5 died at it ~1068 (13:55)** with its Claude session — third run lost this way. By then the new guides were
  learning (it 950 → 1068: leg_progress 0 → 0.29, leg_pose 0 → 0.30, squat 0.095 → 0.14; rs_v4 was flat).
- **Fix: `training/tools/detached_run.ps1`** — training + watch_gate started through WMI (`Win32_Process.Create`), so the
  job's parent is WmiPrvSE, outside the session's process tree. **rs_v5b** (14:37): from rs_v5 model_1050, 500 its
  (→ 1549), entropy 0.0025, same gates (prefix rs_v5 → `parity/watch_rs_v5.jsonl`) at 1150 / 1350 / 1549.
  2.7 s/it at the start (rs_v5: 8.8 s/it on a throttled GPU). Unity editor open alongside (opened 14:32, left running).
- **rs_v5b gates** (median of 10 seeds; knee lift from a 10 s march probe at 1.3 Hz / 0.25 m):

  | it | G1 S | squat err | march knee lift | flamingo touch / slip | torso | reach | Rung 2 |
  |---|---|---|---|---|---|---|---|
  | rs_v4 950 | torso 5 | 0.258 m | 1.8 cm | 154 / 0.184 m | 5/10 | 0.227 m | 8/10 |
  | 1150 | torso 7 | 0.231 | 3.7 | 51 / 0.134 | 7/10 | 0.213 | 9/10 |
  | 1350 | torso 5 | 0.201 | 5.2 | 14 / 0.095 | 5/10 | 0.209 | 9/10 |
  | 1549 | torso 4 | 0.139 | 6.0 | 50 / 0.165 | 4/10 | 0.236 | 8/10 |

  0 falls in every gate. Squat and march lift finally trend (training: squat_coarse 0.146 → 0.21, march_lift 0.017 →
  0.031); flamingo training reward flat at ~0.10 in every run (drill is one command per seed → noisy). Torso trades off.
  Known guide flaw: leg_progress floors err0 at 0.1 rad, so near the march profile's zero crossings standing still pays
  → if rs_v5c stalls, pay march progress on the swing leg only / lift curriculum. Review: `parity/tb/rs_v5/review.html`.
- **rs_v5c** (15:11, chained detached via `runs/chain_rs_v5c.ps1`): from rs_v5b model_1549, 1000 its (→ 2548), same
  recipe, gates at 1800 / 2050 / 2300 / 2548. GPU at its 87 °C limit (~40 W), ~4 s/it → ~70 min.
- **rs_v5c gates** (G1 S per drill · Rung 2): it1800 squat **10/10**, torso 5, rest 0 · R2 7/10 (1 fall) | it2050 squat 10,
  torso 7 · R2 5/10 (tracking_lin misses, 0 falls) | it2300 squat 10, torso **8** · R2 8/10 | it2548 squat 10, torso 5,
  **march 1/10** (first ever) · R2 6/10. Squat solved from 1800 on; torso noisy 5-8; flamingo / reach 0/10. Rung 2 speed
  tracking erodes with only 20 % locomotion envs → next run: locomotion share up (0.20 → ~0.35), march swing-leg-only
  progress, flamingo stance-slip penalty; reach needs a new approach. Best checkpoint so far: **it2300**. Cooler Boost
  measured: 2.9 s/it vs 3.7-4.1 s/it on Auto fans (~30-40 % faster).
- **Unity:** `Testbed_Stance.unity` (StanceTestbed.Build / Configure, StanceSkillDemo) plays rs_v5 brains on mattbio:
  squat and torso aim visible, reach short of the target.

## 2026-09-30 20:27 · rs_v6 — locomotion share, swing-leg march progress, flamingo slip penalty (MATT, mattbio body)
- **Change** (task `PoOlympic-MattBio-RungS-Stance-v6`, `stance_env.matt_stance_v6_env_cfg`): from rs_v5c **it2300**
  (best so far), 1000 its (→ 3299), entropy 0.0025, 8192 envs. (1) mode_probs locomotion 0.20 → 0.35 (skills 0.13 each):
  Rung 2 G1 eroded 9 → 5-8/10 in rs_v5c. (2) `skill_leg_progress_v6`: march progress on the swing leg only, while its
  profile is > 30 % up (rs_v5 averaged both legs, err0 floored at 0.1 rad → standing still earned march progress near
  the profile's zero crossings). (3) `skill_flamingo_slip` −2 × stance-foot xy speed (capped 1 m/s) in flamingo mode
  (skill_flamingo's slip factor only scales a reward that is ~0 while the lifted foot is down). Reach unchanged.
- Launch: WMI could not start the Store `pwsh` (ReturnValue 9 / silent exit) and `-File` does not group single-quoted
  args → `detached_run.ps1` header documents powershell.exe + double quotes; LoadRun/Checkpoint now optional (scratch runs).
- Archived rs_v5b / rs_v5c intermediate checkpoints (kept the gated ones + all TensorBoard logs) to runs_archive.
- **Gates** (G1 S per drill · Rung 2; 0 falls everywhere; 1000 its in 50 min, 2.5-2.9 s/it):

  | it | squat | flamingo | march | torso | reach | Rung 2 |
  |---|---|---|---|---|---|---|
  | rs_v5c 2300 (start) | 10 | 0 | 0 | 8 | 0 | 8/10 |
  | 2550 | 10 | 10 | 0 | 5 | 0 | 4/10 |
  | 2800 | 10 | 0 | 2 | 2 | 0 | 6/10 |
  | 3050 | 10 | 0 | 4 | 4 | 0 | 5/10 |
  | 3299 | 10 | 0 | 4 | 5 | 0 | 6/10 |

- **March: the fix worked.** 3299: every seed lifts 83-97 % of the commanded lift (0.16-0.24 m; rs_v5c 2300: 5 seeds no
  lift, the rest ~0.12 m). 5 of the 6 misses are pelvis drift 0.43-0.90 m (bar 0.30 m), all at cadence >= 1.3 Hz; one is
  cadence 6 % fast. Nothing pays staying on the spot -> next: drift penalty in march mode. Squat worst err 2.5 -> 1.5 cm.
- **Flamingo: not fixed, and the drill is ONE trial.** `drill_flamingo` draws no random command, so all 10 seeds are
  identical: "10/10" at 2550 = one pass (0 touches, 7 mm slip), then 55-176 touch ticks / 14-33 cm slip. Training
  `skill_flamingo` share-corrected 0.07-0.08 vs rs_v5c 0.09, `skill_flamingo_slip` flat at -0.048 (the penalty is not
  being reduced). Next: randomise the drill's start (settle time / small push) before trusting its count.
- **Locomotion keeps eroding** despite 35 % walking envs: track_ang 0.99 -> 0.83, track_lin 1.23 -> 1.18 inside the run,
  Rung 2 G1 4-6/10 (start 8/10). Cause not established (candidates: in-place marching drives the same gait clock; the
  walking-away habit of the march). Torso 8 -> 2-5/10, reach median 0.22 -> 0.25 m.
- Skill terms pay per env in that mode: 13 % share (was 16 %) -> the same skill scores ~19 % lower on the charts.
- **Checkpoints:** 3299 = squat + march; 2550 = the only flamingo pass; rs_v5c 2300 still best for torso + Rung 2.
  Review: `parity/tb/rs_v6/review.html`.
- Side check: `rs_v5_it2300.onnx` on the plain MATT body (not mattbio) passes the squat drill 5/5 (worst err 1.3-2.4 cm,
  same as mattbio) → Event 3 can run in the MATT event scenes before the mattbio promotion.

## 2026-09-30 21:00 · GRANDMA body (Phase G) — rig, MJCF, Rung 0 task
- Source scan has no skin → `SourceArt/Grandma/rig_grandma.py` (Blender 5.2 via MCP): weld UV-seam islands, landmark
  skeleton (22 Mixamo bones, symmetric), bone-heat weights, legs straightened + baked as rest, 1.600 m.
- `bodies.BODIES["grandma"]`: 65 kg, λ = 0.871, strength 0.6, self-collision, stance abdomen 10° / hip 14° / knee 24° /
  ankle 10° / hip_abd 3°. Joint torque caps (× 0.425 of MATT's): knee / hip 119 Nm, ankle 93, abdomen 85, shoulder 34,
  elbow 30 Nm. MJCF 65.000 kg, root z 0.678 m, fall line 0.390 m (0.55 × 0.678 / 0.955).
- `grandma_env.grandma_rung0_env_cfg`: MATT Rung 0 + zombie-style Froude scaling; style: `stoop` (torso pitch 10°,
  σ 10°, w 1) replaces upright, `steady_stance` (feet 0.26 m apart, w 0.5). C1 PASS.
- **g0_v1** chained after rs_v6: from scratch, 1500 its. **FAILED, my launch mistake: 1 environment** (this task has no
  fast_sim default; the zombie runs passed `--env.scene.num-envs 4096` on the CLI). 0.5 s/it, episode length 40 -> 47
  ticks in 260 its, every episode ending in torso_tilt. Run deleted. A zombie probe launched the same way is equally flat.
- While chasing that (before finding the env count) two body changes were made on their own merits; neither is proven
  necessary: (1) **`Body.stiffness` = 1.0** for GRANDMA (`gain_scale`): kp / kv no longer scaled by the 0.6 strength,
  only the torque caps are. With kp x 0.6 her hip stiffness (127 Nm/rad) was below the gravity gradient of her trunk
  (~150 Nm/rad) and the trunk folded to 62 deg in 0.8 s under a passive hold. Zombie MJCF byte-identical. (2) **stance**
  hip 14 -> 17 deg, ankle 10 -> 7 deg (pelvis level, COM 43 % -> 32 % of heel -> toe; MATT 34 %, zombie 39 %).
  Passive hold now: trunk sags ~14 deg at the waist, falls at ~1.2 s (zombie passive: 51 deg at 0.5 s; MATT ~2.3 s) —
  the policy has to balance actively, as for the others. Joints not saturated (abdomen 50 / 85 Nm, hip 32 / 119).
  New fingerprint de971d49, contract_grandma.json regenerated, test_model 13/13.
- Probe with 4096 envs (100 its): episode length 14 -> 421 ticks (zombie z0_v1: 371 at it 100) -> learning normally.
- **g0_v2** (21:35, detached): `--env.scene.num-envs 4096`, 1500 its, 1.8 s/it (~45 min), gates rung 0 at 500/1000/1499
  (prefix grandma_r0).

## 2026-09-30 21:50 · Event 3 Deep Squat Endurance playable (Unity)
- Rung S brains carry the mattbio fingerprint, so the event scene is all-mattbio (`compose_mixed.py squat8 mattbio x 8`);
  `BodyAssets("mattbio")` = contract_mattbio + MATT's visual; LaneLineup / BuildMeetScene fall back to the lane's own
  body when a scene has no MATT.
- Rules bug found with the stronger brain: the fall line returned to standing height at the down -> up switch while the
  pelvis was still 43 cm down (5 athletes "FELL" at exactly 34.9 s with rs_v6_it3299; rs_v5_it2300 never squatted
  that deep). Now the rep's depth lowers the line for the whole rep (Python + C#).
- Brain `rs_v6_it3299.onnx` (squat worst err 1.5 cm). CPU, 6 seeds: winners 96-104 pts, 1-4 finishers per heat, exits
  13-41 s (all STEPPED: feet creep > 20 cm). test_events 11/11.
- Unity: `Event_DeepSquat.unity` built through the Unity CLI, play-through seed 1: S5 wins 101.0 pts (DONE) after
  40.9 s; hub + main menu rebuilt (12 playable events). Captures `parity/e3/squat_{live,down,result}.png`.
- Not done: G0 fingerprint check of the squat8 scene and a G5-style parity run of one attempt; EditMode test suite
  not re-run after the LaneLineup / EventScenes changes.

## 2026-09-30 22:35 · GRANDMA Rung 0 PASSED (g0_v2) · Android deploy with Event 3
- **g0_v2** (4096 envs, from scratch, 1500 its, 58 min; 1.8-2.4 s/it with the Unity editor open and an Android build
  running alongside): episode length >= 900 at it 119 (zombie z0_v1: 214, MATT r0_v2: 210); reward plateau ~87 from it 600.
- **G1 rung 0 (grandma bars): it 500 9/10 · it 1000 10/10 · it 1499 10/10**, 0 falls, every hit recovered. 30 other
  seeds (first-seed 2000): 24/30 for both 1000 and 1499 (zombie 25/30, MATT 29/30); all misses are foot excursion
  0.44-0.52 m vs the 0.435 m box (recovery steps), 0 falls. Largest lean after a hit 9.6 deg (1000) vs 6.8 deg (1499)
  -> **`parity/brains/grandma_rung0.onnx` = g0_v2 it 1499** (re-verified 10/10).
- Quiet stance (5 s): torso pitch 9.1 deg (target 10), foot width 0.254 m (0.26), knees 24 deg, pelvis 0.673 m;
  torques 24 / 85 Nm waist, 18 / 119 Nm hip. Relative effort (torques term) -0.155 vs MATT -0.049, zombie -0.204.
- Review: `parity/tb/g0_v2/review.html` (charts vs z0_v1 and r0_v2 + a stand / shove / recover render). Intermediate
  checkpoints archived to runs_archive/grandma_rung0 (kept 500 / 1000 / 1499).
- **Android:** `AndroidBuild.Build` -> 236 MB APK, 15 scenes, 5.0 min; installed on the Pixel 9 Pro. Menu 60 fps with
  "03 Deep Squat Endurance"; one Event 3 heat on the device: 60 fps, S6 wins 101.2 pts (DONE) after 40.9 s (editor,
  same seed: S5 101.0 — no ARM parity run for this scene yet). Captures `parity/android/{menu_e3,e3_live,e3_result}.png`.
  Seen, not fixed: the bottom bar tagged S3 "LEADER" while the standings listed S5 first.
- Next for GRANDMA: Unity testbed (visual binding, G0 / G2-G5), roster card + lineups, then Rung 2 (short careful steps).

## 2026-09-30 23:10 · 8 h+ training block (user: "train for at least the next 8 hours")
- **Queue** `tools/train_queue.sh` (detached through WMI + Git bash; reads the next job from `runs/queue.txt`, log
  `runs/queue.log`; stop = create `runs/queue.stop`). New: `scratch:<experiment>` inits and a `prep` field (a command
  that builds the job's warm start from the previous run, so dependent runs chain without a live session).
  Unity editor + MuJoCo viewer closed for the block (GPU idle 81 C / 57 W with them open, 76 C / 24 W after).
- **Order** (sequential, ~8.6 h): r2bio_v1 (mattbio Rung 2, 400) → **g1_v1** GRANDMA Rung 1 from scratch (2000, gates 1)
  → **g2_v1** Rung 2 base, warm_start from g1_v1 it1500 (2000, gates 2) → **g2_v2** Rung 2 final recipe, plain_init from
  g2_v1 it1500 (2000, gates 2) → **rs_v7** (1500, gates S,2) → r0bio_v1 (300) → r2fbio_v1 (300) → crawl_bio_v1 (600) →
  getup_rev_v1 (1500).
- **GRANDMA locomotion tasks** (`grandma_env.py`, zombie chain on her scale, λ 0.871: speeds × 0.933, rates × 1.072):
  `-Rung1-Run` (MATT rung 1 + `careful_steps` = feet_low, foot box centre <= 0.09 m, w 0.3), `-Rung2-Omni-Base`
  (MATT r2_v1 recipe, stage 0 vx −0.93…2.80, vy ±0.47, wz ±1.07), `-Rung2-Omni` (zombie z2_v11 recipe in one function:
  symmetric runner, full envelope, lateral-accel cap 3, track_lin std 0.5 w 3 + coarse 1.5, stride-filtered yaw terms,
  yaw-wobble penalty). `num_envs` = 4096 is now in the cfg. Smoke 3/3, C1 PASS (rung2). The warm-start checkpoints
  (it1500) are a fixed guess copied from the zombie; to be changed in queue.txt if the gates say otherwise.
- **rs_v7** (`matt_stance_v7_env_cfg`, from rs_v6 it3299, entropy 0.0025 in the task's rl cfg): `skill_spot`
  exp(−(pelvis xy distance from the command's start / 0.15 m)²) in every stance mode, w 1.5 (march drifted 0.43-0.90 m);
  `skill_flamingo_lift` lifted-foot height with or without contact, w 1.5. Locomotion erosion: cause still unknown,
  nothing changed for it.
- **G1 S flamingo drill changed:** extra settle U(0, 1) s + random foot order, so the 10 seeds are 10 different trials
  (they were identical). Older flamingo counts are one trial each.

## 2026-09-30 23:40 · Correction: rs_v6 it2550 flamingo is a real pass
- Re-gated with the randomised flamingo drill (10 different trials): **rs_v6 it2550 flamingo 10/10** (0 touch ticks,
  stance slip 0.5-2.2 cm; squat 10, torso 5, march 0) · **it3299 0/10** (65-85 touch ticks, 14-18 cm slip; squat 10,
  march 4, torso 5). The "one lucky trial" reading above (and in the first version of the review page) was wrong: the
  skill was learned by 2550 and lost between 2550 and 2800 while march was learned. Review page corrected.
- Consequence: no single rs_v6 checkpoint has both. Options: per-event brains (Event 6 = it2550, Events 3 / 7 = it3299
  line; events already use different brains) or a run that keeps both (rs_v7 starts from 3299 with the new terms; if
  flamingo does not come back, try from 2550 with march's swing-leg term). User decision pending on per-event brains.

## 2026-10-01 01:40 · Queue killed by a Ctrl-C at 23:43 — 2 h lost; PowerShell queue
- **r2bio_v1** (mattbio Rung 2 fine-tune, 400 its, 29 min): **G1 rung 2 10/10 at it 100 / 200 / 300 / 399**, 0 falls
  (bio probe it300: walk 1.20, run 3.00, sprint 3.62 m/s at a 3.8 command, no torque clipping).
- **g1_v1 died at it 71 (23:43:28)** together with the bash queue: `runs/queue_launcher.log` ends in `^C`. The queue was
  started detached through WMI, but as Git bash; it got an interrupt when a Claude turn ended. Found at 01:39 (my
  watcher only looked for END lines). Lost: 23:43-01:41 of GPU time. During that window the GPU was used only by the
  other project's run (PoDecath `r1_speed`, 8192 envs, 23:12 → ~02:55), which also slows this queue ~35 % while it runs.
- **Fix:** `tools/train_queue.ps1` (same queue.txt format; prep = uv arguments with `{run:<exp>/<name>}` tokens), started
  like detached_run.ps1 (WMI → Windows PowerShell), which survived every turn end on 2026-09-30. Watcher now reports
  a training log that has not been written for 10 minutes. g1_v1 restarted from scratch 01:41 (2.7 s/it).

## 2026-10-01 03:15 · GRANDMA Rung 1 (g1_v1) → Rung 2 base (g2_v1b)
- **g1_v1** (from scratch, 4096 envs, 2000 its, 92 min at 2.7 s/it with the GPU shared): episode length 990 by it 170;
  track_lin 1.67 at it 250, 1.58 at the end (top command 2.80 m/s; MATT r1_v2 1.60, zombie z1_v1 1.57 at it 2000);
  yaw error 0.57 (it 500) → 0.66 → 1.05 after the last speed stage (zombie's curve; MATT 0.75).
- **G1 rung 1 (grandma bars: vel RMS < 0.14 m/s, lateral < 0.435 m):** it 500 6/10 (2 falls at 1.7 / 1.9 m/s commands,
  not yet in the curriculum) · **it 1000 8/10** (0 falls; lateral <= 0.29 m; 2 misses on speed, worst 0.21) · it 1500
  6/10 (speed <= 0.144, lateral up to 2.02 m) · **it 1999 8/10** (speed <= 0.104, lateral 0.17-0.55 m, 2 over). Zombie's
  best Rung 1 was 3/10. No checkpoint passes 10/10; Rung 1 is a stepping stone (no event uses a Rung 1 grandma brain).
- **Warm start for Rung 2 changed to it 1000** (best lane keeping, as on the zombie): the queue had started g2_v1 from
  it 1500; stopped after 14 its, **g2_v1b** = same task from g1_v1 it 1000 (03:15, 2.1 s/it, GPU no longer shared);
  g2_v2's prep now points at g2_v1b it 1500 (to be re-checked against g2_v1b's gates).
- Review: `parity/tb/g1_v1/review.html` (vs r1_v2 and z1_v1; their event files were copied back from runs_archive for the
  screenshots and removed again).

## 2026-10-01 04:55 · GRANDMA Rung 2 base (g2_v1b): good at it 1000, speed collapse after → g2_v2b from it 1000
- **g2_v1b** (`-Rung2-Omni-Base`, warm start g1_v1 it 1000, 2000 its, 96 min). G1 rung 2 (grandma bars: lin < 0.187 m/s,
  yaw < 0.32 rad/s per segment, turntable < 2.8 s):

  | it | G1 | falls | turntable | lin seeds | yaw seeds | sprints > 2 m/s: median lin err | speed at 3.7 cmd |
  |---|---|---|---|---|---|---|---|
  | 500 | 0/10 | 1 | 0/10 (3.3-3.5 s) | 0 | 3 | 0.26 | 3.20 |
  | **1000** | **3/10** | 0 | 10/10 (2.64 s) | 3 | 4 | 0.25 | **3.33** |
  | 1500 | 2/10 | 0 | 10/10 | 2 | 6 | 1.13 | 0.24 |
  | 1999 | 2/10 | 0 | 10/10 (2.4-2.5 s) | 2 | 7 | 2.09 | 0.24 |

- **Speed collapse after the last curriculum stage** (top vx 3.27 → 3.73 m/s at it ~1040): training error_vel_xy 1.49
  (it 1250) → 2.09 → 2.61, track_lin_coarse 0.72 → 0.62, track_ang 0.44 → 0.64, pelvis_low terminations and torque effort
  down. CPU speed probe (8 s per command): it 1000 runs 2.36 / 2.92 / 3.33 m/s at 2.5 / 3.1 / 3.7; it 1500 and 1999
  do 0.24-0.46 m/s at every command >= 2.5 (2.0 → 1.76 still fine). A cliff in the policy, not a strength limit. Mirror
  image of z2_v2, which kept speed and lost yaw.
- The queue had started g2_v2 from it 1500 (fixed guess); stopped after 20 its. **g2_v2b** (04:53): `-Rung2-Omni` (zombie
  z2_v11 recipe: lin 3 + 1.5 with std 0.5, stride-filtered yaw, cap 3, wobble penalty, symmetric runner) from **g2_v1b
  it 1000**, 2000 its, 3.2 s/it, gates every 400. If the sprint collapses again under the full envelope: lower her vx top.
- Review: `parity/tb/g2_v1b/review.html`.

## 2026-10-01 05:20 · g2_v2b stopped at it 443 (entropy pulling the policy apart) → g2_v3 with entropy 0.001
- **g2_v2b it 400 gate:** 0/10, 0 falls · yaw tracking **10/10 seeds** (base it 1000: 4), turntable 10/10 (2.6-2.8 s),
  brake 10/10 · tracking_lin 1/10, backward 0/10 (19.2 m in the time allowed). Speed probe: 2.0 → 1.20, 2.5 → 1.25,
  3.7 → 1.80 m/s (start checkpoint: 1.87 / 2.36 / 3.33). A slide, not the base run's cliff.
- **Not a cheaper optimum:** mean reward 111.7 → 102.1 (it 40 → 410), action std 0.51 → 0.57, entropy loss 15.1 → 18.0,
  pelvis_low terminations up, track_lin −0.26, track_lin_coarse −0.15, track_yaw_filt +0.13, yaw_wobble unchanged
  (−0.32 → −0.34). The entropy bonus (0.005) raises the action noise; noisy fast running falls, the mean policy slows.
  Rung S hit the same thing at rs_v3 (0.005 → 0.0025).
- **g2_v3** (05:19): same task and warm start (g2_v1b it 1000), `entropy_coef` 0.001 (set in the task's rl cfg),
  `plain_init --max-std 0.3` (new option; start std 0.20-0.30 instead of ~0.5), 2000 its, gates 300 / 600 / 1000 / 1500 /
  1999. One variable changed vs g2_v2b apart from the std cap.

## 2026-10-01 06:00 · g2_v3 stopped at it 659 (speed plateau ~2 m/s) → g2_v4: base recipe + lateral-accel cap
- **g2_v3** (final recipe, entropy 0.001, start std 0.3): std held at 0.29, mean reward ~130 (g2_v2b: 112 → 102), so the
  entropy fix worked. But speed still below the start checkpoint and not recovering:

  | command (m/s) | start (g2_v1b it1000) | g2_v2b it400 | g2_v3 it300 | g2_v3 it600 |
  |---|---|---|---|---|
  | 1.0 | 1.00 | 0.98 | 0.61 | 0.91 |
  | 2.0 | 1.87 | 1.20 | 1.57 | 1.57 |
  | 2.5 | 2.36 | 1.25 | 1.84 | 1.90 |
  | 3.7 | 3.33 | 1.80 | 2.34 | 1.89 |

  Gates: it 300 1/10, it 600 0/10 (0 falls; yaw 8/10 seeds, turntable + brake 10/10, lin 1-2/10, backward 10 → 0/10 at
  18.6 m). The z2_v11 recipe's yaw terms (per-tick track_ang 2 + 1, stride-filtered 2, wobble −1) cost her speed.
- **Rule what-ifs on the existing reports** (no new sims): judging yaw per stride (the zombie ruling) and dropping
  sprint segments above 2.4-3.0 m/s lifts g2_v1b it 1000 only from 3/10 to 4/10 — her misses are linear-tracking
  errors on turns / crabs / sprints generally, so relaxed rules alone do not make a pass.
- **Hypothesis for the base run's cliff:** `-Rung2-Omni-Base` samples vx and wz independently (no lateral-accel cap). In
  its last stage most 3.3-3.7 m/s commands carry a 2-2.7 rad/s turn = 7-9 m/s² sideways, not executable → "fast
  command = do not run". MATT's Sym2+ recipes added the cap for this reason.
- **g2_v4** (05:58): `PoOlympic-Grandma-Rung2-Omni-Cap` = base recipe + `max_lateral_accel` 3.0, full stage-3 envelope
  from the start, entropy 0.001, start std <= 0.3, plain runner, from g2_v1b it 1000; 1200 its (2.7 s/it), gates 300 /
  600 / 900 / 1199. Not a test of the hypothesis alone (entropy and std changed too vs the base run).

## 2026-10-01 06:30 · g2_v4 stopped at it 623 (the cliff again, with the cap) → g2_v5: forward range cut to 2.8 m/s
- **g2_v4** (base recipe + lateral-accel cap 3, full envelope, entropy 0.001): it 300 **3/10** (0 falls; turntable 2.46 s,
  brake, backward 10/10; yaw 6, lin 3) · it 600 2/10 (yaw 8, lin 2). Speed probe:

  | command (m/s) | 1.0 | 1.5 | 2.0 | 2.5 | 2.8 | 3.1 | 3.4 | 3.7 |
  |---|---|---|---|---|---|---|---|---|
  | start (g2_v1b it1000) | 1.00 | 1.34 | 1.87 | 2.36 | 2.66 | 2.92 | 3.14 | 3.33 |
  | g2_v4 it300 | 0.95 | 1.33 | 1.73 | 2.24 | 2.53 | 2.67 | 2.51 | 1.83 |
  | g2_v4 it600 | 0.98 | 1.39 | 1.86 | 2.18 | 1.61 | 0.66 | 0.35 | 0.12 |

  The refusal starts at the top command and moves down (non-monotonic at it 300, a cliff at ~2.6 by it 600) while the
  mean reward RISES (100 → 114): the cap alone does not prevent it, so the infeasible-turn hypothesis is at best part of
  the cause. At 60 % strength the 3.3-3.7 m/s band is not worth attempting under this reward (falls cost −200 × …,
  the kernels pay little for a 0.4 m/s shortfall).
- **g2_v5** (06:29): `PoOlympic-Grandma-Rung2-Omni-Cap28` = g2_v4's recipe with vx ∈ [−1.40, **2.80**] m/s (3.0 × SS, her
  Rung 1 top), same warm start, 1200 its, gates 300 / 600 / 900 / 1199. EXPERIMENT ONLY: G1 still samples sprints to
  3.55 m/s and events still scale commands by √λ, so official G1 cannot reach 10/10 with this brain; the reading
  "segments with |vx| <= 2.8" is reported next to it. Lowering her envelope (and the bars / event commands with it)
  is a user decision.

## 2026-10-01 07:30 · g2_v5 finished: stable, best 4/10 → provisional `grandma_rung2.onnx`; rs_v7 started
- **g2_v5** (vx <= 2.80 m/s, 1200 its, 59 min at 2.9 s/it): error_vel_xy flat at 1.00-1.08, action std 0.29 → 0.20, mean
  reward 113 → 124, yaw error 1.1 → 0.71. G1 rung 2 (official bars, sprints sampled to 3.55 m/s): it 300 2/10 ·
  **it 600 4/10** · it 900 3/10 · it 1199 3/10; 0 falls, turntable 2.36-2.64 s, brake + backward 10/10 at every gate; lin
  2-4 seeds, per-tick yaw 5 seeds. Reading "stride yaw + segments <= 2.8 m/s": 3 / 5 / 4 / 4.
  Speed probe: 2.5 / 2.8 / 3.7 → 2.37 / 2.66 / 3.30 (it 600), 2.10 / 2.38 / 3.18 (it 1199): no cliff, monotonic.
- Remaining misses inside her range: 6-8 % slow above 2.3 m/s (sprint lin median 0.26-0.32 vs 0.187), crab segments up
  to 0.33, per-tick yaw 0.37-0.41 on sprints (0.03-0.07 per stride).
- **`parity/brains/grandma_rung2.onnx` = g2_v5 it 600 — provisional, NOT a G1 pass.** Rung 2 for GRANDMA stays open:
  (1) user decision on her command envelope (≈ 2.8 m/s top, bars + event scaling with it) and on judging yaw per
  stride like the zombie; (2) precision work on the recipe either way.
- Review of the four attempts: `parity/tb/g2_v5/review.html`.
- Queue: **rs_v7** started 07:28 (4.5 s/it at the start), then r0bio_v1, r2fbio_v1, crawl_bio_v1, getup_rev_v1.
