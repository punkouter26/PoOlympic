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
