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
