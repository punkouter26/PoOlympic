# PoOlympics — Approved Design (v0)

Approved 2026-09-27. Changes to any value here must be logged in `rl_optimization_log.md` and re-propagated to the model fingerprint + ONNX metadata.

## 1. Assumptions & Architecture Contract

### Engine & runtime
- Unity 6000.6.0f1 URP (this project). Physics = official MuJoCo Unity plug-in (`org.mujoco`), native MuJoCo version pinned identical to training.
- **PhysX forbidden**: no `Rigidbody`/`Collider`/`CharacterController`/PhysX joints in any athlete scene; an editor validator fails the scene. Ground, props, obstacles, shove/drop cubes are MuJoCo entities (`MjBody`/`MjGeom`/`MjFreeJoint`). Dynamic cubes come from a pre-allocated pool activated by writing `qpos`/`qvel` — no runtime `Instantiate`/`Destroy`.
- Inference: ONNX (opset 17, batch=1) via Unity Inference Engine (`com.unity.ai.inference`, ex-Sentis), output written straight into `mjData.ctrl`.
- Platform: Windows first (Editor + standalone) with 9:16 portrait framing/HUD. Android (arm64 MuJoCo build + ARM↔desktop trajectory check) is a separate phase after Rung 1 parity. iOS deferred.

### Training
- MuJoCo Warp, native Windows, RTX 5070 Ti Laptop 12 GB (Blackwell sm_120 → CUDA ≥ 12.8 builds). PPO in an mjlab-style framework; WSL2 Ubuntu fallback.
- One MJCF = single source of truth, loaded by training and imported by Unity. No USD/URDF hop.
- TensorBoard + MuJoCo viewer; every run/decision logged in `rl_optimization_log.md`.

### Athletes
- Canonical body: MATT (`SourceArt/test_MATT_Avaturn.glb`, 1.85 m, T-pose, symmetric, Mixamo-style names).
- Up to 8 lanes share MATT's policy; per-lane *athlete traits* (strength scale, latency, obs noise) produce real stat differences → betting odds.
- ZOMBIE (AccuRig FBX, needs rescale + CC twist-bone cleanup) and GRANDMA (GLB, **unrigged — must be rigged first**) become separate physics bodies at a later rung.
- Pure RL rewards (no mocap/AMP) for Rungs 0–2.

### Robustness curriculum (all rungs)
Random directional shoves (pelvis/torso) + MuJoCo cubes dropped on the athlete + DR ±10–20 % mass, friction, restitution, PD kp/kv.

### Behaviour ladder — first slice "Sprint Series"
Fall = pelvis < 0.55 m OR torso tilt > 60° OR any non-foot body touching ground. All bars: **10/10 consecutive seeds**, evaluated in CPU MuJoCo.

| Rung | Behaviour | Events | Pass bar |
|---|---|---|---|
| 0 | Stand + shove recovery | 1 Iron Pedestal, 5 Gust Gauntlet | 0 falls in 20 s with 0.5 m/s random-direction pelvis Δv every 3–5 s + 2 kg cube from 1.5 m; torso tilt < 10° within 1.5 s after each hit; feet stay inside 1 m × 1 m |
| 1 | Forward velocity walk/run | 8 30m Dash, 19 Terminal Velocity | RMS vel error < 0.15 m/s for cmd 0.5–3.0 m/s; 30 m, no fall, lateral drift < 0.5 m with 0.3 m/s shoves; stretch top speed ≥ 4 m/s |
| 2 | Omnidirectional + yaw | 9 Inverted Sprint, 10 Crab Shuffle, 11 Slalom, 12 360 Turntable, 22 Emergency Brake | cmd vx ∈ [−1.5, **3.8**], vy ∈ [−1, 1], ωz ∈ [−2, 2]; vel err < 0.2 m/s, yaw err < 0.3 rad/s (**steady-state**: measured from max(1.5, 0.5 + \|Δv\|/1.5) s after a command change); 360° < 3 s with < 0.3 m drift (**commanded at the max trained rate, 2.5 rad/s**); stop from 3 m/s within 2 m; 20 m backward, no fall |

*Rung 2 rulings (user, 2026-09-28): tracking is judged at steady state (acceleration is scored by the dash events); MATT's top speed with the elite torque caps is ≈ 3.8 m/s, so the envelope tops out there; the 360 Turntable drill commands 2.5 rad/s because ωz ≤ 2 cannot make 360° in < 3 s. Contract v3 (2026-09-28): gait clock speed = \|v_xy\| + 1.2·\|ωz\|. Zombie ruling (user, 2026-09-29): its yaw-rate tracking is judged averaged over one stride of its gait clock (the hunched shuffle sways the pelvis ±5° per step). Contract v4 stance skills approved 2026-09-29 (docs/CONTRACT_V4_STANCE_PROPOSAL.md).*

Later: R3 get-up (27) · R4 ramp/rubble/stairs (14–17) · R5 jumps/hurdles (18, 20, 21) · R6 Grandma/Zombie bodies · R7 optional motion-prior polish · R8+ crawl/carry/bench/kick/flip/parkour · Android port after R1 parity · game layer (betting, broadcast, ticker, records).

### Open risks
UnityMCP must be running for in-editor authoring. Licensing (Avaturn, Hunyuan3D) unresolved before commercial release. mjlab on Windows/Blackwell and `org.mujoco` on Unity 6000.6 to be verified in Phase A/B.

## 2. Body Specification (`matt.xml`)

**Frames**: MuJoCo x-forward, y-left, z-up. glTF (x-left, y-up, z-fwd) → MuJoCo (X,Y,Z) = glTF (z, x, y) (proper rotation). **qpos = 0 ≡ MATT T-pose bind pose.** All MjBody frames are world-aligned at qpos = 0; MATT's bones have non-identity bind rotations (Mixamo convention), so Unity bone offsets are constant per bone and computed once at qpos = 0 (`offset = inverse(body_bind) · bone_bind`).

| Body (bone) | kg | Geom | Joints (anatomical range, deg) | Act |
|---|---|---|---|---|
| pelvis (`Hips`) | 8.94 | capsule | free | — |
| torso (`Spine`+`Spine1`) | 13.06 | capsule | abdomen flex +60/−30, lat ±35, twist ±45 | 3 |
| chest (`Spine2`) + 2 clavicles (fixed) | 11.77 + 2×0.5 | capsule | welded | 0 |
| neck + head | 1.0 + 4.55 | capsule + sphere | welded (R0–2) | 0 |
| upper arm ×2 | 2.17 | capsule | shoulder flex 180/ext 60, abd 170/add 30, rot ±80 | 3 ea |
| forearm+hand ×2 | 1.30 + 0.49 | capsule | elbow 0–145 (wrist welded) | 1 ea |
| thigh ×2 | 11.33 | capsule | hip flex 120/ext 30, abd 45/add 30, rot ±40 | 3 ea |
| shin ×2 | 3.46 | capsule | knee 0–150 | 1 ea |
| foot ×2 | 0.85 | box | ankle dorsi 25/plantar 50, inv 30/ev 20 | 2 ea |
| toe ×2 | 0.25 | box | dorsi 60/plantar 30, **passive** spring-damper | 0 |

Total 80.0 kg (de Leva 1996 male). 23 actuators, 2 passive joints; nq = 32, nv = 31. Shoulder ranges are offset for the 90° T-pose abduction. Contacts: foot/toe↔ground expected; leg↔leg on; other self-contacts excluded.

**Actuators** — `<position kp kv forcerange>`:

| Group | kp | kv | force cap (Nm) | armature |
|---|---|---|---|---|
| hip | 300 | 30 | 280 | 0.02 |
| knee | 300 | 30 | 280 | 0.02 |
| ankle | 200 | 20 | 220 | 0.01 |
| abdomen | 300 | 30 | 200 | 0.02 |
| shoulder | 60 | 6 | 80 | 0.01 |
| elbow | 50 | 5 | 70 | 0.01 |

18 rad/s velocity limit = training penalty + pass criterion (≤ 5 % of frames), not enforced by physics.
Default pose (action offset): shoulders −80° from T (arms down), elbows 15°, hip flex 10°, knee 20°, ankle dorsi 10°. `target = default + 0.25 rad · action`, clipped to joint range.

**Solver** (identical both sides): timestep 0.005 s, `implicitfast`, Newton, iterations 20, ls_iterations = MuJoCo default (Unity plug-in cannot set it), pyramidal cone, condim 3. Ground plane friction (1.0, 0.005, 0.0001). Decimation 4 → policy 50 Hz.

## 3. Policy Interface

Observation (84, 50 Hz, built only from `mjData` qpos/qvel — never Unity Transforms):

| # | Term | Dim | Frame |
|---|---|---|---|
| 1 | pelvis lin vel | 3 | heading (yaw-only) |
| 2 | pelvis ang vel | 3 | pelvis local |
| 3 | projected gravity | 3 | pelvis local |
| 4 | pelvis height | 1 | world |
| 5 | command (vx, vy, ωz) | 3 | heading (zeros R0) |
| 6 | gait phase sin/cos | 2 | clock (frozen when cmd = 0) |
| 7 | joint pos − default | 23 | joint |
| 8 | joint vel × 0.05 | 23 | joint |
| 9 | last raw action | 23 | — |

- MLP 512-256-128 ELU. Obs normalization baked into the ONNX graph.
- ONNX outputs `ctrl[23]` (final clipped targets) and `action_raw[23]`. C# copies `ctrl[i] → mjData.ctrl[i]` with no math.
- ONNX metadata: actuator name order, default pose, gains, dt, decimation, MuJoCo version, SHA-256 of model fingerprint. Unity refuses to run on mismatch.
- Obs noise: training only. Unity: per-lane seeded trait noise only.

Tick order (Python and C#): read state → build obs → infer → write ctrl → apply scheduled disturbance → 4 × `mj_step`. Unity: `Time.fixedDeltaTime = 0.005`, `Physics.gravity = (0, −9.81, 0)`; PolicyRunner subscribes to `MjScene.preUpdateEvent` (fires right before each `mj_step`) and runs inference every 4th event. `ctrlCallback` is never used. Pinned MuJoCo = 3.11.0 everywhere.

## 4. Parity Contract

- **Assets**: Unity imports the same `matt.xml` via the plug-in importer; no hand axis conversion. Skinned bones copy MjBody poses each frame; merged bones (Spine1/2, fingers, hair) follow parents rigidly.
- **Model fingerprint**: Python and Unity each dump compiled `mjModel` (counts, body mass/inertia, jnt range/axis, actuator gain/bias/forcerange, geom size/friction, opt) to JSON. Ints exact, floats ≤ 1e-6 rel.
- **Shove**: instantaneous Δqvel on pelvis free joint at a control-tick boundary.
- **Cube**: 0.2 m box, 2 kg, friction 0.8, free joint. Pool: 4 per env (training), 16 per scene (Unity), parked resting at `x = 50 + 2i`; fire = write qpos(7) + qvel(6).
- **Lane isolation**: lane k geoms `contype = conaffinity = 1<<k`; ground + cubes carry all lane bits → athletes never collide with each other (matches 1-athlete training envs).
- **Athlete traits**: strength 0.85–1.15 (gainprm/biasprm/forcerange), latency 0–4 substeps, obs-noise σ. Same ranges in training DR.
- **DR** (training only): mass ±15 %, friction ±20 %, kp/kv ±15 %, restitution via solref. Unity uses nominal.
- **Precision**: Warp is float32, MuJoCo C is float64 → the golden reference is a **CPU MuJoCo** rollout (same C version as Unity).
- **`reference_trajectory.json`**: meta (fingerprint hash, mujoco version, dt, decimation, seed), disturbance script (tick, type, target, qvel/qpos), 250 frames @ 50 Hz with t, qpos[32], qvel[31], obs[84], action_raw[23], ctrl[23], actuator_force[23].

### Gates

| Gate | What | Tolerance |
|---|---|---|
| G0 Fingerprint | Python vs Unity `mjModel` | exact / 1e-6 rel |
| G1 Sim-to-sim | Warp-trained policy in CPU MuJoCo | meets rung bar |
| G2 Obs builder | C# obs from recorded qpos/qvel | max abs < 1e-5 |
| G3 Policy replay | recorded obs → Unity inference | max abs < 1e-4 |
| G4 Physics replay | recorded ctrl open-loop into Unity MuJoCo | qpos drift < 1e-3 over first 1 s |
| G5 Closed loop | policy + same disturbance script, 5 s | same fall outcome; cadence ±5 %; mean \|τ\| ±10 %; pelvis-height RMS < 3 cm |
| G6 8-lane | 8 athletes + cube pool in one scene | each lane passes G5 vs its solo run |

Zero-Brain (Phase B) = G0, G2, G3, G4. Rung 1 must pass G0–G6 before any Rung 2 training.

## 5. Training specifics per rung

| Rung | Command sampling | Episode / termination | Key reward terms |
|---|---|---|---|
| 0 | zero | 20 s, fall | upright, pelvis height, CoM over support, no foot slip; penalties torque, action rate, joint-vel |
| 1 | vx 0.5–3.0 (curriculum → 4) | 20 s | lin-vel tracking, phase-consistent contacts, air-time; + R0 terms |
| 2 | vx −1.5–4, vy ±1, ωz ±2, plus zero-cmd stops | 20 s | vel/yaw tracking, drift; + R1 terms |

Shove/cube curriculum from R0; intensity ramps when success > 80 %.
