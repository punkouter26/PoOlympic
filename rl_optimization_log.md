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
