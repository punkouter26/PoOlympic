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
- [x] **C7 Rung 2 — train** omnidirectional + yaw to bar. **GATE G1.** — `rung2.onnx` = **r2_v8 it 600**: G1 **10/10** (0 falls, turntable 2.64/2.58 s, brake 1.16 m, 20 m backward); 20-seed gate 18/20; margin 23/30 official protocol / 28/30 steady-state variant — the remaining misses are sprints from standstill (acceleration ~1.5 m/s², top speed ≈ 3.8 m/s < the 4.0 envelope top; see log, open questions for the user). Path: r2_v1 (yaw decay) → r2_v2 (yaw rewards) → symmetry aug + mirror loss → **contract v3** clock (yaw weight 1.2) → widened envelope → lateral-accel command cap (r2_v6 10/10) → sharper speed kernel (r2_v7) → sprint-focus sampling (r2_v8).
- [x] **C8 Rung 2 — Unity spot-check.** G3, G5, G6 with rung2 brain; commands switchable from HUD. — G2 4.4e-16 · G3 ≤ 4.3e-6 · G4 ≤ 9.5e-7 on 8 lane refs (stand / back / crab ± / spin ± / sprint 3.5 / walk-turn); **G6 PASS 8/8** in Testbed_Rung1 (1 s drift ≤ 4.6e-6); Testbed_Rung1 HUD presets Stop / Walk / Sprint / Back / Crab ◀▶ / Spin. **Phase C complete.**

## Phase D — Engine Polish & Game Loop (detailed plan, written 2026-09-27 after C6 passed)

Rules: every surface an athlete can touch is a MuJoCo geom generated into the MJCF (training, CPU gates and Unity share it); stadium / dressing art from Blender is render-only (no colliders — PhysXGuard). Events run on control ticks with seeded RNG (reproducible attempts). Runs in parallel with C7/C8 (GPU trains, editor builds).

- [x] **D1 Event 1 — Iron Pedestal (vertical slice, 1 biped)**
  - [x] physics: `scene_pedestal.xml` — 1 m × 1 m × 0.5 m block, top at z = 0, ground at −0.5 (athlete pose / obs / fall rule unchanged)
  - [x] CPU check: rung-0 brain survives 40/40 seeds on the pedestal (20 s, 0.5 m/s gusts + cubes); foot overhang ≤ 0.10 m
  - [x] Unity: `IronPedestalEvent` (countdown → 20 s live → result → auto reset; gusts + cube drops on seeded tick schedule; out = fall rule / foot below pedestal top / non-foot contact), `EventHud` (D2 anchors), `Event_IronPedestal.unity` via *PoOlympic › Events › Build Event 1*; play-through SURVIVED 20.00 s · EditMode 18/18
  - [x] G0 for the pedestal scene (Python vs Unity fingerprint) + a G5-style closed-loop parity run of one attempt (2026-09-29):
    **G0 pedestal 0 mismatches**; G5 `pedestal_rung0` (Rung 0 brain, standard shove + cube, 5 s) vs Unity
    `pedestal_rung0_play`: drift 1.1e-6 @ 5 s, torque ratio 1.0000001, same outcome — **PASS**
  - [x] Iron Pedestal fine-tune (foot-on-pedestal term, trained on the pedestal geometry) for edge margin — tried twice
    (ped_v1, ped_v2; see D4), neither beat the Rung 0 brain → kept r0_v2_it1000
- [x] **D2 Stadium (Blender MCP, Olympic realistic)** — `SourceArt/Stadium/stadium.blend` → `Assets/PoOlympic/Art/Stadium/Stadium.glb` (1.5 MB, ~28k tris, render-only): World-Athletics 400 m oval (84.39 m straights, R 36.5 m, 8 × 1.22 m lanes = meet layout), 20 m start extension, finish / 100 m lines; two-tier bowl with front-view crowd textures; roof canopy + floodlight ring; LED boards; 2 scoreboards. **Venues for every event category** with `VENUE_*` anchors (glTF nodes + `venues.json`): HomeStraight (8, 19, 9, 10, 22, hurdles) · CentreStage (1, 5) · Agility (11 slalom, 12 turntable) · Terrain (14–17, parkour) · Jumps (runways + pit) · Mats (27 get-up, crawl, flip) · Skills (carry, bench, kick + goal) · BackStraight. `EventScenes.PlaceStadium(venue, groundY)` snaps a venue onto the MuJoCo origin (verified: home anchor at Unity (−57.81, −0.50, −41.38)). Iron Pedestal scene now plays inside the stadium (`parity/d1/stadium_live.png`).
  - [x] full 30-event list received (Phase E catalogue): every event maps onto an existing venue — no new venue type needed
  - [x] **one venue per event (30), each for 8 competitors** — reproducible builder `SourceArt/Stadium/build_venues.py` (+ `stadium_helpers.py`), run inside Blender. Five infield rows + track: centre row = stationary events (1 Iron Pedestal: 8 × 1 m × 1 m × 0.5 m iron pedestals in a row, 3 m pitch; 2, 3, 4, 5, 6, 7, 12, 18 as 2 × 4 station grids); rows ±13 / −26 / +26 = 8-lane blocks with props (rails, gates, 15° ramp, stairs, rubble, stepping stones, parkour wall/drop/hurdles, crawl ceiling, crates, benches, mats, goals, long-jump runways + pit); track = 8, 20, 22, 27 (home straight) + 19 (back straight). Phase-coloured pads, painted event names + lane numbers; no pad overlaps, all inside the kerb (checked). 240 competitor anchors `E##_L#` + `venues.json` (lane pos + yaw, MuJoCo axes) — the future source for the physical props in each event MJCF. GLB 3.9 MB, ~77k tris, 1555 objects (merge per event in D6).
  - [x] **Stadium Hub** `Stadium_Hub.unity` (*PoOlympic › Events › Build Stadium Hub*): all 30 events as `EventVenue` objects (catalogue `events_catalog.json` = user rules, phase, skill, brain + the 8 competitor anchors), `StadiumDirector` event picker (flies the camera to a venue, rules card, Play ▸ when the event scene exists); Build Settings = hub + built event scenes. Captures `parity/d2/hub_E*.png`.
  - [x] **z-fighting fixed**: decal layers ≥ 1 cm apart (`Z_PAD … Z_LABEL` in build_venues.py; shell lane / finish lines + start extension lifted), redundant pedestal cap removed, audit = 0 overlapping different-material decals < 8 mm; event camera near plane 0.05 → 0.2 m (hub 0.5 m).
  - [x] `EventScenes.PlaceStadium(event, lane)`: stadium turned + shifted so the competitor spot lands on the MuJoCo origin facing +x; Event 1 athlete = lane 4 (`E01_L3`), stadium pedestal ≡ MuJoCo pedestal (bounds verified), 7 pedestals ready for D4 · `parity/d1/event1_venue.png` · EditMode 18/18
  - [x] **dressing pass** `build_dressing.py` (render-only): calm crowd in home colours, scoreboard / LED branding, Olympic rings + gates + roof flags + cauldron, venue materials, plaza / park / skyline. GLB 6.5 MB, 415 anchors unchanged.
  - [x] **realism pass, Blender half** `build_realism.py` (textures: `fetch_textures.py`, Poly Haven CC0 1K): PBR (albedo + OpenGL normal + ARM → glTF ORM) on 18 materials + generated striped turf; world-scale box UVs on textured faces (crowd UVs kept) + `UVLightmap`; bevel + weighted normals on 233 props; albedo clamp 0.04–0.9; alpha glass ribbon; kerb, plinth, tree soil, start-line wear; 13 flags + cauldron flame split out with pivots. Export with tangents, 26.5 MB (8.8 MB textures), 415 anchors identical.
  - [x] **realism pass, Unity half** `StadiumLook.cs` (hooked into `PlaceStadium`): `StadiumAtmosphere.prefab` (procedural sky + linear fog + ambient, global Volume = ACES / bloom / colour adjustments / vignette, realtime-on-awake reflection probe, 4 floodlight banks), `FlagWave` / `FlameFlicker`, camera post + FXAA. SSAO already on the PC renderer (not added to Mobile: cost). Verified in the rebuilt event scenes; floodlights moved 3 m in / 2 m down from the roof ring at 150 (on the ring they blew the roof out and bloom washed the frame), Neutral tonemapping, bloom clamp 8 (`StadiumLook.Retune` edits the assets in place).
  - [x] **indoor arena + polish pass** (2026-09-28, user: the canopy shadow made the home straight too dark → closed
    roof + spotlights): `build_indoor.py` (domed ceiling, facade seal, truss grid, 24 field + 12 crowd-wash spot fixtures
    with `Spot_##`/`SpotAim_##` anchors) + `build_polish.py` (rest of the 20-point list: n-gons triangulated, prop pivots,
    drain + pad edging, tint/grime/baked AO in COLOR_0, grunge detail maps, binary metal, more bevels; `export_stadium()`
    keeps the glTF scene name "Scene"). Unity `StadiumLook.ArenaLighting`: roof casts no shadows, near-vertical key
    light, 36 spot lights under Stadium/ArenaLights, trilight ambient, static batching; all 11 stadium scenes relit;
    Mobile renderer → Forward+. APK 207 MB re-deployed to the Pixel 9 Pro (`parity/lighting/android_menu.png`).
- [x] **D3 Broadcast** — 9:16 tracking cameras per event, HUD polish (UI Toolkit), result cards (2026-09-29)
  - `BroadcastHud` (UI Toolkit, `UI/BroadcastHud.uss`, authored 1080 wide, scaled onto the letterboxed camera rect) replaces
    the IMGUI RaceHud / StandingsHud / HeatHud in all 10 event scenes: top bar, standings with body chip + win odds,
    countdown banner, play-by-play ticker (`Commentary`: start, lead changes with 4 s cooldown, athletes out, winner,
    records, bets), result card (podium, bet payout, record, gauntlet points / Next event)
  - `IBroadcastBoard` (Broadcast.cs) on every 8-lane event: phase, heat, `HoldStart` (betting window), winner's mark
  - `BroadcastDirector` tracking cameras: races follow the leader along the race axis (trackside / head-on / high-wide
    cuts every 5 s), arenas hold the establishing shot then orbit, result = winner close-up
  - `CaptureTools` (Game view pinned to 1080 x 1920, screenshots outside Assets/); annotated before/after:
    `parity/ui/d3_hud_before_after.png`, `parity/ui/menu_gauntlet_before_after.png`; play-throughs `parity/d3/`
- [x] **D4 8-lane Iron Pedestal heat** — rules per catalogue: 8 runners, **last one standing wins**
  - physics `scene_pedestal8.xml` (+ `pedestal8_layout.json`): 8 lane-isolated athletes, each on its own 1 × 1 × 0.5 m pedestal; lane origins from `venues.json` (E01, relative to lane 4) → physics pedestals ≡ stadium pedestals · **G0 pedestal8 PASS 8/8**
  - athlete traits (contract `trait_ranges` / `obs_noise`): strength ×[0.85, 1.15] on force limits, latency 0–4 substeps (mjlab delay semantics), sensor noise ×[0, 1] of training noise — `PolicyRunner.SetTraits`, nominal = parity (EditMode 18/18)
  - rules (Python `poolympic/events/iron_pedestal.py` ≡ Unity `IronPedestalHeat`): 3 s rounds, gust 0.3 m/s + 0.05/round (own seeded direction per lane), cube every 3rd round, out = fall rule / stepped off (`AthleteJudge`), ranking by elimination time; tuned on CPU: heats 27–41 s (mean 35), first out ~17–26 s
  - Unity `Event_IronPedestal_Heat.unity` (*Build Event 1 — Iron Pedestal Heat*), `HeatHud` standings (status, out time, traits), winner banner; hub Event 1 ▸ opens the heat. Play-through: L1 wins after 31.6 s, 11 rounds (`parity/d4/heat_*.png`)
  - [x] odds from traits (betting layer): `training/tools/fit_odds.py` — 600 CPU heats (60 per playable event, mixed
    MATT/zombie lineups, random traits) → Plackett-Luce rating per event (zombie, strength, latency, noise) →
    `Models/odds_model.json`; Unity `Odds` (softmax, 10 % shrink to uniform, 10 % margin, odds 1.01-50). Favourite win
    rate: 23-83 % (uniform 12.5 %); Emergency Brake / Iron Pedestal ≈ uniform (traits do not predict the winner)
  - [x] Iron Pedestal fine-tune tried twice (ped_v1 fixed ±0.8 m/s gusts; ped_v2 adaptive gusts) — neither beat r0_v2_it1000 on identical heats (28.2 s mean survival); Event 1 keeps the Rung 0 brain (see log)
- [x] **D5 remaining Sprint Series events** — done: 5, 8, 9, 10, 11, 12, 13, 19, 22 (5 = `scene_shaker8.xml` spring platforms + `events/gauntlet.py` ≡ `GustGauntletEvent`; all 8-athlete, Rung 2 brain; pattern: venue layout → `compose_meet(origins, props)` MJCF (G0 8/8 per lane, props checked with `ParityTools.DumpProps`) → CPU rules in `poolympic/events/` ≡ Unity controller → `EventScenes.Build*` via the shared `BuildMeetScene`. 8/9/19/22 = straight-track framework (`scene_track8.xml`, `events/track.py` ≡ `TrackRaceEvent` + `RaceHud`); 10 = `scene_crab8.xml` + rails, `events/crab.py` ≡ `CrabShuffleEvent`; 11 = `scene_slalom8.xml` + poles, `events/slalom.py` ≡ `SlalomEvent`; 12 = `scene_turntable8.xml`, `events/turntable.py` ≡ `TurntableEvent`; per-tick `PolicyRunner.steer` hook; generic `StandingsHud`); 13 Steeplechase Jog (2026-09-29) = `track.py` steeple ≡ `TrackRaceEvent` Steeplechase: 50 m at 3.5 m/s, flight brain `r2f_v3_it100.onnx` (zombie: zombie_rung2), scored on **ground time** = finish − hang time (flights ≥ 20 ms counted per physics substep, `FootGait`), `Event_SteeplechaseJog.unity` on venue E13; CPU winners ≈ 8.9-9.1 s ground (air 6.7-6.9 s), Unity heats 9.0-9.1 s — the rest of the catalogue: **Phase E** below
- [x] **D6** full suite re-validation + perf pass (2026-09-29): pytest 32/32, EditMode 36/36, **G0 PASS on every event scene**
  (meet8, pedestal8, track8, turntable8, crab8, shaker8 — solo check now ignores the lane-owned shaker —, slalom8,
  meet8_mzmzmzmz, trench8 via compose --verify, solo pedestal). Perf (desktop editor, 8 athletes): mj_step 0.33-0.40 ms
  (7-8 % of real time), 8.9 ms CPU frame, 194 draw calls / 8 SRP batches → the planned per-event stadium merge is not
  needed. Mobile: see Android (57.6 fps on the Pixel 9 Pro).
- [x] **Crowd contact** (2026-09-30, user: "more interaction and collisions … do all 6"): every event scene lets every
  body part of every athlete collide with every part of the other 7 (`build_mjcf.crowd_bits`; G6 testbed stays isolated).
  Six events laid out for contact (venues: `build_venues.py` + Blender pass `build_crowd.py`, Stadium.glb re-exported;
  rules in `poolympic/events/*` ≡ Unity controllers): 1 one iron beam (0.7 m) · 5 one shaker floor (2 × 4, 0.8 m) ·
  8 crawl lanes 1.1 m (`scene_crawl8`, DNF by distance) · 11 mirror slalom (1.4 m lanes, 1.6 m/s) · 12 spin ring 0.75 m
  (DQ 0.75 m) · 19 lane break at 15 m (field squeezes to 0.61 m). Verified: `tools/check_crowd_contacts.py` (16,184/16,184
  part pairs per scene, all-MATT + mixed) + Unity `CrowdContactTests` 11/11; odds refitted; tuning in rl_optimization_log.md.
  - [ ] optional crowd training run (2-4 colliding athletes per env) if contact falls feel too frequent (19, 11)
  - [ ] mixed lineups: the zombie is narrower, so 11 and 19 see little MATT↔zombie contact

## Phase E — The 30 Olympic Events (catalogue, added 2026-09-27)

All 30 events are unlocked from the start (instant exhibition play or gauntlet construction). Each event = **venue** (render-only, `Stadium.glb` anchor) + **props** (MuJoCo geoms in the event MJCF, shared by training / CPU gates / Unity) + **brain** (the rung that owns the skill) + **event controller** (rules, scoring, elimination, seeded schedule; pattern: `IronPedestalEvent`). An event is done when: CPU scoring script passes with the brain, the Unity scene plays it end-to-end (1 biped, then 8 lanes), and a G5-style parity run of one attempt passes.

Skill gaps: events marked **S** need behaviours the current contract cannot command (upper-body / hand targets, pelvis height, single-leg, cadence, …) → one "stance skills" rung with an extended command block (**contract v4** — v3 is the gait-clock change). **Approved 2026-09-29** (`docs/CONTRACT_V4_STANCE_PROPOSAL.md`): one shared Rung S brain warm-started from rung2, MATT first, ranges as proposed, welded wrists / forearm-tip reach target. **All non-training work done 2026-09-29**: contract v4 in Python + Unity (G2-G5 PASS on a v4 test brain), task `PoOlympic-Matt-RungS-Stance` (C1 PASS), warm start `tools/expand_obs.py`, G1 drills `eval_cpu.py --rung S`, v4 export — **only the training run is left** (commands in the proposal doc). Events marked **M** likely need a motion prior (R7).

| # | Event | Rules (summary) | Venue | Props (MJCF) | Brain / rung | Status |
|---|---|---|---|---|---|---|
| **Phase 1 — Stability & Balance** |
| 1 | The Iron Pedestal | 8 runners on 1 m × 1 m pedestals; last to keep equilibrium without stepping off wins | CentreStage | pedestal (per lane in the meet) | R0 | **playable**: 8-runner last-standing heat (D4) + solo practice scene |
| 2 | Torso Archer | feet planted, track fast overhead flight targets with the upper body; angular accuracy, zero foot slip | CentreStage | flying target (mocap-free kinematic body) | **S** upper-body target cmd | Rung S ready to train (G1 drill `E2_*`) |
| 3 | Deep Squat Endurance | rhythmic squat reps; lowest torso drop + balance retention | CentreStage | — | **S** pelvis-height cmd | **CPU playable** (2026-09-30): `scene_squat8.xml` (venue 2 x 4 grid), `events/squat.py` — 12 metronome reps, 0.25 → 0.45 m deep, tempo 4.5 → 2.2 s; 10 pts/rep for depth accuracy; FELL / STEPPED (foot slid > 0.20 m) = out. Runs **all-mattbio** (`compose_mixed.py squat8 mattbio×8`: the brain's training physics; Unity refuses a brain on a foreign fingerprint). +10 balance bonus for finishing. Heats 42 s, 0-2 finishers, exits 24-40 s (test_deep_squat_heat). Unity `DeepSquatEvent` + `EventScenes.BuildDeepSquat` written, **scene build after training (Unity closed)** |
| 4 | Precision Javelin Reach | single-arm extension to dynamic targets at max reach, stance anchored | CentreStage | target marker | **S** hand-target cmd | Rung S ready to train (G1 drill `E4_*`) |
| 5 | The Gust Gauntlet | lateral wind bursts + floor shakers; scored on recovery time back to centre | CentreStage | shaker platform (spring-mounted, x/y slide joints) | R2 + homing steering | **playable** (10 rounds of escalating lateral bursts, floor jolt every 2nd round; rank by total recovery time, fall / stepped off = out, `Event_GustGauntlet.unity`) |
| **Phase 2 — Fundamental Track & Gait** |
| 6 | The Flamingo Classic | one foot raised; time until touchdown | CentreStage | — | **S** single-leg stance cmd | Rung S ready to train (G1 drill `E6_*`) |
| 7 | Cadence March | high-knee marching in place to a rising metronome | CentreStage / Agility | — | **S** cadence / march-in-place cmd (phase clock with zero velocity) | Rung S ready to train (G1 drill `E7_*`) |
| 8 | **30m All Fours** (replaced The 30m Dash, 2026-09-29) | race on hands and feet from a face-down start; falls never eliminate, standing up = DQ | HomeStraight | — | Crawl (`crawl_matt.onnx`, `crawl_zombie.onnx`) | **playable** (8 runners, any MATT/zombie lineup, `Event_30mAllFours.unity`; `events/all_fours.py` ≡ `TrackRaceEvent` AllFours) |
| 9 | The Inverted Sprint | 20 m backwards; DQ on lane drift or backward tumble | HomeStraight | — | R2 | **playable** (8 runners at cmd −1.5 m/s, lane-drift DQ, stadium turned 180°, `Event_InvertedSprint.unity`) |
| 10 | Crab Shuffle Relay | side-step between parallel rails without crossing legs | HomeStraight / Agility | boundary rails | R2 (+ leg-cross check) | **playable** (8 athletes side-step 20 m at 1.2 m/s between physical rails; +1 s per leg crossing / rail touch, `Event_CrabShuffle.unity`) |
| **Phase 3 — Omnidirectional Agility** |
| 11 | Slalom Sprint | weave through gates; penalties for missed gates / clipped flags | Agility | gate poles + flags | R2 + gate-following steering | **playable** (7 physical poles on each lane's centre line, weave at 2.2 m/s on a seeded racing line; +0.5 s per clip, +2 s per wrong side, `Event_SlalomSprint.unity`) |
| 12 | The 360 Turntable | rapid in-place turns on a marked spot; rotational speed, zero drift | Agility (turntable pads) | — | R2 | **playable** (8 spots, 3 turns at 3 rad/s, seeded direction, score = time + 2 s/m drift, `Event_360Turntable.unity`) |
| 13 | Steeplechase Jog | 50 m run with sustained aerial flight phases between strikes | HomeStraight | — | R2 flight (`r2f_v3_it100.onnx`) | **playable** (8 runners, 50 m at 3.5 m/s, ranked by ground time = finish − air, `Event_SteeplechaseJog.unity`) |
| 14 | The Alpine Ramp | ascend a 15° ramp into a finish sensor | Terrain | 15° ramp | R4 | todo |
| 15 | Cross-Country Rubble | traverse randomized mounds and ruts | Terrain | heightfield / box rubble (seeded) | R4 | todo |
| **Phase 4 — High Impact & Jumping** |
| 16 | The Platform Drop | rapid stair descent; descent speed + soft landing | Terrain | stairs (down) | R4 | todo |
| 17 | Stadium Stair Climb | 20-step climb without catching toes on step lips | Terrain | 20 stairs (up) | R4 | todo |
| 18 | The Olympic High Jump | static squat jump; highest pelvis clearance | Jumps / Mats | — | R5 | todo |
| 19 | Terminal Velocity Sprint | open sprint to top speed until saturation or collapse | BackStraight (84.39 m) | — | R2 | **playable** (8 runners, peak-speed ranking, `Event_TerminalVelocity.unity`) |
| 20 | Low Hurdle Dash | 30 m with 0.3 m hurdles; high-knee clearance | HomeStraight | 0.3 m hurdles | R5 | todo |
| **Phase 5 — Heavy Athletics & Transitional Motion** |
| 21 | The Sandpit Long Jump | run-up, forward launch into the sand pit | Jumps | take-off board, pit (soft contact) | R5 | todo |
| 22 | Emergency Brake | full sprint to a red stop line; full standstill without crossing it | HomeStraight | stop line (visual) | R2 | **playable** (per-runner "nerve" brake point, DQ on crossing, `Event_EmergencyBrake.unity`) |
| 23 | The Trench Crawl | low-ceiling tunnel forces all-fours crawling | Terrain | tunnel ceiling (0.72 m, `build_mjcf.trench_props`) | Crawl (no new training) | **playable** (16 m all-fours race under a 12 m see-through ceiling; MATT squeezes through 15-19 s, zombie untouched ~17 s → mixed heats contested, `Event_TrenchCrawl.unity`) |
| 24 | The Courier Carry | carry a weighted crate 15 m without dropping / pitching back | Skills | crate (free body) + hand contact | R8 carry (needs hand/wrist contact on the body) | todo |
| 25 | The Bench Relay | approach a bench, stable seated rest, explode back into a sprint | Skills | bench | R8 sit/stand | todo |
| **Phase 6 — The Extreme Decathlon** |
| 26 | Stepping Stones | narrow elevated pads, zero room for error; miss-step = drop out | Terrain | elevated stepping pads | R4 + foot-placement targets (**S**) | todo |
| 27 | The Resurrection Dash | start flat on the back; fastest to rise and sprint 5 m | Mats | — | R3 get-up | todo |
| 28 | Floor Acrobatic Sprint | flip / cartwheel across a gymnastics mat | Mats | mat (soft contact) | R8 acrobatics (**M**) | todo |
| 29 | Striker Shootout | intercept a rolling ball mid-stride and kick past a target | Skills (goal) | ball (free sphere, pooled) | R8 kick (+ ball obs) | todo |
| 30 | The Grand Parkour Vault | approach, wall vault, drop landing, hurdle sprint | Terrain | wall, drop, hurdles | R8 parkour (**M**) | todo |

Order of work (follows brain availability): 1 (8-lane) → 8, 19, 13 (rung1) → 9, 10, 11, 12, 22 (after C7/C8) → 5 → contract-v3 stance skills (2, 3, 4, 6, 7) → R3 (27) → R4 (14, 15, 16, 17, 26) → R5 (18, 20, 21) → R8 (23, 24, 25, 28, 29, 30). Per-event: MJCF props via `build_mjcf.py`, CPU scoring in `poolympic/events/`, Unity controller + `EventScenes.Build*`, 1-biped then 8-lane, parity spot-check.

## Phase Z — ZOMBIE: second athlete body, trained from scratch (added 2026-09-28, **Approved** 2026-09-28)

User decisions (2026-09-28): own body at **zombie size (1.1 m)**; movement personality from **style rewards** (no mocap);
**"weaker but relentless"** (≈70 % of MATT's size-scaled torque, slower, very stable); brains **Rung 0 + Rung 2** (all 9
playable events); **full self-collision** (arms ↔ torso/legs too — MATT keeps legs-only until retrained). AGENTS.md rules
apply: TensorBoard + MuJoCo viewer for every run, close Unity during 30 min+ runs, realistic mass/torque for the size.

- [x] **Z1 Asset** — `SourceArt/Zombie/convert_zombie.py` (Blender) → `SourceArt/Zombie/zombie.glb`: FBX mesh/skeleton
  misalignment fixed, bundled T-pose baked as rest with legs straightened (the file's T-pose kept a 36° knee crouch),
  twist/share/face bones merged into their limbs, 22 bones renamed to Mixamo names, metres, glTF axes. 25k verts, all
  weighted, 1.135 m, L/R skeleton symmetric to 5 mm. *Accept: renders as a clean straight-legged T-pose (done).*
- [x] **Z2 Body pipeline per athlete** — parametrise extract_skeleton / build_mjcf / contract / tasks / exporters / Unity
  by body id (`matt` | `zombie`) with MATT's outputs byte-identical (G0/G2–G6 unchanged). Zombie specifics: no finger
  bones (forearm capsule ends at the hand), de Leva masses scaled to the size (~22 kg), torques scaled for dynamic
  similarity × 0.7, joint speed limits human-like, full self-collision (only parent/child + touching thigh pair excluded).
  *Accept: zombie.xml / scene_zombie.xml compile, mass/inertia report, default standing pose settles (zero brain falls
  naturally, 0 penetrations), Python fingerprint written; MATT artefacts unchanged.*
- [x] **Z3 Size-scaled rules** — Froude scaling (L ≈ 0.62 of MATT's leg length): fall height, command envelope (top speed
  ≈ 0.8 × MATT's), gait clock, push/gust and cube magnitudes, G1 drill distances/times. One table in DESIGN.md.
- [x] **Z4 Style rewards** (reward terms in; tuning after the first runs) — hunched trunk lean, arms held forward, wide stance, low foot clearance (shuffle), lateral
  lurch; tuned so the gait is visibly different from MATT's at the same command without failing G1.
- [x] **Z5 Rung 0 zombie** — train (TensorBoard + viewer), G1 rung0 (scaled drills). *Accept: G1 10/10.*
- [ ] **Z6 Rung 2 zombie** — curriculum (walk → run → omni + yaw) with the Rung 2 fixes (symmetry + mirror loss,
  lateral-acceleration cap, sprint focus). *Accept: G1 rung2 10/10 on the scaled drills.*
  - ~~Kept: z2_v2 it 1500 (0/10)~~ → **2026-09-29 night: `zombie_rung2.onnx` = z2_v11 it400 — G1 8/10, 30-seed margin
    21/30 (23/30 with stride-averaged yaw), 0 falls**, turntable 2.21 s, runs 2.0 → 2.06 m/s. Fixes (z2_v5-v14, see log):
    yaw terms on the stride-filtered yaw rate, lateral-accel cap 3 m/s², speed weights 3 + 1.5, yaw-wobble penalty.
    Remaining misses: 2.4-2.85 m/s sprints 0.003-0.04 m/s over the speed bar. Unity re-validated with the new brain:
    EditMode 36/36 (G2-G4), G6 mixed 8/8. **Ruling (user, 2026-09-29): zombie yaw tracking is judged averaged over one
    stride** (`yaw_rms_stride`) → margin 23/30; still open for 10/10: sprint precision (2.4-2.85 m/s runs 0.003-0.04 m/s
    over the speed bar) — needs training.
- [x] **Z7 Unity** — contract per body in PolicyRunner, zombie MATT-style visual binding, G0/G2–G5 on the zombie, mixed
  MATT + zombie meets (lane body chosen per lane → `compose_meet` with per-lane bodies), menu roster card, events read
  MeetLineup. *Accept: G6 with a mixed 8-lane meet; every playable event runs with any lineup.* — PASSED 2026-09-28:
  `Testbed_Zombie.unity` (`ZombieTestbed`): **G0** 0 mismatches · **G2** ≤ 4.4e-16 · **G3** ≤ 3.8e-6 · **G4** (1 s) ≤ 1.3e-6
  on zombie_rung0 / zombie_rung2 + the 8 mixed-plan lane refs · **G5** zombie_rung0 closed loop drift 7e-7 @ 5 s, torque
  1.0000. `Testbed_Mixed.unity` (`MeetTestbed.BuildMixed`, `scene_meet8_mzmzmzmz.xml`): **G0 8/8** (each lane == its body's
  solo athlete) · **G6 8/8** (`make_g6.py mixed g6mixed`: Rung 2 commands in MATT units, zombie lanes × √λ exactly as
  `PolicyRunner.BodyCommand`; 1 s drift ≤ 4e-6, height RMS ≤ 4e-4 m, same outcomes). Lineups: roster scenes (backlog).

## Phase G — GRANDMA: third athlete body (added 2026-09-30)

User decisions (2026-09-30): Claude rigs the unrigged scan in Blender (no AccuRig / Mixamo), **1.60 m**, personality
**"frail but steady"**; start with the easiest rung (Rung 0). Body profile `bodies.BODIES["grandma"]`: 65 kg (stocky),
λ = 0.871, strength 0.6 of the size-scaled torque, full self-collision, slightly stooped soft-kneed stance.

- [x] **G1 Rig** — `SourceArt/Grandma/rig_grandma.py` (run in Blender): `test_GRANDMA_riggedTenCent.glb` has no skin
  (mesh + texture only) → weld the 451 UV-seam islands (6236 duplicate verts, else bone heat fails), symmetric 22-bone
  Mixamo skeleton from mesh cross-sections, automatic weights (0 unweighted verts), legs straightened (knee 19.4°)
  and baked as rest, two-pass scale → 1.600 m, soles on z = 0 → `SourceArt/Grandma/grandma.glb` + `rig_report.json`.
- [x] **G2 Body pipeline** — extract_skeleton (22 joints, mirror 0 mm, A4 PASS), build_mjcf (pelvis fitted to all
  hip-slab skin: bone heat gives the buttocks to the thighs), `grandma.xml` / `scene_grandma.xml` (65.0 kg, root z
  0.678 m), fingerprint, `contract_grandma.json`; test_model 13/13 with POOLYMPIC_BODY=grandma.
- [ ] **G3 Rung 0** — `tasks/grandma_env.py` (MATT's Rung 0 recipe, Froude-scaled like the zombie; stoop 10° +
  steady stance style rewards), task `PoOlympic-Grandma-Rung0-Stand` (C1 PASS). Run g0_v1 (from scratch, 1500 its,
  gates 0 at 500 / 1000 / 1499) chained after rs_v6. *Accept: G1 rung0 10/10.*
- [ ] **G4 Unity** — visual binding + Testbed_Grandma (G0 / G2-G5), roster card, mixed lineups.
- [ ] **G5 Rung 2 grandma** — short careful steps, size-scaled envelope.

## Backlog (later rungs & platforms)

  - **Menu lineups (roster scenes):** `compose_mixed.py <scene> roster` puts MATT (L<k>_) + zombie (Z<k>_) in every lane of every event scene (verify: every roster athlete == its one-body composition, 18/18); `LaneLineup` keeps the menu's pick per lane before MuJoCo compiles (bodies, actuators, excludes off), events drop the others, camera/pool follow. Zombie commands = MATT's × √λ (speeds) / ÷ √λ (yaw), `PolicyRunner.BodyCommand`; gauntlet gusts × √λ; shaker is the lane's (L<k>_shaker). All 9 events rebuilt; mixed 30m Dash + Gust Gauntlet played clean.
- [x] Android: arm64-v8a MuJoCo build via NDK + ARM↔desktop trajectory parity + mobile perf pass
  - [x] **first device run (Pixel 9 Pro, Android 17):** `tools/build_mujoco_android.sh` builds libmujoco.so from MuJoCo 3.11.0 with Unity's NDK r27 (API 28, `_POSIX_C_SOURCE`; mujoco-bin's prebuilt .so is 3.5.0 = wrong ABI for the 3.11 plugin), `AndroidBuild.Build` (IL2CPP arm64, portrait, min API 28, link.xml) → 167 MB APK. URP post-processing shader stripping had to be turned off (stripped Uber pass = blank 3D view). Event 1 heat ran on device (winner M1 after 31.2 s). Open: ARM↔desktop trajectory parity, perf pass, InputSystem 'InputUpdateType.None' warning at startup.
  - [x] **ARM↔desktop parity + perf (2026-09-29):** parity APK (`AndroidBuild.BuildParityApk`, package
    com.poolympic.parity, `DeviceProbe`): the pedestal G5 attempt recorded on the Pixel 9 Pro vs the desktop CPU reference
    → drift 1.09e-6 @ 5 s, torque ratio 1.0000001 — **G5 PASS on ARM**. Perf in an 8-athlete Steeplechase heat: 57.6 fps,
    frame p50 16.7 / p95 16.9 / p99 33 ms, mj_step 0.30 ms (8 athletes), 0.10 ms solo (`parity/android/device_probe.json`).
    InputSystem warning: not reproduced (cold start of the game APK and the parity APK); the only startup error is
    Unity probing for Play Asset Delivery classes (unused).
- [ ] R3 get-up · R4 ramp/rubble/stairs · R5 jumps/hurdles
  - R3 first attempts 2026-09-29 (getup_v1-v5, tasks `PoOlympic-Matt-Getup..5`, `tools/getup_probe.py`): with a fading
    torso assist it learns to sit up and to get onto all fours, not yet to stand. Next: reverse curriculum from
    squat / kneel starts, or a motion prior (R7).
- [ ] R6 bodies: ~~rig GRANDMA~~ (→ Phase G), clean + rescale ZOMBIE, derive MJCFs, train variants
- [ ] R7 optional motion-prior polish · R8+ remaining skill events
- [x] Game layer: betting slip & odds from lane stats, PBP ticker, records, gauntlets (2026-09-29): virtual-coin
  `Wallet` (100 start, 10-coin stake, top-up when broke), betting slip before every heat (12 s window; **off by default since 2026-09-29, user: "just play"** — `BroadcastHud.offerBets`, odds column still shown), `Records` (best
  winning mark per event, PlayerPrefs), `Gauntlet` series (menu GAUNTLET toggle: tap events to order them; one heat per
  event, 10-8-6-5-4-3-2-1 points per lane; final podium back on the menu)
- [x] **Demo (attract) mode** (2026-09-30, user ideas 1 2 3 4 8 9): `Runtime/DemoMode.cs`. Main menu idle 45 s (or ☰ →
  Demo mode) → endless Gauntlet of the 11 playable events, shuffled per series, random lineup per series (50 % mixed,
  25 % all one body, 25 % rivalry: 4 v 4 / one among 7); result card counts down "Next event in 6" and advances itself;
  after the last event the series goes into the season table (PlayerPrefs: series, wins per body, points per event per
  body; result box + menu) and the next series starts; any tap / key / pad button exits to the menu. `DemoRunner`
  watchdog skips a stalled event (no board 20 s, Ready 30 s, live > 180 s, result not advanced, 10 exceptions, scene >
  300 s). Verified in Play: idle start after 46 s, full 11-event series in 7.5 min with 0 skips, rollover into series 2,
  injected touch exits to MainMenu; EditMode `DemoModeTests` 3/3 (+ RuntimeRules, BroadcastFx, PortraitLayout,
  Showcase, Steering all pass). Before/after: `parity/ui/demo_before_after.html`.
  - Found by the watchdog: Gust Gauntlet threw on every shake round in Unity (`mj_name2id("shaker_x")` misses the plug-in
    name `shaker_x_255` → per-lane `L5_shaker_x` fallback) and All Fours read `xmat[-1]` for the torso tilt (same cause).
    Fixed with `ModelFingerprint.Id` (suffix-tolerant lookup).
- [x] **Calm camera** (2026-09-30, user: "fast movements hurt my eyes"): `BroadcastDirector` "Calm camera" fields,
  applied at runtime (no scene rebuild; tune in the Inspector): 2.5 s ease-in/out blends between nearby cameras (≤ 15 m,
  ≤ 45°), cuts between far ones (a 2.5 s blend to the podium flew the camera at 75 m/s) landing on a camera at rest;
  framing targets glide (move with the athlete's average velocity, stride sway / bob filtered 0.5 / 1.2 s, ≤ 2.5 m/s
  slide to a new athlete); close-ups orbit ≤ 20°/s after a spinning athlete; damping floors 0.6 / 0.5 s (close-ups
  half); hand-held noise ≤ 0.15; no impact shake; shots 1.5× longer. `Editor/CameraMotionProbe` (per-frame camera
  speed): fast frames (> 60°/s or > 12 m/s) 21 % → 0.0 % (Turntable), 15 % → 0 % (Terminal Velocity), 0 in Slalom,
  All Fours, Iron Pedestal; turn p95 144 → 0 °/s.
- [x] **Broadcast FX** (features 2, 3, 5, 6, 7, 10 + world records; 2026-09-29). Free add-ons: Cinemachine 3 (6.6.0),
  Visual Effect Graph 17.6.0 (+ "Additions" sample), PrimeTween 1.4.11 (npm), Kenney CC0 sounds + Gregor Quendel crowd
  (CC-BY 4.0, credit required — docs/LICENSING.md). All render / audio only, read from mjData (parity untouched).
  - `AthleteTelemetry` per athlete (after every mj_step): joint stress |τ|/limit, power Σ|τ·ω|, speed + peak, cadence,
    ground contact, CoM vs support polygon (balance margin), impacts (foot strike / body slam / cube hit, peak contact
    force over 30 ms), brain confidence
  - **5 brain confidence**: `tools/export_critic.py` (critic ONNX beside each event brain) + `tools/fit_confidence.py`
    (P(still up in 2 s), AUC 0.62-0.84, see log); PolicyRunner runs the critic at 10 Hz, never touching ctrl
  - `TensionMeter`: danger per athlete (confidence, smoothed balance, tilt), race closeness, athletes out → tension 0..1,
    hot athlete, near-fall / save / fall events (debounced)
  - **2** `BroadcastDirector` on Cinemachine: 5 CinemachineCameras (establishing/trackside, head-on, high-wide, hot close-up
    with hand-held noise, winner) following director-moved proxies; tension shortens race cuts; a real near fall eases to
    the athlete in trouble (5 s cooldown); phase changes hard-cut; scenes without the rig keep the old framing
  - **3** `ImpactFx`: pooled foot dust / slam dust + shockwave (Shuriken), VFX Graph sparks at cube hits (Shuriken
    fallback without compute), Cinemachine impulse shake, 70 ms hit-stop (timeScale 0; physics just pauses)
  - **6** `BalanceOverlay`: support polygon + CoM ring + plumb line (green / amber / red); stationary events all athletes,
    races only the hot athlete
  - **7** HUD: CONF column (sparkline + %), stats card (PrimeTween pop) for the hot athlete / leader, telemetry
    commentary (top speed, "is wobbling", "What a save"), result card heat bests
  - **10** `ArenaAudio`: crowd bed + rhythmic tension layer following tension, reactions on near falls / falls / saves,
    cheers at the result and for records, 3D thuds / cube hits by contact force, hot athlete's footsteps, countdown ticks
  - **World records**: `Records` keeps the top 5 per event (holder, body, heat, date; old single record migrated);
    result card WORLD RECORDS panel (top 3, new record pulses, else the record to beat); main menu WORLD RECORDS board
    (tap an event for its top 5) + the selected event's record under its rules
  - built into every event scene by *PoOlympic › Broadcast › Upgrade broadcast FX* (`BroadcastFx.Install`, also called by
    `EventScenes.AddBroadcast`); EditMode 47/47 PoOlympic tests (5 new: records, hull geometry, confidence model, scene
    rig + no PhysX); before/after `parity/ui/broadcast_fx_before_after.html`
  - **Docked HUD** (user, 2026-09-29: "HUD on top and bottom so the gameplay is not covered"): top dock (title, clock,
    standings) + bottom dock (one-row stats card, ticker, buttons) on the screen's safe area; the camera renders only into
    the gap (`BroadcastCamera.SetViewport`, replaces the 9:16 letterbox in event scenes). Only the countdown banner and the
    result card / betting slip sit over the game; the winner close-up is framed below the result card. Before/after
    `parity/ui/dock_before_after.png` (in the same HTML page)
  - [x] Android: rebuild the APK and re-measure (VFX Graph, 8 critics at 10 Hz, audio) on the Pixel 9 Pro — done
    2026-09-30 together with the showcase re-measure below
    (`parity/android/perf/Event_SteeplechaseJog_20260930_133623.csv`)
- [x] **Showcase pass — GFX / sound top 10** (2026-09-29, user: "do all"). Blender half `SourceArt/Stadium/build_showcase.py`
  (run after build_polish; re-exports Stadium.glb, 125k tris, 240 lane anchors asserted unchanged), Unity half
  `Editor/StadiumShowcase.cs` (*PoOlympic › Stadium › Upgrade showcase (all scenes)*, called by StadiumLook.Dress +
  BroadcastFx.Install). All render / audio only (parity untouched).
  - **1 lighting**: crowd-wash fixtures re-aimed between the tiers + 115°/30° soft cones (no more 12 ovals); baked GI
    (`Settings/ArenaLighting.lighting`: GPU lightmapper, 3 texels/m, non-directional, AO) with the 36 arena spots baked,
    key light realtime; 504-probe grid over the event area; ~40 s bake per scene, 1 lightmap
  - **2 screens**: centre-hung video cube + both scoreboards on `Screen_Live` → `ScreenFeed` (UI Toolkit into a
    1280×448 render texture: standings | live feed camera of the story athlete at 1/3 rate | event + clock; winner /
    WORLD RECORD banners); the cube shows the centre 840 px
  - **3 crowd**: 4,704 one-metre crowd cards on the rows (the fans painted on 0.4 m risers read as thin lines from
    trackside), seated + cheering textures of the same 64 seats (`Art/Stadium/Crowd/`), `PoOlympic/Crowd` shader (URP
    Simple Lit lighting + lightmaps): per-seat stand-up / bob from `CrowdDirector` (tension, bursts at the start / saves
    / result / records, Mexican wave in quiet spells, camera flashes)
  - **4 roof**: catwalks on the four long trusses, 8 hanging line-array speakers (`Speaker_##`), acoustic ceiling
    panels, cube cables; 24 light-beam cones (`PoOlympic/LightBeam`: additive, fresnel edge, near / fog fade)
  - **5 infield**: pads = granule sports floor tinted a muted phase colour (world-scale UVs) + a thin saturated border
  - **6 podium**: 1-2-3 podium + 3 flag poles on the free infield D; `PodiumCeremony` (2.5 s into the result: photo
    statues of the top 3 = SkinnedMeshRenderer.BakeMesh on the steps, flags up, confetti, fanfare, flashes;
    `BroadcastDirector` cuts to `CM_Podium` for 5.5 s)
  - **7 branding**: Olympic rings, facade identity, cauldron + surroundings → non-exported `Offstage` collection
    (docs/LICENSING.md updated; the "POOLYMPICS" name is still the open blocker)
  - **8 audio**: `Audio/PoOlympicMix.mixer` (Crowd / Sfx / Ui / Announcer; snapshots Ready, Live, Result, Announce =
    duck), 4 spatial crowd sectors, PA announcer (31 lines, Windows TTS; chime, echo; marks / set / lead changes / out /
    save / photo finish / winner / record / next heat), starter gun / whistle / horn / buzzer / fanfare / shutters
    (synthesised), arena reverb zone, listener low-pass during hit-stop
  - **9 HUD** (one screen, portrait): 4 standings rows (top 3 + the story; tap for all 8), ODDS only with betting,
    CONF only under 90 %, ≥ 28 px text at 1080, 3 stats tiles, 2-line ticker, 104 px buttons, one action row (result
    card no longer repeats New heat; New heat turns green at the result), safe-area strips in the dock colour.
    Before / after: `parity/ui/showcase_before_after.html`
  - **10 perf**: `PerfOverlay` (F3 / three-finger tap / tap ⓘ): fps, CPU / GPU ms, sim ms (FixedUpdate bracketed in
    the player loop), draw calls, Android thermal + battery °C, quality tier; CSV per second in persistentDataPath/perf;
    auto quality on phones (render scale −0.1 → beams off + screen feed ⅓ → critics 5 Hz + shadow distance ½)
  - **camera + card follow-up** (user: "camera on the current winner and anyone close to falling", "too many cuts",
    "closer so you can see the face", "back up so the whole body is in the picture", "not from behind the stands",
    stats card "no more than once a second"): `TensionMeter.Leader` = the live ranking's first row, ties (everyone
    "IN") broken by the steadiest athlete (3 s smoothed danger, held ≥ 5 s); `BroadcastDirector` live = full-body front
    close-up of the winner (`CM_Leader_Closeup`, 3.2 m, 42°, facing from the pelvis yaw) 8 s ↔ one context shot 5 s,
    ≥ 4 s per shot, near-fall cuts at danger ≥ 0.7 / 8 s apart, `ClampToBowl` keeps every camera inside the track's
    outer edge; `Subject` drives the stats card (speed / power / confidence, 1 Hz) and the stadium screens
  - [x] Android: rebuild + re-measure with the showcase (2026-09-30): APK 233 MB (8.5 min build, 14 scenes). 8-athlete
    Steeplechase heats on the Pixel 9 Pro, 124 s (4+ heats, auto-cycling): **60 fps** every second but the scene-load one
    (p50 60, CPU frame 16.7 ms = vsync-bound, **GPU 9.2 ms** mean / 10.7 max, sim 5.4 ms = physics + 8 brains + 8 critics),
    thermal status 0, quality tier 0 throughout, battery 30.5 → 33.7 °C (baseline 2026-09-29: 57.6 fps). Found: the CONF
    column shows 0 % for every athlete at the start line — the r2f_v3_it100 confidence fit (w_speed 4.3, b −17) was
    calibrated on running samples only, so a standing athlete (cmd 0) extrapolates to ~0 %; not a device issue (open).
- [x] **UI consolidation — portrait top 10** (2026-09-29, user: "do all"): one viewport, nothing scrolls, no dropped
  features. Before / after with annotations: `parity/ui/consolidate_before_after.html` (shots in `parity/ui/consolidate/`)
  - **1 anchors** `HudAnchors` (+ `UI/HudAnchors.uss`): one 5-slot frame on every UI Toolkit screen — TL title + one-line
    subtitle · TC FPS pill (0.5 s average, amber / red under 90 % / 50 % of the target) · TR ☰ sheet (+ `Extra` slot:
    the race clock) · BL debug · BR version. The IMGUI dev HUDs (`EventHud`, `TestbedHud`) use the same corners
  - **2 perf** `PerfOverlay.Line`: the readout shows in the BL debug slot (grows up from the corner, above the bottom
    dock); the overlay's own label only on screens without the frame (now bottom-left too)
  - **3 actions** New heat / Main menu / all 8 · top 4 / performance → the TR sheet; the bottom bar holds only the primary
    action at the result (New heat, or Next event in a gauntlet); coins only with betting on
  - **4 result in place** the result card overlay is gone: rows 1-3 gold / silver / bronze, WR badge on a record winner,
    one world-record line (+ "#n ALL-TIME" / "NEW RECORD!" pulse), heat bests as TOP / PWR / CLOSE badges, bet and
    gauntlet lines; the winner banner clears after 2.5 s (`winnerBannerSeconds`)
  - **5 + 6 story strip** stats card (3 tiles) + ticker box → one 64 px strip (chip · lane · tag · speed · power ·
    confidence + sparkline; refreshes at once on a phase change) and a 2-line ticker in the bottom bar
  - **7 menu roster / lanes** roster cards = "all 8 lanes" (×8), lane tiles 4 × 2 (tap = next athlete); no ROSTER / LANES
    headings, no hint, no "Fill all 8"; the ☰ sheet adds All MATT / All ZOMBIE / Alternate
  - **8 no scrolling** event list → fixed 3-column grid that fills the rest of the 1920-unit column; the WORLD RECORDS
    screen → the ★ toggle (world record on every tile, the selected event's top 5 in the fixed-height detail panel);
    status line → the title chip's subtitle; the menu keeps to the safe area
  - **9 guard** `HudLayoutAudit` + `Tests/Editor/PortraitLayoutTests` (8 cases): menu + HUD at 1080×1920, Pixel 9 Pro
    (with insets), 720×1280, 1080×2400 — no ScrollView, nothing outside the safe area, text ≥ 28, anchors in their
    corners, 104 px buttons / tiles, camera gap ≥ 60 % live / ≥ 55 % at the result. EditMode 58/58
  - **10 tokens** `UI/Tokens.uss` (`--po-text` 28, `--po-btn` 104, `--po-chip`, spacing, colours) imported by all
    three style sheets; the menu's 16-24 px text is gone
  - **no slow motion during the race** (user, same day): `ImpactFx` hit-stop (70 ms timeScale 0) is off while a heat is
    live (`hitStopWhileLive`, default off; shake + effects still play); its release moved from a WaitForSecondsRealtime
    coroutine to Update (the coroutine never resumed under editor single-stepping and froze the heat)
  - [x] Android (Pixel 9 Pro, APK 228 MB): menu + heat fit the safe area, result in place, FPS pill + debug readout
    work (`parity/ui/consolidate/android/`). Found there: the game runs at Unity's 30 fps phone cap, but PerfOverlay's
    auto quality compared against 60 → "throttling" on a cool phone walked it to tier 3 (render scale −0.1, beams off,
    critics 5 Hz, shadows ½). `PerfOverlay.TargetFps` (targetFrameRate, else 30 on phones / 60 elsewhere) → Q0 at 33 °C
- [x] Licensing review (Avaturn, Hunyuan3D) before any commercial release → `docs/LICENSING.md` (2026-09-29). Blockers
  to act on before a store release: Olympic rings / name, Hunyuan3D territory (EU / UK / KR), Avaturn notification +
  attribution; checklist in the doc
