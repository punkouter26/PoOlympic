@AGENTS.md

# Working in this repo

## Layout
- `Assets/PoOlympic/Runtime` game code (events, `PolicyRunner`, HUD, camera, audio) · `Editor` scene builders (`EventScenes`, `BroadcastFx`, `StadiumShowcase`) · `Tests/Editor` EditMode tests · `UI` USS / UXML (`Tokens.uss` = the design tokens).
- `training/poolympic` contract, tasks, events (Python mirror of the Unity events) · `training/tools` queue, gates, exports · `training/runs` checkpoints (git-ignored).
- `parity/` gate reports, brains (`parity/brains`), review pages (`parity/tb/<run>`), UI captures (`parity/ui`; UI before/after reports go in `SCREENSHOTS/`).
- Brain names: deployed = `<role>_<body>.onnx` (`crawl_kim`, `slope_matt`; MATT's oldest are `rung0` / `rung2` / `rs_v8_it1199`) + `.critic.onnx`, listed in `parity/brains/manifest.json`; candidates = `<run>_it<N>.onnx`, never copied into Unity (`ParityHarness.SyncArtifacts` syncs only brains the manifest or the C# names).
- History: `docs/log/` (September log entries, finished task sections); the root `rl_optimization_log.md` / `tasks.md` hold the current ones.

## Commands
- Fresh clone: `git lfs pull`, then `cd training; uv sync`.
- Python tests: `cd training; uv run pytest tests -q`. After editing `poolympic/tasks/__init__.py`: `pwsh tools/check_bodies.ps1` (imports the tasks once per body); after touching shared task code: `uv run python tools/dump_task_cfgs.py --diff` (existing task cfg hashes must not move).
- After deploying a brain: `uv run python tools/brain_manifest.py` (`--check` verifies `parity/brains/manifest.json`).
- Training: add a line to `training/runs/queue.txt`, start `training/tools/train_queue.ps1` (and `keep_awake.ps1`) detached; gates run from `tools/watch_gate.py`.
- Checkpoint clean-up: `pwsh training/tools/prune_checkpoints.ps1` (dry run; `-Delete` to act).
- Unity (editor open): `unity command eval_file --file x.cs`, `unity command menu --path "PoOlympic/…"`, tests `unity command run_tests --mode EditMode --filter PoOlympic.Tests --async_tests true` then poll `test_status`.
- Scene rigs are built by editor code, not by hand: after changing `BroadcastFx.Install` run the menu "PoOlympic/Broadcast/Upgrade broadcast FX (all event scenes)". To move something by hand in an event scene and keep it across rebuilds, add the `HandTuned` component to it (`HandTunedSave.SaveBuilt` restores its local transform by hierarchy path).

## Things that bite
- Play mode starts **without a domain reload**: every static needs a `[RuntimeInitializeOnLoadMethod(SubsystemRegistration)]` reset (see `DemoMode`, `MeetLineup`, `GameSettings`).
- Runtime code may not call `Instantiate` / `Destroy` or any PhysX API (`RuntimeRulesTests`); new scene objects go through the editor builders as pools.
- Brains are tied to the model fingerprint and observation contract: never change `training/assets/*.xml`, `contract.py` dims or an existing task's config in place — add a new task ID / contract version.
- Scenes are loaded only through `MeetLineup.Load`.
