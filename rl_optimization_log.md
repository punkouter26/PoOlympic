# RL Optimization Log

One entry per run or decision. Newest at the bottom.

<!-- Template
## YYYY-MM-DD · <run-id or DECISION>
- **Change:** what was changed (config diff, code, asset)
- **Hypothesis:** why
- **Result:** metrics (TensorBoard tags, eval_cpu report path, gate report path)
- **Decision:** keep / revert / next step
-->

September 2026 entries (A1 .. the 2026-09-30 runs): [docs/log/rl_optimization_log_2026-09.md](docs/log/rl_optimization_log_2026-09.md).

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

## 2026-10-01 09:05 · rs_v7 finished: march 9/10 (spot-keeping works); flamingo still 0/10 → rs_v8 queued (entropy)
- **rs_v7** (rs_v6 it3299 + `skill_spot` + `skill_flamingo_lift`, 1500 its, 95 min; 4.7 s/it at the start with another
  project's 8192-env run on the GPU, 2.8 s/it at the end). Gates (G1 S per drill · Rung 2; 0 falls everywhere; flamingo
  drill now 10 different trials):

  | it | squat | flamingo | march | torso | reach | Rung 2 | march drift (m) | march lift / called |
  |---|---|---|---|---|---|---|---|---|
  | rs_v6 3299 (start) | 10 | 0 | 4 | 5 | 0 | 6/10 | 0.17-0.90 | 0.83-0.97 |
  | 400 | 10 | 0 | 8 | 4 | 0 | 5/10 | 0.21-0.33 | 0.89-1.00 |
  | 800 | 10 | 0 | 9 | 6 | 0 | 6/10 | 0.11-0.23 | 0.88-0.99 |
  | 1200 | 10 | 0 | 1 | 5 | 0 | 7/10 | 0.14-0.47 | 0.98-1.05 |
  | **1499** | 10 | 0 | **9** | 5 | 0 | 6/10 | **0.06-0.14** | 0.99-1.09 |

- March: the spot term fixed the drift; the one miss at 1499 is cadence +6 % (bar ±5 %; cadence is counted from knee
  peaks). Not monotonic (1/10 at it 1200 as the lift overshot). Squat worst error 1.4 cm.
- Flamingo: 120-166 touch ticks, 11-26 cm slip at every gate; `skill_flamingo_lift` flat at 0.17 of 1.5. Starting from
  a checkpoint that had lost the skill did not recover it. Reach flat (median 0.23-0.27 m). Torso 4-6/10.
- track_ang 0.83 → 0.71; action std 0.50 → 0.54 at entropy 0.0025 (rs_v6: 0.44 → 0.50). Same noise growth as GRANDMA's
  g2_v2b, where entropy 0.001 + capped std stopped it → suspect for the locomotion erosion (untested here).
- **Brains:** rs_v7 it 1499 → Events 3 + 7. rs_v6 it 2550 → Event 6 (if per-event brains are accepted). rs_v5c it 2300
  best torso (8/10). Review: `parity/tb/rs_v7/review.html`.
- **rs_v8** queued after r0bio_v1: `-RungS-Stance-v8` = rs_v7's env, entropy 0.001, `plain_init --max-std 0.3`, from
  rs_v7 it 1499, 1200 its, gates 300 / 600 / 900 / 1199. One question: does lower action noise stop the erosion (and
  does flamingo come back)?

## 2026-10-01 09:25 · Second Ctrl-C kill (09:09, PowerShell queue) → queue hardened: auto-resume, own consoles
- r0bio_v1 (started 09:04) and the queue died at 09:09:41, `^C` in `runs/queue_launcher.log` again, 5 min into the run
  (the 23:43 kill was 4.5 min into g1_v1). Found at 09:20 by the stall watcher (13 min lost). The PowerShell queue had
  survived about fifteen turn ends in between and four runs; PoDecath's run had ended at 08:39. **Cause not found.**
  r0bio_v1 it 100 gate before the kill: G1 rung 0 10/10.
- `tools/train_queue.ps1` now: ignores Ctrl-C itself, starts training and watcher in their own hidden consoles, and
  **resumes a run that ends early from its newest checkpoint** (up to 3 times; second `*_<name>` run dir, iteration
  numbers continue, watcher restarted for the remaining gates, stale watcher stopped). Tested by killing r0bio_v1's
  trainer on purpose at it ~96: RESUME from model_100.pt, continued at it 128/300.
- Queue restarted 09:22: r0bio_v1 → rs_v8 → r2fbio_v1 → crawl_bio_v1 → getup_rev_v1. GPU no longer shared (1.95 s/it).

## 2026-10-01 10:50 · End of the training block (user: "finish up", git sync)
- **r0bio_v1** (mattbio Rung 0, 300 its, resumed once at it 100 by the deliberate kill test): G1 rung 0 **10/10** at it
  100 / 200 / 299.
- **rs_v8** (rs_v7's env, entropy 0.001, start std <= 0.3, from rs_v7 it 1499; 1200 its, 62 min): action std 0.27-0.28
  (rs_v7 0.50 → 0.54), mean reward 158 → 168 (rs_v7 142), training track_ang 0.82 → 1.16 (rs_v7 0.83 → 0.71). Gates
  (squat / flamingo / march / torso / reach · Rung 2): it 300 10 / 0 / 5 / 6 / 0 · 6/10 (1 fall) | 600 10 / 0 / 6 / 6 / 0 ·
  6/10 | 900 10 / 0 / 4 / 5 / 0 · 5/10 | **1199 10 / 0 / 9 / 7 / 0 · 5/10** (squat worst 0.8 cm, march drift 0.06-0.17 m).
  **The noise hypothesis explains the training curves, not the gates:** Rung 2 stays 5-6/10 and flamingo 0/10 with the
  noise gone. March drift hovers at the bar between gates (4-6/10 at 300-900). Torso misses are the same two seeds
  with the same 0.20-0.21 rad error at every checkpoint since rs_v6 (targets possibly outside the joint range;
  unchecked). Review: `parity/tb/rs_v8/review.html`. Best single stance brain: **rs_v8 it 1199** (Events 3 + 7, torso
  7/10); flamingo only in rs_v6 it 2550.
- **r2fbio_v1** (mattbio Rung 2 flight, 300 its): G1 rung 2 9/10 at it 100 and 200, **10/10 at it 299** (0 falls).
- Queue stopped by `runs/queue.stop` after r2fbio_v1. **Not run:** crawl_bio_v1, getup_rev_v1 (still in
  `runs/queue.txt`; delete `runs/queue.stop` and restart `tools/train_queue.ps1` to run them).
- **Block summary (23:10 → 10:50):** 13 runs started, 9 completed, 4 stopped on purpose after a gate. ~2 h 15 min of
  GPU time lost to two unexplained Ctrl-C kills (23:43, 09:09). mattbio: Rung 0 10/10, Rung 2 10/10, flight 9/10 →
  promotion to MATT now has brains for stand / walk / flight; crawl + get-up still to train. GRANDMA: Rung 1 8/10,
  Rung 2 provisional 4/10 (decision pending on her envelope). MATT stance: march 9/10, squat 10/10, torso 7/10 on one
  brain; flamingo on another; reach unsolved.

## 2026-10-01 afternoon · User decisions applied (GRANDMA envelope + stride yaw, per-event stance brains); no training
- **Decisions (user: "do recommendations"):** (1) GRANDMA's envelope tops out at 2.8 m/s; (2) her yaw is judged per
  stride; (3) MATT's stance events use per-event brains.
- **Gate code:** `bodies.Body.vx_max` / `yaw_per_stride` → `evaluate.RUNG2_VX_MAX`, `BRAKE_V`, `rung2_checks` (the
  zombie's 2026-09-29 stride ruling is now the pass criterion too, not report-only). Contract `vx_max` (GRANDMA only) →
  Unity `PolicyRunner.BodyCommand` and Python `body_command` cap the forward command.
- **GRANDMA Rung 2 re-scored** (G1, new bars, `parity/watch/g1cap_rung2_*.json`): g2_v5 it 300 3/10 · it 600 4/10 · it 900
  4/10 · **it 1199 5/10**; g2_v1b it 1000 2/10; g2_v4 it 300 3/10. 0 falls, yaw now passes on every seed, turntable /
  brake / backward 10/10; every miss is `tracking_lin` (sprint segments 0.21-0.37, crab 0.20-0.34 vs 0.187). The rule
  change alone is worth one seed: precision still needs training.
- **Torso aim check** (rs_v8 it 1199, CPU): the three failing seeds (1001, 1003, 1007: 0.21 / 0.20 / 0.18 rad) are aims
  of yaw 57° at low pitch. The command range is ±60° but `abdomen_twist` stops at ±45°; a grid sweep of the three abdomen
  joints gives a best possible error of 0.14-0.19 rad for those aims and ≤ 0.06 for every passing aim. Not a training
  problem; narrowing the commanded yaw to about ±50° is a contract-range change (user).
- **Per-event stance brains, measured on the events (CPU heats):** Deep Squat with rs_v8 it 1199: 0 finishers in 4 heats
  (everyone STEPPED at 20-37 s); with rs_v6 it 3299: 1-4 finishers per heat, 80-104 pts → Event 3 keeps rs_v6 it 3299
  although both pass the squat drill 10/10 (the drill squats to 40 cm with a 5 cm bar, the event to 45 cm with a 20 cm
  foot-slide rule). Cadence March with rs_v8 it 1199: rhythm exact at every stage, 2-4 finishers; with rs_v7 it 1499:
  half the field wanders off by 18 s → Event 7 uses rs_v8 it 1199. Event 6 = rs_v6 it 2550 (event not built yet).
- **Brain confidence:** critics exported for grandma_rung0 / grandma_rung2 / rs_v8_it1199; `fit_confidence.py --groups
  flight,stand_g,rung2_g` — grandma_rung0 AUC 0.70, grandma_rung2 0.67, r2f_v3_it100 0.87. The flight brain's fit now
  includes standstills (1 command in 4): w_speed 4.32 → 0.73, b −17.0 → −0.42, so a runner standing on the start line no
  longer reads 0 % (the 2026-09-30 Android finding).
- **Odds** refitted with a GRANDMA weight (40 CPU heats per event on the MATT / zombie / GRANDMA lineup): she is the
  underdog in the inverted sprint (−14.9), terminal velocity (−9.5) and crab (−8.5), neutral on the pedestal, turntable
  and brake.
- **Not run:** crawl_bio_v1, getup_rev_v1 (still in `runs/queue.txt`).

## 2026-10-01 14:10 · Training block started: crawl_bio_v1 → getup_rev_v1 (Unity closed)
- Unity EditMode before closing the editor: full run 801 tests, 791 passed. The 8 `Mujoco.*` failures are the plug-in
  package's own tests. PoOlympic: all passed except (1) `EventScenesCarryTheBroadcastFxLayer` — the Deep Squat runners
  had no critic (since Event 3 was built; `rs_v6_it3299.critic.onnx` exported and attached now) and (2) the crowd-contact
  case of Event_InvertedSprint, which checked all 145,656 part pairs with 0 failures but took 289 s against the runner's
  180 s default (`[Timeout(900000)]` added). Crowd contacts with GRANDMA in the roster: 145,656 cross-lane part pairs
  per 24-athlete scene, 0 failures in every scene.
- The re-run of the PoOlympic assembly did NOT finish: the Iron Pedestal crowd case, 98 s in the first run, ran ~35 min
  (one core busy, free RAM 3 GB, a second Unity project open) and the editor was force-closed while it was on the next
  scene. Both fixes are therefore unverified by the test runner: re-run `BroadcastFxTests` and `CrowdContactTests` when the
  editor is open again.
- Housekeeping: 588 MB of superseded TensorBoard runs moved to `runs_archive/2026-10-01/` (rs_v1-v5b, g2_v2b / v3 / v4);
  TensorBoard on `training/runs` (:6006); queue restarted through WMI at 14:10 (2.5 s/it at the start, GPU 85 C / 58 W
  with another project's Unity editor open and the fans on Auto).

## 2026-10-01 15:20 · Self-collision rule (AGENTS.md) audited and fixed: thighs now collide on the zombie and GRANDMA
- **Rule (user):** simple shapes fitted inside the mesh; every pair collides except parent-child and pairs overlapping at
  rest; no contact in the T-pose / default stance / a normal swing; self-contact never ends an episode; prefer warm starts.
- **Audit** `tools/check_self_collision.py` (135 part pairs per body): mattbio PASS (118 collide, thighs off: they overlap
  6.4 mm at rest). MATT FAIL as known (97 pairs off: legs-only). Zombie and GRANDMA: the thigh pair was excluded although
  the thighs are 17 mm / 58 mm apart at rest. No body has a self-contact in the T-pose, the default stance or a 21-phase
  arm and leg swing. Terminations: time-out, pelvis height, torso tilt, non-foot ground contact, off pedestal — none on
  self-contact.
- **Fix:** `build_mjcf` tests every pair of a self-colliding body, the thighs included (`touching_pairs`); mattbio.xml is
  byte-identical, zombie.xml / grandma.xml lose their one `<exclude>`. All composed scenes with a zombie or GRANDMA
  regenerated (9 roster, 9 MATT / zombie, 7 MATT / zombie / GRANDMA); roster verify 35/35, crowd audit 22/22, pytest 65/65.
- **Brains unchanged, no retraining:** G1 before → after on the same checkpoints: zombie Rung 0 10/10 → 10/10 (identical
  report), Rung 2 8/10 → 8/10 (stride yaw rule), crawl 5/5 in 31.7-31.8 s → the same; GRANDMA Rung 0 10/10 → 10/10
  (identical), Rung 2 4/10 → 4/10. With the old model the thighs never overlapped in 60 s of mixed commands (closest 1 mm
  zombie, 23 mm GRANDMA); with the pair on, one of the 13 references changed (g6mixed_L5, the zombie's −1.9 rad/s spin).
- **Parity gap closed:** the model fingerprint (schema 1) does not cover `<exclude>` pairs, so the hash did not change
  and G0 could not have seen a stale Unity scene. `fingerprint.excludes()` + the `excludes` key in Unity's dump are now
  compared in G0 without touching the hash (re-keying every brain was not worth it). Unity after the change: G0 zombie /
  grandma / meet8_mzmzmzmz 8/8 PASS incl. excludes; G2-G4 13/13 references PASS; the 11 event scenes with zombies or
  GRANDMA rebuilt and baked.
- AGENTS.md rule 5 reworded (user: "fix"): prefer a warm start; from scratch only for a body's first brain or where a
  warm start has been tried and failed (MATT's r1_v1).

## 2026-10-01 15:55 · Training block finished: crawl_bio_v1 good, getup_rev_v1 stuck at the kneel stage
- **crawl_bio_v1** (mattbio crawl, 600 its, 31 min, 2.5-3.2 s/it with the Unity editor of another project open; viewer shown
  100 s at the it-150 gate: 2.8 → 6.5 s/it while open). Solo gate (5 seeds, 30 m): it 150 5/5 in 26.0-27.2 s · it 300 5/5
  25.7-26.8 s · **it 450 5/5 25.4-26.5 s, lane 0.07 m** · it 599 5/5 26.1-29.6 s, lane 0.13 m.
  8-athlete event heats (4 seeds, crowd contact, mattbio x 8 on `scene_crawl8_mattbio` / `scene_trench8_mattbio`):

  | heats | finished | median time |
  |---|---|---|
  | 30 m, MATT `crawl_matt` (deployed) | 31/32 | 26.2 s |
  | 30 m, mattbio it 450 | 30/32 | 33.1 s |
  | 30 m, mattbio it 599 | 25/32 | 35.3 s |
  | trench 16 m, MATT `crawl_matt` | 29/32 | 18.1 s |
  | trench 16 m, mattbio it 450 | 29/32 | 22.0 s |
  | trench 16 m, mattbio it 599 | 31/32 | 17.4 s |

  Alone the mattbio crawler is as fast as MATT; in a field of eight it is ~7 s slower over 30 m (arms and trunks now
  collide between and within athletes). Provisional mattbio crawl brain: it 450. Not deployed.
- **getup_rev_v1** (mattbio, reverse curriculum, 1500 its, 69 min, Unity editor open from 14:53 on user request):
  stage 0 (squat) passed inside 100 its, then stage 1 (kneeling on both knees) at 0 % frontier success to the end; the
  assist faded out at it 600 without a success before or after. Gate (supine start): 0/10 at 300 / 600 / 900 / 1200 /
  1499. standing_tall ~2.0 (getup_v4: 0), mean reward 83 (getup_v4: 32, not like for like: 40 % of episodes start from
  the solved squat). The start poses have no self-contact. Kneel pose: feet 0.48 m behind, COM 0.44 m ahead of them —
  the way up is a half-kneel, which no stage teaches. No brain kept.
  **Next recipe:** a half-kneel stage between squat and kneel; the assist faded per stage, not once.
  Review: `parity/tb/getup_rev_v1/review.html`.
- Queue empty at 15:51. mattbio now has Rung 0, Rung 2, flight and crawl; get-up is the one brain missing for the
  promotion to MATT.

## 2026-10-01 16:55 · Event 6 The Flamingo Classic playable (no training: rs_v6 it 2550)
- **Brain probe** (CPU, 8 athletes with traits on `scene_flamingo8_mattbio.xml`, lift command at GO): the foot leaves the
  ground 0.3 s after the command and is held 16.6 cm up (nominal traits). Without disturbance about half the field
  stands on one leg for 120 s (some drift or hop up to 1.9 m), the rest tap the foot down within 80 s; athletes with
  high latency / noise tap within 1.5-6 s. Nobody falls, even under gusts of 1.5 m/s: they put the foot down and recover.
- **Rules** (`poolympic/events/flamingo.py` == Unity `FlamingoEvent`): everyone lifts the same foot (seeded per heat),
  LIFT_S 2 s to get it up, then a gust every ROUND_S 2 s of GUST_START 0.04 + GUST_STEP 0.012 m/s per round in the lane's
  own seeded direction. Out = TOUCHDOWN / HOPPED (stance foot > 0.30 m from its spot) / FELL. Mark = time on one leg; the
  heat runs until the last one is down (cap 120 s).
- **Tuning:** gust 0.1 + 0.05 every 3 s ended every heat at 9.5-12.9 s, bunched on the round edges. 0.04 + 0.012 every
  2 s (10 seeds): winner 22.8-38.5 s (mean 31), median exit 14-17 s, first out 2.3-11 s; 79 TOUCHDOWN, 1 HOPPED, 0 FELL.
  A longer lift time (3 s, 4 s) does not save the early tappers (they go out at the end of the grace instead): kept 2 s.
- **Unity:** `Event_FlamingoClassic.unity` (venue 6 grid, baked), hub + main menu (14 playable; the hub scene was patched
  in place, a rebuild drops its baked lighting and showcase dressing). Heat seed 1: S3 wins 38.1 s, identical on three runs.
  Critic exported for rs_v6_it2550 (not calibrated, like the other stance brains).
- **Parity:** G0 flamingo8_mattbio 8/8 lanes 0 mismatches; reference `flamingo_rs_v6` (left foot up, standard shove +
  cube): G2 1.8e-15, G3 4.3e-6, G4 4.1e-7 @ 1 s (first tick over tolerance 194, after the cube hit), **G5 PASS** in
  Testbed_Stance: 1 s drift 1.6e-6, 5 s drift 2.7e-2 (one-leg recovery from the cube is chaotic; march was 1.2e-5),
  pelvis height RMS 1.0e-4 m, torque ratio 1.0004, same outcome.
- **Tests:** pytest 66/66 (new `test_flamingo_heat`); compose --verify 35/35; Unity EditMode BroadcastFxTests 5/5 (this
  also verifies the morning's Deep Squat critic fix), PortraitLayout 8/8, DemoMode 3/3, Showcase 3/3, RuntimeRules 2/2,
  ContractV4 4/4, CrowdContact on Event_FlamingoClassic PASS and on Event_InvertedSprint PASS in 98.5 s (the morning's
  timeout case). The other crowd-contact scenes were not re-run today.
- Before / after page: `parity/ui/flamingo_before_after.html`.

## 2026-10-01 21:15 · Get-up solved on mattbio: getup_rev_v2e, 10/10 from supine at it 600-1499 (30/30 on 30 seeds)
- **Recipe (tasks/getup_env.py, tasks `PoOlympic-MattBio-Getup-Ladder` .. `-e`):** a pose ladder squat -> tuck (the
  squat's joints, rocked back onto the back) -> supine with 3 interpolated start poses per leg (7 stages, root placed on
  the ground; CPU check: no ground or self penetration at any stage), 6 s episodes, and an upward torso assist that
  follows the frontier's success rate (no assist for 15 its, then +0.03 BW / it at 0 % success, -0.02 at 100 %, cap 0.8;
  frontier episodes only; the frontier advances after 5 its at zero assist). A first ladder squat -> long sit -> supine
  was dropped before training: its blends hang the pelvis 0.4 m up on the heels and drop the athlete onto its back.
- **v2** (from getup_rev_v1 it 50): stages 0 and 1 passed unassisted by it 75 (assist rose to 0.30, fell back to 0);
  stage 2 (seated, leaning back) 0 % for 150 its with the assist at its 0.6 cap. Stopped at it 230.
  CPU rollout: within 0.2 s it throws the knees open (knee action -1.6, target 0 deg), uses the kick to sit bolt upright
  (tilt 1-5 deg, pelvis 0.16 m) and stays, at any assist.
- **v2b** (+ `feet_under_pelvis` std 0.3 w 3, assist cap 0.8): 0 %, stopped at it 100. **v2c** (std 0.5, w 4): 0 % for
  270 its, stopped at 300. **v2d** (upright kernel paid in proportion to height; `tuck_when_low`, a kernel on the squat
  leg angles): 0 %, tuck reward 0.036 = the first frames only, stopped at 150. Cause: zero action = the standing pose
  and holding 125 deg of knee flexion needs an action of +7.3, so every sampled action opens the knees, and a kernel
  reward is flat once they are open: no gradient.
- **v2e** (`legs_folded_when_low`: linear in hip / knee angle between the standing and squat poses, clamp -0.5..1, w 3;
  frontier starts at stage 2; init v2c it 50 with action std floor 0.5; 1500 its, 72 min, 2.2-3.6 s/it): folded reward
  rose from it 0, stage 2 success 1 % at it 140, 8 % at 300, 56 % at 420, 95 % with zero assist at it 489; stages 3-6
  in 15 its each; frontier = the event start from it 535 at 99 %. standing_tall 4.45 / 5 at the end.
- **Gate (tools/getup_probe.py, supine, CPU):** it 300 0/10 · it 600 / 900 / 1200 / 1499 **10/10**, up at the end
  10/10. 30 seeds: 30/30 at all four, 0 falls after getting up; mean time to up 0.98 / 0.76 / 0.72 / 0.71 s.
  Motion: knees up, sit up with one arm on the ground (8-14 ticks of arm contact), squat, stand. Fastest joint 19-21
  rad/s (ankle inversion), 0.2-0.3 % of substeps above 18 rad/s (bar 5 %); knees at the 280 Nm cap, hips 200-249 Nm.
- **Brain:** provisional get-up brain = `getup_rev_v2e_it600.onnx` (0.98 s, stands with the arms down; it 1499 is up in
  0.71 s, faster than a trained human, and ends with one arm held out). Not deployed. mattbio now has every brain MATT
  has (Rung 0, Rung 2, flight, crawl, stance, get-up). Open: prone starts were not trained (prone_fraction 0);
  Event 27 needs its scene and the hand-over get-up -> Rung 2 sprint; the promotion of mattbio to MATT is a user call.
- **Lost time:** Windows Modern Standby (5 min idle on AC) froze v2 for 53 min (17:16-18:09) and the v2b smoke run for
  13 min. `tools/keep_awake.ps1` (ES_SYSTEM_REQUIRED while a queue / train process exists) is now started with the queue.
- Housekeeping: getup_v4 and the stopped v2 / v2b / v2c / v2d runs moved to `runs_archive/2026-10-01/`; Unity closed
  17:10-21:12; viewer shown after the run (final brain). Review: `parity/tb/getup_rev_v2e/review.html`.

## 2026-10-01 23:10 · Night block (user: "8 hours to train, find things to train"); Unity closed
- **Chosen from tasks.md** (what is open and trainable without a user decision): GRANDMA crawl (events 8 / 23 run MATT in
  her lanes), GRANDMA Rung 2 precision (5/10), get-up from face down on mattbio (v2e: supine only), first get-up brains
  for GRANDMA and the zombie (Event 27). Not chosen: zombie sprint precision (z2_v12-v14 already tried the known levers),
  stance skills on the other bodies (contract v4 for them first), terrain / jumps (new props and commands = design).
- **New tasks:** `PoOlympic-Grandma-Crawl` / `-Crawl2` (`crawl_env.py`), `-Grandma-Rung2-Omni-Cap28-Sharp`
  (`grandma_env.py`: track_lin std 0.28 + 30 % sprint-band commands), `-MattBio-Getup-Prone`, `-Grandma-Getup`,
  `-Zombie-Getup` (`getup_env.py`: `prone_fraction` on the last ladder stage; `body_getup_ladder_env_cfg` = the v2e
  ladder on another body, "standing" scaled by its standing pelvis height K = 0.71 GRANDMA / 0.59 zombie, from the
  squat). `getup_probe.py --prone` + body scaling, gate `getup_prone` in `watch_gate.py`. All five pass a 2-iteration
  smoke run at 64 envs.
- **gcrawl_v1** (zombie's final crawl recipe on GRANDMA, from her Rung 0 brain, 1.5-1.9 s/it): **stopped at it ~370.**
  Reward flat at 100 from it 180: lies face down with her hands on the floor (crawl_up 0.65 / 1.5 = pelvis at 0.19 m,
  hands 0.87, progress 0.003 / 3); gate it 300: 0 m, all fours 0 %, hands 100 %. Her arms cannot press her up: shoulder
  34 Nm, elbow 30 Nm (60 % strength, 65 kg; a push-up needs ~50 Nm per arm). And the crawl height was wrong for her: FK
  puts her hands-and-knees pelvis at 0.32 m (hands and knees touch together at trunk pitch 74°), not MATT's 0.5 m × λ
  = 0.435 m, which is a bear crawl with her short thighs.
- **gcrawl_v2** (23:24): + `reset_crawl_mix` (60 % of episodes start in one of four poses between lying and
  hands-and-knees, placed on the ground; 40 % lying as before), crawl height 0.32 m. Same warm start (Rung 0).
- Queue after it: g2_v6 → getup_prone_v1 → ggetup_v1 → zgetup_v1. TensorBoard :6006 (running since 14:10; no obsolete
  runs in `training/runs`). `keep_awake.ps1` running.

## 2026-10-02 00:00 · Audit (user: "/ai PLAN and then TRAIN" on the MuJoCo stack) + gcrawl_v2 crawls
- **gcrawl_v2:** gate it 300 0 m (all fours 0 %) · it 600 0.1 m but **all fours 100 % from the face-down start** · it 900
  **22.8 m in 60 s**, all fours 100 %, hands 78 %, lane 0.44 m, 0 tumbles. The pose-continuum starts and the 0.32 m crawl
  height fixed what v1 could not do; the forward-progress reward took off at it ~750 (0.27 → 1.26 of 3 by it 930).
- **Observations / actions (all bodies, contract v3):** 84 obs (heading-frame velocity 3, angular velocity 3, gravity 3,
  pelvis height 1, command 3, gait phase 2, joint positions 23, joint velocities 23, last action 23), 23 joint position
  targets (default + 0.25·a, clipped to the joint range), 50 Hz control on a 200 Hz simulation. No action masking
  (continuous control); the critic sees the actor's observations without noise. All PPO; no SAC / self-play in the stack.
- **Mass distribution (every body):** trunk + head 50.4 %, arms 9.9 %, thighs 28.3 %, shanks 8.6 %, feet 2.7 % = de Leva.
- **Joint ranges:** the same human ranges on every body (knee 0-150°, elbow 0-145°, hip flexion −30-120°, ankle −50-25°,
  trunk twist ±45°).
- **PD drives** (damping ratio on the joint's own inertia at the default pose): trunk flex / lateral 0.33-0.41 at 1.2 Hz
  (soft, under-damped), hips 0.5-0.63, shoulders 0.6, knees 1.0-1.3, ankles 3.5-5 unloaded (the foot alone; loaded by the
  body they are not over-damped). No passive joint damping. Unchanged: every brain and the Unity parity gates depend on it.
- **Torque caps:** mattbio has direction-specific human caps (knee 280 / 150, hip 280 / 200, ankle 220 plantar / 60 dorsi,
  hip rotation 80, ankle inversion 45 / 60 Nm). **The zombie and GRANDMA still have one symmetric cap per joint group:**
  GRANDMA's hip rotation (119 Nm), ankle dorsiflexion (93) and ankle inversion (93) are stronger than the strong
  adult male's, at "60 % strength". Her arms: shoulder 34, elbow 30 Nm. Fixing the caps changes her body (every GRANDMA
  brain retrains, Unity fingerprints change): user decision, not done.
- **Hardware:** 4096 envs: 1.8-1.9 s/it (≈ 52k steps/s), GPU 55-60 % busy, 4.4 of 12 GB, 88 °C at 52-68 W = thermally
  limited (fans on Auto; Cooler Boost measured +30-40 %). 8192 envs + 8 mini-batches (recipe v5, validated on mattbio
  only) would add ~35 % samples/s; not applied to the queued GRANDMA / zombie runs (untested on their contact buffers).
  Control rate and solver settings are contract-fixed (Unity parity).
- **Manifest.** P0 (queued, PPO fine-tunes): GRANDMA crawl (30 m from face down, 5/5), GRANDMA Rung 2 precision (G1
  10/10, now 5/10), mattbio get-up from face down (10/10 both ways). P1: first get-up brains for GRANDMA and the zombie
  (ladder from the squat, ~65 min each). P2 (not tonight): zombie sprint precision (three failed attempts, needs a new
  idea), Event 4 reach (0/10 in every stance run: diagnose first), stance skills on the other bodies (contract v4 for
  them), terrain / jump rungs (props + commands to design), GRANDMA / zombie direction-specific torque caps (user).
- **gcrawl_v2 finished 00:03 (1200 its, 39 min): gate it 1199 = 30 m in 29.3-29.9 s, 5/5**, all fours 100 %, hands 55 %,
  0 tumbles, lane 0.16-0.20 m (MATT 25.7-26.7 s, zombie 31.7 s). Provisional brain `gcrawl_v2_it1199.onnx`, not deployed
  (the all-fours scenes hold MATT + zombie only). Progress 1.74 / 3 and mean reward +6 per 100 its at the end →
  **gcrawl_v3** queued last (`-Crawl3`: no assist, full speed range from the start, from it 1199, 800 its). Review:
  `parity/tb/gcrawl_v2/review.html`.
- **g2_v6** started 00:03 (1.6-1.9 s/it).

## 2026-10-02 00:26 · g2_v6 stopped at it ~700 → g2_v7 (bands + symmetric runner)
- **g2_v6** (cap28 + track_lin std 0.28 + 30 % forward sprint band, from g2_v5 it 1199): it 300 **5/10**, it 600 4/10, 0
  falls. Forward sprint segments over the bar 7 → 4 → 5 of 19; crab 3 → 3 → 2 of 12; at it 600 the backward drill failed
  on 2 seeds for the first time (the sprint band took mass from backward commands).
- **What the misses are** (same segments at g2_v5 it 1199 and g2_v6 it 300 / 600): backwards at 1.08-1.23 m/s (RMS
  0.22-0.38 vs 0.187), side-steps to her right at 0.77-0.88 m/s (0.24-0.36; none to her left), forward 2.0 / 2.6 m/s
  with a right turn of 0.35-0.38 rad/s. One-sided: the cap28 line was trained with the plain runner.
- **g2_v7** (00:26, `-Cap28-Bands`, from g2_v6 it 300): SymmetricRunner + command bands (`AthleteCommandCfg.bands`):
  forward 2.0-3.0 × √λ 20 %, backward 0.8-1.5 × √λ 15 %, side-step |vy| 0.4-1.1 × √λ 20 %; std 0.28 kernel kept. 1200
  its, gates every 300.

## 2026-10-02 01:30 · g2_v7 finished: G1 6/10 at every gate; better in the events → candidate for `grandma_rung2.onnx`
- **g2_v7** (bands + SymmetricRunner, from g2_v6 it 300, 1200 its, 45 min at 2.1-3.1 s/it): G1 rung 2 **6/10** at it 300 /
  600 / 900 / 1199, 0 falls (g2_v5 it 1199: 5/10). Side-steps: 3 of 12 over the bar → 0 from it 900; the left / right
  difference is gone (symmetry loss 0.36 → 0.055). Left over at it 1199 (4 seeds): backward 1.19-1.23 m/s commands
  (0.19-0.23; she walks 1.05 m/s), 2.57 m/s (0.22; she runs 2.49) and two accelerations of ~1.9 m/s that must settle
  within 1.75 s (0.28-0.33; steady state at the same command 0.10).
- **CPU probe, one command for 10 s** (`g2_v7_it1199` vs the deployed g2_v5 it 600): side-step 0.88 m/s RMS 0.11 vs 0.32
  · backward 1.23 command 1.05 vs 0.87 m/s · 2.03 m/s RMS 0.10 vs 0.18 (ripple sd 0.08 vs 0.13) · 2.8 command 2.65 vs
  2.71 m/s · time to 90 % of 2.03 m/s 2.4 s vs 1.8 s. Steadier, better backwards and sideways, slower off the mark.
- **Event heats** (`tools/compare_brain_events.py`, new: 6 CPU heats per event on the MATT / zombie / GRANDMA lineup, her
  3 lanes; deployed → g2_v7 it 1199): Gust Gauntlet mean place 5.6 → 4.6 (still in at the end 1 → 5 of 18) · Inverted
  Sprint 21.1 → 17.9 s (place 7 → 5) · Crab 28.9 → 22.2 s (7 → 5) · Slalom 30.6 → 24.3 (6.1 → 2.9; falls 4 → 1) ·
  Turntable 6.92 → 7.41 (4.7 → 5.6) · Steeplechase 17.5 → 18.3 s · Terminal Velocity peak 2.77 → 2.73 m/s · Emergency
  Brake DQ 3 → 2. Better in five events, slightly worse in three. it 300 sits between the two.
- **Not deployed** (Unity closed; a swap needs the critic export, confidence fit, odds refit and new parity references):
  `g2_v7_it1199.onnx` is the recommended `grandma_rung2.onnx`. The remaining G1 misses are acceleration (the 1.5 m/s²
  settle rule is not body-scaled) and backward top speed: bars for a 60 %-strength body = user decision.
- **getup_prone_v1** started 01:11.

## 2026-10-02 01:30 · getup_prone_v1: get-up from face down solved at it 250 (stopped there)
- **getup_prone_v1** (mattbio, v2e recipe with the last ladder stage 50 % face down, from v2e it 1499, 8192 envs,
  2.1-2.3 s/it): frontier success 0.50 at it 25 (all supine), assist rose to 0.29 by it 100, success 0.99 with the
  assist back at 0 by it 132. **Gate it 250: supine 10/10 (0.68-0.72 s), face down 10/10 (1.14-1.18 s); 30 seeds each:
  30/30 and 30/30, 0 falls after.** Stopped at it ~270 (the remaining 730 its would only make it faster; v2e it 1499
  is already quicker than a person). Provisional brain `getup_prone_v1_it250.onnx`; joint speeds not re-measured.
- **ggetup_v1** started 01:30 (GRANDMA, ladder from the squat, 4096 envs, 2000 its).

## 2026-10-02 01:55 · ggetup_v1: GRANDMA gets up from her back (stopped at it ~1050); reach diagnosis → rs_reach_v1 queued
- **ggetup_v1** (the v2e ladder on her body from the squat, from her Rung 0 brain, 4096 envs, 1.2-1.5 s/it): squat
  passed at it ~140, stage 1 at ~210, stage 2 (seated, leaning back) from it 212 to ~740 with the assist at its 0.8 cap
  (success 0.09 → 0.31 → 0.89), stages 3-6 by it 800; frontier success 0.999 without assist from it 900. Action std
  0.50 → 0.89 (it 700) → 0.36 (it 1000). **Gate it 500 0/10 · it 1000 10/10; 30 seeds 30/30, up in 0.84-0.88 s, 0 falls
  after.** Face down 0/10 (untrained) → `ggetup_prone_v1` queued (`-Grandma-Getup-Prone`, from it 1000). Provisional
  brain `ggetup_v1_it1000.onnx`. 0.86 s is fast for the "frail" body; joint speeds not measured.
- **Event 4 reach, diagnosed on CPU** (rs_v8 it 1199, 24 single targets, `K.sample_reach_target`): the hand ends a median
  0.095 m from the target at 2 s (same at 4 s), 8 of 24 inside the 0.08 m bar — the drill needs 6 in a row, hence 0/10
  in every run. Error vs target distance r = 0.65: targets at 0.67-0.71 m from the shoulder end 0.10-0.16 m short,
  elevations above 40° end 0.09-0.17 m low. FK: the straight arm reaches 0.611 m (commanded up to 0.712 m); 92 % of
  the targets are within 0.08 m of the arm-only workspace, the rest need a trunk lean, which the uprightness reward
  charged for (12° = 0.28 per step).
- **rs_reach_v1** queued last (`-MattBio-RungS-Reach`: rs_v7's env, 50 % reach commands, extra std 0.06 kernel,
  uprightness off while reaching, entropy 0.001, from rs_v8 it 1199, 1200 its, gate S).
- **zgetup_v1** started 01:54.

## 2026-10-02 02:20 · zgetup_v1 stopped at it ~1040 (hangs tucked from the assist) → zgetup_v2
- **zgetup_v1** (the ladder on the zombie from the squat, from its Rung 0 brain, 1.4-1.5 s/it): stages 0 and 1 passed by
  it 200; stage 2 (seated, leaning back) 0.0000 success from it 212 to 1040 with the assist at its 0.8 cap; gates it 500
  and 1000: 0/10. GRANDMA had passed the same stage by it 740 (0.09-0.31 success on the way).
- **CPU rollouts of the it 1000 brain from stage 2** (`ladder_probe.py`, scratch): no assist: tips over backwards in 1 s
  and lies with folded legs. Assist 0.6-0.8 body weights: the 22 kg body hangs from the torso force — pelvis 0.26-0.31 m
  (standing 0.56), feet 0.12-0.16 m off the ground, trunk tilted 51-60°, knees held at 107-112° by an action of +5.
  `legs_folded_when_low` pays (1 − z / standing height) × fold: half its weight at that height, and opening the knees
  loses it before the feet reach the ground. A local optimum made by the assist and the reward together; the heavier
  bodies did not levitate.
- **zgetup_v2** (queued after gcrawl_v3; `-Zombie-Getup2`): the folded-legs reward fades to 0 at half the standing
  pelvis height (0.28 m), from zgetup_v1 it 200 with stage 2 as the frontier, 1500 its.
- **gcrawl_v3** started 02:19.

## 2026-10-02 02:45 · gcrawl_v3 finished: 27.1-28.6 s, 5/5 at every gate
- **gcrawl_v3** (continuation of gcrawl_v2 it 1199 without assist, full speed range; 800 its, 23 min): gates it 200 /
  400 / 600 / 799: 27.8-28.6 s (lane 0.01 m) · 27.3-28.0 (0.08) · **27.1-27.9 (0.10)** · 27.5-28.3 (0.07), 5/5 each
  (v2 it 1199: 29.3-29.9 s, lane 0.18 m). Levelled off: provisional GRANDMA crawl brain = `gcrawl_v3_it600.onnx`
  (MATT 25.7-26.7 s, zombie 31.7 s). Not deployed.
- Review of g2_v7 (45 min run): `parity/tb/g2_v7/review.html`. **g2_v8** queued last (same recipe from g2_v7 it 1199).
- **zgetup_v2** started 02:42.

## 2026-10-02 03:00 · zgetup_v2 stopped at it ~545: the zombie does not get up from sitting; rs_reach_v1 started
- **zgetup_v2** (folded-legs reward fading to 0 at half height, from zgetup_v1 it 200): stage 2 success 0.0000 for 540
  its at the 0.8 assist cap; gate it 500 0/10. It no longer hangs tucked: the it 500 brain, without assist, ends propped
  on both forearms, one thigh and one foot, twisted sideways (abdomen twist 42°, hip rotation 41°), pelvis 0.17 m, head
  0.36 m, and stays there. With 0.4-0.8 body weights of assist it reaches pelvis 0.23-0.32 m, trunk tilted 53-62°, feet
  0.10-0.17 m up, static. It is trying MATT's route (sit up on an arm), which needs an arm push: its shoulder / elbow
  caps are 10 / 8 Nm for a 13 kg trunk, and its abdomen 24 Nm against ~25 Nm of trunk weight about the hips.
- **Zombie get-up: open.** Not a tuning problem of this ladder. Idea not run: a front ladder (squat → hands down → all
  fours → face down), which uses the legs it already crawls with, then a roll from the back as a later step.
- Checked and dropped: a mattbio crawl continuation. crawl_bio_v1 solo is already 25.4-26.5 s at it 450 (MATT 25.7-26.7);
  the 33.1 s is the 8-athlete heat with full self-collision, which a solo fine-tune and the solo gate do not address.
- **rs_reach_v1** started 02:57.

## 2026-10-02 04:05 · rs_reach_v1 finished: single reaches inside the bar, the 6-target drill still 1/10 → rs_reach_v2
- **rs_reach_v1** (rs_v7's env, 50 % reach commands, std 0.06 kernel, uprightness off while reaching, entropy 0.001, from
  rs_v8 it 1199; 1200 its, 67 min at 3.3 s/it with the symmetric runner). 24 single targets (CPU, `reach_probe.py`):
  median error 0.095 m (start) → 0.047 (it 300) → 0.041 (600) → 0.031 (900); inside the 0.08 m bar 8 → 16 → 21 → 22
  of 24. **Drill E4 (6 chained targets, worst one counts): 0 / 1 / 1 / 1 of 10** at it 300 / 600 / 900 / 1199; worst
  error per seed 0.066-0.29 m at the end (0.08-0.32 at it 300). Other drills on this brain: squat 10, torso 7, march
  3-7, flamingo 0.
- **What the drill misses are** (`reach_chain.py`, scratch): not the chaining (the shoulder is within 0.05 m of neutral
  when each target is drawn; most targets in a chain end 0.02-0.05 m off) but the targets past the straight arm
  (0.60-0.72 m from the shoulder): the hand ends 0.09-0.22 m low and 0.09-0.18 m short towards the arm's own side with
  the trunk tilted 1-5°. 30 % of targets are that far, so 88 % of 6-target chains contain one.
- **rs_reach_v2** queued (`-RungS-Reach2`, `AthleteSkillCommandCfg.reach_far_fraction` 0.5: half of the reach commands
  from 80-95 % of the reach radius), from rs_reach_v1 it 1199, 1200 its.
- Open (user, contract): `hand_reach` 0.75 m × 0.95 = 0.71 m is 0.10 m past the straight arm's 0.61 m, as the torso yaw
  range is past the spine's.
- **ggetup_prone_v1** started 04:04.

## 2026-10-02 05:00 · GRANDMA face down: prone-fraction recipe fails, roll ladder works (6/10 so far)
- **ggetup_prone_v1** (MATT's recipe: last stage 50 % face down, from ggetup_v1 it 1000, 500 its): frontier success
  0.51 → 0.64 = the supine half, assist 0.69-0.78 throughout; gates it 200 / 400 / 499: supine 10/10, **face down 0/10**.
  Her arms cannot press her up (the gcrawl_v1 finding again).
- **ggetup_roll_v1** (`-Grandma-Getup-Roll`, new: `roll_stages` 60° / 120° / 180° after the supine stage =
  `_reset_rolled`, lying rolled about the long axis either way; from ggetup_v1 it 1000, 800 its): the 60° stage passed
  in 20 its without assist; the 120° stage 0.15 → 0.48 by it 620 with the assist at 0.8, then flat while the action std
  rose 0.44 → 0.93 (entropy 0.005). Gates: supine 10/10, face down 0/10 at it 200-799.
- **ggetup_roll_v2** (`-Getup-Roll2`: frontier starts at the 120° stage, entropy 0.001, start std capped at 0.5; from
  roll_v1 it 799, 1000 its, 24 min): std 0.50 → 0.40; 120° stage success ~0.62 with the assist falling 0.49 → 0.28
  (the controller holds success at 60 %). No gate ran during the run (gate iterations 250 / 750 do not exist with a
  save interval of 100: the watcher waited for them; it was stopped). By hand, it 999: **supine 10/10 (0.78 s), face
  down 6/10 (0.62-0.70 s), 0 falls after.**
- **ggetup_roll_v3** queued after rs_reach_v2 (same task from roll_v2 it 999, 800 its); g2_v8 cut to 800 its to fit
  the block. **rs_reach_v2** started 05:00.

## 2026-10-02 05:55 · GRANDMA gets up from face down (roll ladder); rs_reach_v2 stopped
- **rs_reach_v2** (half of the reaches from 80-95 % of the radius, from rs_reach_v1 it 1199): stopped at it ~320. Gate it
  300: drill 0/10, worst error per seed 0.085-0.27 m (v1 it 1199: 0.066-0.29). ~0.01-0.02 m per 300 its on the far
  targets: not reachable tonight. Event 4 reach stays open; the far targets need a lean the brain does not learn, or a
  shorter `hand_reach` (user).
- **ggetup_roll_v3** (same task from roll_v2 it 999, 800 its): 120° stage passed without assist at it ~390; face-down
  stage success ~0.62 with the assist falling 0.62 → 0.19 by it 799. Gates face down 6 / 6 / 6 / 7 of 10, supine 10/10.
- **ggetup_roll_v4** (from roll_v3 it 799, 600 its; re-passed the 120° stage in 60 its): assist 0 from it 200, face-down
  stage success 0.67 → 0.92 by it 440. **Gates it 200 and 400: supine 10/10, face down 10/10. 30 seeds on it 400: supine
  30/30 (0.74-0.80 s), face down 30/30 (0.50-1.16 s, mean 0.69), 0 falls after.**
- **Joint speeds during the get-up** (`getup_speed.py`, scratch; share of physics steps with a joint above the body's
  bar, G1 allows 5 %): MATT `getup_prone_v1_it250` 0.6 % supine / 0.3 % face down (peak 22 rad/s) · GRANDMA
  `ggetup_v1_it1000` 0.6 % supine · GRANDMA `ggetup_roll_v4_it400` 0.3 % supine, **5.2 % face down (peak 34 rad/s, a
  shoulder during the roll)** — at the allowance; half a second from face down to standing is not "frail".
- Provisional GRANDMA get-up brain (both ways): `ggetup_roll_v4_it400.onnx` (it 599 to be checked). Not deployed.
- **g2_v8** started 05:54 (the last run of the block).

## 2026-10-02 06:45 · End of the night block: g2_v8 8/10; summary
- **g2_v8** (continuation of g2_v7 it 1199, same recipe, 1200 its, 48 min): G1 rung 2 **7 / 5 / 7 / 8 of 10** at it 300 /
  600 / 900 / 1199, 0 falls; segments over the bar 4 / 6 / 4 / **2** of 50 (g2_v5 it 1199: 10). Left at it 1199:
  backward 1.19 m/s (0.203) and the speed-up to 2.03 m/s (0.253). track_lin 0.90 → 0.99, mean reward 116 → 119
  (g2_v7: 103 → 116): diminishing returns.
- **Event heats** (6 per event, her 3 lanes; deployed g2_v5 it 600 → g2_v8 it 900 / it 1199): Gust Gauntlet still in at
  the end 1 → 1 / **12** of 18 · Inverted Sprint 21.1 → 17.5 / 17.5 s · Crab 28.9 → 21.5 / 21.2 s · Slalom 30.6 (4 falls)
  → 24.5 (0 falls) / 25.6 (4 falls) · Turntable 6.92 → 7.18 / 7.08 · Steeplechase 17.5 → 17.6 / 17.6 s · Terminal
  Velocity 2.77 → 2.75 / 2.75 m/s · Brake DQ 3 → 3 / 3. **Recommended `grandma_rung2.onnx`: `g2_v8_it1199.onnx`**
  (it 900 if Slalom falls weigh more than Gauntlet survival). Not deployed. Review `parity/tb/g2_v8/review.html`.
- `ggetup_roll_v4_it599.onnx`, 30 seeds: supine 30/30 (0.74-0.78 s), face down 30/30 (0.48-0.88 s); joint speeds 0.0 %
  / 5.3 % over the bar.
- **Block summary (23:10 → 06:42, 7.5 h of GPU, 17 runs started, 9 run to the end, 8 stopped by hand at a gate):**
  new provisional brains — GRANDMA crawl `gcrawl_v3_it600`, GRANDMA walking `g2_v8_it1199` (G1 5 → 8/10), GRANDMA
  get-up both ways `ggetup_roll_v4_it599`, mattbio get-up both ways `getup_prone_v1_it250`, mattbio reach
  `rs_reach_v1_it1199` (single targets yes, drill no). Open: zombie get-up (2 runs, 0 %), Event 4 drill (far targets),
  GRANDMA's last two G1 misses (strength / bars). Nothing deployed to Unity; no body or contract changed.

## 2026-10-02 10:15 · Night-block brains deployed; play-through
- **GRANDMA walking** `grandma_rung2.onnx` = g2_v8 it 1199 (old brain kept as `g2_v5_it600.onnx`): critic + confidence
  (AUC 0.68), parity references re-recorded, Unity G2 2.2e-16 / G3 8.3e-7 / G4 7.9e-8, odds refitted (40 fresh heats
  per event with her, Events 5, 9, 10, 11, 12, 13, 19, 22).
- **GRANDMA crawl** `crawl_grandma.onnx` = gcrawl_v3 it 600 (+ critic, confidence AUC 0.70): `compose_mixed` crawl8 /
  trench8 on the full roster (verify 37/37, crowd-contact audit 24/24 scenes), Events 8 and 23 rebuilt + baked, menu
  rebuilt (she is no longer replaced by MATT there), odds on 40 MATT / zombie / GRANDMA heats each. Trench odds stay
  near chance (winner log-likelihood −2.31 vs −2.08 uniform).
- **Not deployed (no event uses them yet):** GRANDMA get-up `ggetup_roll_v4_it599`, mattbio get-up
  `getup_prone_v1_it250`, mattbio reach `rs_reach_v1_it1199`.
- Unity EditMode `PoOlympic.Tests` 81/81 after the rebuild.
- **Play-through in Unity** (10 single events from the menu, lineup M Z G M Z G M Z, each: heat → podium → match card →
  OK → menu, 10/10): 30 m All Fours G3 2nd in 29.98 s (MATT 26.04), G6 6th 33.02; Trench Crawl G6 5th 31.4 s, G3 7th
  45.1 s (zombies 17.0-17.1, MATT 16.7); Crab 21.1 / 21.2 s (4th / 5th, ahead of the zombies' 25.5-26.2; old brain
  28.9 s); Inverted Sprint 17.1 / 17.7 s (4th / 5th; old 21.1); Slalom G3 **wins** 23.40 (G6 4th); Gust Gauntlet both
  still in at the end (4th / 5th; second run one fell at 38.7 s); Turntable 3rd / 6th; Steeplechase 17.2 / 18.3 s;
  Terminal Velocity 2.76 / 2.74 m/s (last, by design: 2.8 m/s cap); Emergency Brake G6 **wins** (3 cm short), G3 DQ
  (crossed the line).
- Seen, not new: MATT lanes 4 / 7 DNF in the 30 m crawl with tumbles (2 of 3 MATTs; the heat then runs to the 60 s
  limit) — crawl DNFs were there before GRANDMA joined (2026-10-01 tour: Z4 4 tumbles, L5 58 s).

## 2026-10-11 07:00 · KIM, fourth athlete: body, Rung 0 / 1 / 2, crawl, get-up — all gates passed in one night
- **Start state:** the repo was a fresh clone without Git LFS content (991 pointer files: no brains, visuals, `mujoco.dll`)
  and without `training/.venv` / `training/runs`; `git lfs pull` + `uv sync`. Old checkpoints and TensorBoard logs are
  gone: comparisons with earlier runs use the saved charts in `parity/tb/`.
- **Body** (user: "give her unique characteristics"; picked: "light and quick"): 1.55 m, 54 kg, strength 0.9, stiffness
  1.0, full self-collision, MATT's upright stance. `SourceArt/Kim/convert_kim.py` (headless Blender through
  `blender-launcher.exe`). Her hip joints sit at 0.44 of her height (short legs), while every bar is scaled by height.
- **Recipes:** GRANDMA's, with `grandma_env.G` bound to the process body (`tasks/kim_env.py`); no stoop / careful steps,
  swing-time reward capped at 0.45 s × √λ. Queue (`train_queue.ps1`), 4096 envs, 1.5-2.1 s/it, Unity closed:
  | run | task | from | its | min | gates |
  |---|---|---|---|---|---|
  | k0_v1 | Rung0-Stand | scratch | 1500 | 53 | 9 · 10 · 10 of 10 |
  | k1_v1 | Rung1-Run | k0_v1 it 1499 (warm_start) | 2000 | 71 | 7 · 10 · 10 · 10 of 10 |
  | k2_v1 | Rung2-Omni-Base | k1_v1 it 1999 (warm_start) | 2000 | 67 | 0 · 2 · 3 · 7 of 10 |
  | k2_v2 | Rung2-Omni-Bands | k2_v1 it 1999 (plain_init, std ≤ 0.3) | 1200 | 49 | 5 · 8 · 8 · 9 of 10 |
  | k2_v3 | Rung2-Omni-Bands2 | k2_v2 it 1199 | 1000 | 41 | 10 of 10 at 200 / 400 / 600 / 800 / 999 |
  | kcrawl_v1 | Crawl2 | k0_v1 it 1499 (warm_start) | 1500 | 50 | 0/5 until 1200, 5/5 at 1499 (32.4-33.3 s, lane 5 m) |
  | kcrawl_v2 | Crawl3 | kcrawl_v1 it 1499 | 1000 | 33 | 5/5 at every gate, 28.1-29.1 s, lane 0.15 m at 999 |
  | kgetup_v1 | Getup | k0_v1 it 1499 (plain_init) | ~1030 (stopped) | 26 | supine 0 · 10 of 10 |
  | kgetup_prone_v1 | Getup-Prone | kgetup_v1 it 1000 | 600 | 15 | supine 10/10; face down 0 · 0 · 10 of 10 |
- **Rung 2 diagnosis** (`tools/gate_fails.py`): k2_v1's misses were one-sided — one turntable direction 0.3-0.5 s slower,
  only right-hand side-steps over the bar — as on GRANDMA; the mirrored trainer evened the two directions within 0.2 s.
  What was left at 9/10 were fast side-steps (|vy| 0.76-0.87 m/s); a band of 30 % fast side-step commands (k2_v3) passed
  at its first gate. Not tested: whether k2_v2 alone would have got there with more iterations.
- **Checks beyond the gates:** Rung 0 30 seeds 27/30; Rung 2 30 seeds 28 / 29 / 28 at k2_v3 it 200 / 600 / 999 (0 falls;
  miss = 3.45 m/s sprint with a 0.42 rad/s turn); event heats pick it 600 (no fall, no DQ; it 999 falls in the Gauntlet
  and the Slalom); get-up 30/30 both ways (0.70-0.82 s supine, 1.00-1.06 s face down), joint speeds inside the bar.
- **Deployed:** `kim_rung0` = k0_v1 it 1499 · `kim_rung2` = k2_v3 it 600 · `crawl_kim` = kcrawl_v2 it 999 · `getup_kim`
  = kgetup_prone_v1 it 599 (no event uses it). Critics + confidence (AUC 0.67 / 0.67 / 0.70), odds with a sixth (KIM)
  feature, roster scenes with four bodies per lane (verify 46/46, crowd contacts 32/32), Unity G0 / G2-G4 PASS, 11 event
  scenes rebuilt + baked, menu with her card. Reviews: `parity/tb/k0_v1`, `k1_v1`, `k2_v3`, `kcrawl_v2`; get-up films
  `parity/tb/kgetup_prone_v1`; menu `parity/ui/kim_menu_before_after.html`.
- **Mistakes made on the way:** two scripted multi-line replacements dropped their second half (PowerShell argument
  parsing without parentheses) — a deleted `pedestal8` scene entry and an unterminated C# initialiser, both caught by the
  next run; running three 30-seed CPU evals during kcrawl_v1 slowed it to a crawl for ~15 min.
- The MuJoCo viewer was not opened during these runs (night, user away); the get-up films are the only renders.
## 2026-10-11 14:16 · Training audit: action logs, opt-in reward / curriculum / biomechanics terms, registry hygiene, brain manifest

No training run was started and no brain, MJCF or `contract*.json` changed. Proof that the registered tasks are what
they were: `tools/dump_task_cfgs.py` hashes every task's train + play env cfg, PPO cfg and runner per body (entity
MJCFs by content) — **91 task IDs before, the same 91 hashes after**, 9 new IDs, assets identical. 17 older IDs
(get-up 3-5, crawls, Pedestal2, MattBio-Getup-Rev) show a *code* change only: `getup_assist`, `push_gust` and
`gust_curriculum` gained optional arguments whose defaults are the old behaviour. Tests 76/76 before, 87/87 after
(`tests/test_audit.py`, 11 new).

- **Action logging (AGENTS "Data logging"; nothing logged actions before).** `poolympic/action_log.py`: with
  `$POOLYMPIC_LOG_ACTIONS=<run>` or `--log-actions` (`eval_cpu.py`, `watch_gate.py`) every G1 episode (Rung 0 / 1 / 2 /
  S) and every event heat (all ten event modules: one decorator each, ticks come from `_Lane.control`) writes
  `parity/actions/<run>/<gate>_seed<N>.jsonl.gz`: header, one line per tick (command, pelvis position, tilt, velocity,
  joint speeds, action, ctrl, actuator force; per lane in heats), one summary line (outcome, pass, fall time, the
  episode's own record). ~1 MB per Rung 2 seed. `tools/report_actions.py` → markdown / CSV per run: success rate,
  fall-time histogram, mean |torque| per joint group, action rate, ticks and joints over the speed bar. Not covered:
  the get-up / crawl / flight probes (own loops), GPU training rollouts.
  First numbers (KIM `kim_rung2`, Rung 2 drill, 3 seeds): mean torque 4-10 % of the caps per group, action rate 0.148
  per joint and tick, 0 ticks over the joint-speed bar.
- **Reward gaps, as opt-in terms.** `mdp.self_contact_force` + `matt_env.self_contact_sensor_cfg` (AGENTS: charge the
  force if a policy leans on self-contact) — unit-tested and built once on KIM (random flailing: up to 2.3 body
  weights), in no task. `getup_assist`'s 80 kg default: **no body was trained with a wrong weight** — only
  `PoOlympic-Matt-Getup3/4/5` rely on the default and MATT weighs 80 kg; `MattBio-Getup-Rev` passes
  `C.BODY.total_mass`; every ladder task (mattbio, zombie, GRANDMA, KIM) uses `ladder_assist` with the body's mass
  (zombie 22, GRANDMA 65, KIM 54 kg). New: `body_weight_n=None` reads each env's model mass (`mdp.robot_mass`, DR
  included). Rung 0 never had DESIGN §5's "CoM over support, no foot slip": `mdp.com_over_support` + Rung 1's
  `foot_slip` in new IDs `PoOlympic-{Matt,Zombie,Grandma,Kim}-Rung0-Full` (the old Rung 0 IDs are untouched).
- **Curriculum / DR, opt-in.** `gust_curriculum` generalised (`steps_per_it`, `min_count`: ended episodes pooled, one
  decision per PPO iteration — the ped_v2 / getup_rev lesson) and `push_gust` (`min_fraction`, `record`) →
  `matt_env.add_shove_ramp`: shove size U(0, level), level from the old box's corner (√2 × 0.5 m/s × √λ) up to twice
  that, +3 % per iteration while > 80 % of ended episodes reach the time-out, down below 50 % (DESIGN §5). Restitution
  DR via solref (DESIGN §4, missing): `mdp.dr_geom_solref`, foot / toe time constant × U(0.8, 1.25), damping ratio ×
  U(0.7, 1.0). Both in `PoOlympic-{Matt,Zombie,Grandma,Kim}-Rung2-Ramp` = each body's production Rung 2 recipe (Sym5 /
  Wobble / Cap28-Bands / Bands2) + ramp + solref + privileged critic.
- **Asymmetric critic + export.** `matt_env.add_privileged_critic`: the critic also sees ground reaction per foot,
  friction scale, mass scale and the shove just applied (8 values; mirror map in `symmetry.py`, sizes 92 / 98). Warm
  start: `tools/expand_obs.py --critic-zeros 8` (zero columns: the same function at iteration 0). Consequence:
  `export_critic.py` now refuses a critic whose input differs from the actor's — a privileged critic cannot be Unity's
  confidence critic. `export_brain.py --clip-obs S` clamps the normalised observation to ±S σ inside the graph
  (default off = every deployed brain); new sidecars store the checkpoint path relative to the repo root.
- **Biomechanics in the physics, opt-in (contract v5 proposal, `docs/CONTRACT_V5_FATIGUE_PROPOSAL.md`).**
  `poolympic/biomech.py`: the torque-speed curve the reward term assumes now also limits the actuator force range
  (only on the side pushing with the motion), and a fatigue level per joint group (leaky integral of (effort / 0.25)²,
  10 s to tire, 30 s to recover, cap never below 60 %). Six fatigue levels appended after the v3 observation (84 → 90).
  Training: `mdp.bio_force_limits` (step + reset event), `obs_fatigue`; CPU: `biomech.BioLimits` in `evaluate.Sim` and
  the event lanes for 90-input brains (`POOLYMPIC_BIO=1` forces it for any brain). Task `PoOlympic-Kim-Rung2-Bio`
  (k2_v3's recipe + limits + fatigue block + privileged critic; warm start `expand_obs.py --actor-zeros 6
  --critic-zeros 14`).
  **Measured first (CPU, no training):** `kim_rung2` under both limits passes G1 Rung 2 **10/10**. With effort = torque
  / cap alone no group passed level 0.04 in 55 s (she uses 3-10 % of her caps on average, 0.87-0.92 at the ankle's 99th
  percentile), hence the 0.25 reference; with it the drill reaches ankle 0.60, knee 0.39, abdomen 0.36. The constants
  are proposals; as they stand the limits are mild.
- **Registry.** `-RungS-Stance-v8` registering rs_v7's env function is **not a typo** (log 2026-10-01: "rs_v8 = rs_v7's
  env, entropy 0.001") — left alone. Duplicate `matt_stance_env_cfg` import removed. `SUPERSEDED` set in
  `tasks/__init__.py` (42 IDs; nothing unregistered; `dump_task_cfgs.py --list`). Deleted, unreferenced:
  `tools/train_queue.sh`, `watch_eval.py`, `rank_zombie_r2.py`. `bench_env.py --ckpt` is now required (its default
  pointed at r2_v8, which is not on disk: `training/runs` holds KIM's runs only).
- **Dedupe (hash diff empty).** `matt_env.body_entity_cfg / scale_to_body / scaled_envelope / check_body / scene_root_z
  / body_fall_z` replace the copies in `zombie_env` and `grandma_env`; `MATT_ROOT_Z`, `FALL_PELVIS_Z`, `FALL_TILT_DEG`,
  `JOINT_VEL_LIMIT` live in `bodies.py` (were literals in `evaluate.py`, `matt_env.py`, `iron_pedestal.py`);
  `skill_mdp` imports the reach-sampling constants from `skills.py`.
- **Brain manifest.** `tools/brain_manifest.py` → `parity/brains/manifest.json`: 22 deployed brains (no `_itNNN` suffix,
  or a confidence critic next to them) with body (from the fingerprint), role, events, run, iteration, sha256, gate
  result; `--check` fails on a missing / changed / unlisted file. `fit_odds.py` reads its brains from it (same files as
  its old dict, asserted). Seen while generating: the three stance brains (`rs_v6_it2550`, `rs_v6_it3299`,
  `rs_v8_it1199`) carry **mattbio's** fingerprint; `rung1.onnx` and `r0_v2_latest.onnx` are contract v2 with an old
  fingerprint; no gate record found for `rung2`, `zombie_rung0`, `zombie_rung2`, `crawl_matt`, `crawl_zombie`,
  `r2f_v3_it100`.
- **Only smoke-tested (16 envs, 80-120 random-action steps, `tools/smoke_task.py`):** the nine new tasks build, step,
  stay finite and mirror cleanly on the GPU. Not exercised: a PPO iteration on any of them, the warm starts beyond the
  checkpoint surgery, whether the training-side limits and `BioLimits` agree tick for tick (the formulas are one piece
  of code and numpy == torch is tested; the event timing is by reading mjlab's step order), every reward weight.
- **Suggested first runs** (4096 envs, checkpoints every 100): `-Rung2-Ramp` 600 its, gates 200 / 400 / 599;
  `Kim-Rung2-Bio` 800 its, gates 200 / 400 / 600 / 799; `-Rung0-Full` 400 its, gates 200 / 399 (`--gates 0 --seeds 30`).
  Only KIM's warm-start checkpoints are on disk.

## 2026-10-11 — App audit, Unity side (no training, no contract change)

- Nothing here touches ctrl, the model or the step size: `PolicyRunner` gained only a stopwatch (`BrainMs`), the
  `ActionLog.Tick` call after the brain ran and `DrawTraits` (the same three draws in the same order the nine events
  did inline — heats stay seed-exact). `GameSettings.SimSpeed` is `Time.timeScale` (fixedDeltaTime stays 0.005).
  Instant replay runs at timeScale 0 (no mj_step), writes recorded qpos + `mj_kinematics`, then restores qpos and
  calls `mj_forward`; checked in Event 19: pelvis back to within 7 mm after 10 s more of live standing, no errors.
- Parity gates after the change: EditMode suite 87/87; `gate_report_kim_rung0.json` regenerated for the k0_v2 brain
  deployed this morning (G2-G4 pass, G4 5 s drift 2.3 mm); the other 38 reports only changed their timestamp (restored).
- `AthleteTelemetry` now also reports mean effort, a display-only tiredness (effort above 15 % fills it in ~25 s of
  all-out work, 12 s to recover), the fastest joint and the share of steps over 18 rad/s, slide speed
  (`mj_objectVelocity` of the touching geom, tangential part) and the contact count. Event 19, MATT, mid-race:
  effort 13-19 %, joints 6-7 rad/s, 1 % of steps over 18 rad/s, brain 0.2 ms (8 brains 1.8 ms per tick).
- Unity action logs: `parity/actions/unity/e<event>_h<heat>_<time>.csv.gz` (1.4 MB for a 23 s 8-lane heat) and
  `heats.jsonl` (places, marks, traits, telemetry bests per lane). `training/tools/report_actions.py` reads the Python
  gate logs; the Unity files share the folder but not yet the report.
- Brain clean-up: `parity/brains` 644 → 98 files, `Models/Brains` 262 → 33 MB; `tools/brain_manifest.py --check`: 22
  deployed brains, 0 problems. Checkpoints in `training/runs` were not pruned (script: `tools/prune_checkpoints.ps1`).

## 2026-10-11 23:30 · Training audit, third pass: servo calibration, observation normalisers, fast sim for the staged tasks

No training run, no GPU benchmark (the queue was training `mgetup_prone_v1`), no MJCF / contract / brain / existing
task changed (`dump_task_cfgs.py --diff`: 0 existing cfgs moved, 9 new IDs; pytest 90/90). The 2026-09-29 and
2026-10-11 14:16 audits already cover rewards, curriculum, DR, bio limits and the registry; this pass looked at what
they left out.

- **Servo calibration (CPU, default stance, each joint against its distal inertia; all five bodies).** Same picture on
  every body, as the Froude scaling intends: natural frequency 1.2-2.2 Hz at trunk / hip / shoulder, 3.5-4.5 Hz at
  knee / elbow, 7-19 Hz at hip rotation / ankle (their own segment is light; in stance the ankle carries the body).
  Damping ratio 0.5-1.3 for hips, knees, shoulders, elbows. Two things stand out:
  the **trunk is soft and under-damped** (abdomen flex / lateral: 1.1-1.3 Hz, damping ratio 0.32-0.41; zombie lowest),
  and mattbio's hip rotation / trunk twist / ankle inversion reach their torque cap at a 15-17° position error (one
  action unit is 14°), so those three joints are effectively torque-limited, as the bio caps intend.
  **Decision: no gain change.** kp / kv live in the MJCFs (fingerprint), every deployed brain has learned these servos
  and passes its gates; retuning means a new body version and retraining all 22 brains. If trunk sway ever becomes
  the complaint, the candidate is abdomen kv 30 → ~45 (damping ratio ~0.6) on a new body ID.
- **Actions.** Continuous position targets (default + 0.25 rad × action, clipped to the joint range in training and
  inside the ONNX graph alike) — nothing discrete to map; the clip is the mask. Action std of the deployed KIM brains
  0.22 (≈ 3° of target noise). No change.
- **Observation normalisers (16 newest checkpoints on disk).** Inputs that never vary in a task have std 0 and are
  divided by the 0.01 floor: command (3) + gait clock (2) in Rung 0 and get-up, lateral command in Rung 1 / crawl. A
  non-zero value there at run time would enter the network 100× amplified. Checked the callers: Python `Recovery` feeds
  zeros; Unity `TrackRaceEvent.Stand` / `SlalomEvent` zero the command and switch lane keeping off before
  `SetRecovering(true)` — **no live bug**. Pelvis height is the most amplified real input in the Rung 2 brains (std
  0.018 m against ±0.02 m training noise). Recommendation for every new export of a Rung 0 / get-up / crawl brain:
  `export_brain.py --clip-obs 5` (exists since this morning, default off).
- **Buffers of the self-colliding bodies.** `scale_to_body` gives them nconmax 160 / njmax 800 on a guess. Measured on
  CPU MuJoCo (deployed Rung 0 + Rung 2 brains, G1 drills, 4 seeds, 4-cube scene): peaks 30-34 contacts / 120-136 rows
  per world on MATT, zombie, GRANDMA and KIM alike — self-collision adds nothing measurable when the athlete is on its
  feet. With one pool cube that is ≤ 22 / 88, so `fast_sim`'s 64 / 256 holds for all four bodies (not for lying tasks).
- **Throughput gap closed (staged).** The nine tasks added this morning (Rung0-Full × 4, Rung2-Ramp × 4, Kim-Rung2-Bio)
  ran the slow sim: 4096 envs, 4 cubes, 160 / 800 buffers — the live run does 1.95 s / iteration (1.6 s collection +
  0.3 s learning ≈ 50k steps/s) where recipe v5 measured 105-137k. New IDs `<task>-Fast` (`tasks._register_fast`:
  `fast_sim` + `matt_ppo_v5_cfg`, half the iterations, checkpoints every 50). Smoke-tested at 16 envs (KIM Full / Ramp
  / Bio, zombie Ramp: finite, mirror exact); not benchmarked at 8192 envs on these bodies and no PPO iteration run.
  Decision rate stays 50 Hz (decimation 4 at 5 ms is in the contract).
- **Host.** While training: GPU 40-65 % busy at 86 °C / 79 W, Windows power plan "Balanced", three Unity processes
  open. The earlier measurement stands (Extreme Performance + Cooler Boost: +30-40 %).
- **Staged, not started:** `training/staged_queue.txt` (three KIM lines in queue format with their warm-start prep;
  the queue does not read that file). `runs/queue.txt` was not touched.
- **Noted, not changed:** `biomech.HILL_W_MAX` is in MATT units for every body while the joint-speed bar is scaled
  (18 rad/s ÷ √λ) — settle it with the contract v5 constants.

## 2026-10-12 01:45 · Goal "all behaviours at 100 %": GRANDMA walks 10/10 (g2_v9); zombie warm start from KIM fails → own chain
- **Where the roster stood** (parity/brains/manifest.json): below the bar were GRANDMA Rung 2 (g2_v8 8/10), zombie Rung 2
  (z2_v11 8/10 at one checkpoint), zombie get-up (none) and the mattbio stance drills (torso 7, march 9, reach 0-1 of 10).
  The GRANDMA / zombie checkpoints of September are gone (fresh clone), so nothing could be continued.
- **g2_v9** — `PoOlympic-Grandma-Rung2-Omni-Cap28-Bands` (unchanged task), `plain_init --max-std 0.3` from **KIM's**
  k2_v3 it 600, 1500 its, 64 min (the first 23 its took 9 min: Unity's EditMode suite was running). G1 Rung 2:
  **8/10 at it 300, 10/10 with 0 falls at it 600 / 900 / 1200 / 1499** (g2_v8 needed the whole Rung 0 → 1 → 2 chain
  and ended at 8/10). The only misses at it 300 were two stop segments (0.21 / 0.23 vs 0.187 m/s).
  30 seeds: it 600 29/30 (one stop), it 900 29/30 and it 1499 29/30 (the same 2.76 m/s run with a −0.41 rad/s turn,
  0.19-0.20), **it 1200 30/30**.
- **Event heats** (`compare_brain_events.py grandma`, 6 heats, her 3 lanes; g2_v8 it 1199 → g2_v9 it 1200): Gust
  Gauntlet still in 12 → **18 of 18** · Slalom 30.2 s (1 DNF) → 27.5 s (0) · Inverted Sprint 17.5 → 16.4 s · Crab 21.2 →
  20.3 s · Steeplechase 17.6 → 17.4 s · Terminal Velocity 2.75 → 2.77 m/s · Brake DQ 3 → 1 · Turntable 7.08 → 7.34 s
  (the one event that got slower).
- **Deployed (Python side):** `grandma_rung2.onnx` = g2_v9 it 1200 (export_brain; G1 10/10 re-run on the deployed file),
  critic, confidence group `rung2_g` (AUC 0.687), references `grandma_rung2` + `grandma_rung2_walk` (command 1.4, 0,
  0.3). Still to do: odds refit for Events 5, 9-13, 19, 22 (the 640 cached heats with her old brain are removed from
  `parity/odds/heats.jsonl`; 88 re-run so far), `brain_manifest.py`, Unity "Sync Contract + Brains" + G2-G4.
- **z2_v15, z1_v2, z0_v2 — three zombie runs wasted on ONE environment.** `PoOlympic-Zombie-Rung0-Stand`, `-Rung1-Run`,
  `-Rung2-Omni-Base` and the `-Rung2-Omni-*` line build a single env unless the CLI passes `--env.scene.num-envs` (as
  in September); the queue has no train-args field. z2_v15 (Wobble from KIM's k2_v3: reward 14.7 -> -2.0, 0/10 at every
  gate), z1_v2 (Rung 1 from the zombie's get-up brain: 0/10 x 4) and z0_v2 (Rung 0 from scratch: every episode ends on
  torso_tilt at 59 ticks, reward flat at -3.05) therefore say NOTHING about those warm starts. Seen only at z0_v2,
  from 0.5 s/it (4096 envs: ~2 s) and a from-scratch Rung 0 that did not move (z0_v1 was at 371 ticks by it 100) —
  GRANDMA's g0_v1 died the same way on 2026-09-30. Same trap still open on 10 current `matt` tasks.
- **Fix:** new IDs `PoOlympic-Zombie-Rung0-Stand-4096`, `-Rung1-Run-4096`, `-Rung2-Omni-Base-4096`,
  `-Rung2-Omni-Wobble-4096` (the same cfg functions with `scene.num_envs = 4096`), and the **new task
  `PoOlympic-Zombie-Rung2-Omni-Bands`**: GRANDMA's Cap -> Sharp -> Bands line bound to the zombie body
  (`grandma_env.G`, `zombie_env.zombie_rung2_bands_env_cfg`; 4096 envs from grandma_env). That line was never run on
  the zombie: z2_v12-v14 put its pieces on top of the yaw-heavy Wobble line. `dump_task_cfgs.py --diff`: 0 existing
  cfgs changed.
- **Queued:** z2_v16 (`Zombie-Rung2-Omni-Bands`, plain_init from KIM's k2_v3 it 600, 1500 its — the exact analogue of
  g2_v9) and z2_v17 (`Zombie-Rung2-Omni-Wobble-4096`, same start, 1000 its — the valid rerun of z2_v15). If neither
  carries over: Rung 0 -> Rung 1 -> Base -> Bands on the `-4096` IDs.
- **Mistakes:** `fit_odds.py --workers 6` during a GPU run took every core (each worker ran 2-9 threads) — the training
  process fell to 0.35 of a core and 2 % GPU for ~30 min and the odds heats themselves stopped advancing; stopped.
  `tools/export_critic.py` raised a NameError in its self-check (`sd` undefined) — fixed.

## 2026-10-12 05:50 · Zombie Rung 2: five valid runs, none beats the deployed 8/10; what is left is the top-speed run with a turn
All on 4096 envs, 0 falls at every gate. G1 Rung 2, 10 seeds (bar 0.157 m/s = 0.2 x SS):
| run | task | from | gates |
|---|---|---|---|
| z2_v16 | Rung2-Omni-Bands (new) | KIM k2_v3 it 600 | 0 · 0 · 0 · 1 · — : stands and side-steps, forward runs die (2.76 m/s command: 0.5 m/s by it 600) |
| z2b_v1 | Rung2-Omni-Base-4096 | KIM k2_v3 it 600 | 1 · 1 · 4 · 4 (it 500-1999): full-length episodes by it 50, track_lin 1.47 at it 200 |
| z2_v17 | Rung2-Omni-Wobble-4096 | z2b_v1 it 1999 | 0 · 1 · 2 · 1 · 0 |
| z2_v18 | Rung2-Omni-Bands | z2b_v1 it 1999 | 2 · 5 · 4 · 3: turns, backward walk and side-steps fixed; runs with a turn collapse (2.23 m/s, wz -0.45: 1.87 m/s RMS) |
| z2_v19 | Rung2-Omni-Bands2 (new: forward top 2.83 m/s, KIM's band shares) | z2b_v1 it 1999 | 5 · 2 · 4 · 5: no collapse, but 5-8 % slow at the top and it stops turning in a sprint (2.76 / -0.42 rad/s: -0.08) |
| z2_v20 | Rung2-Omni-Wobble-Polish (new: Wobble + two bands, KL 0.004, std 0.15) | the DEPLOYED brain rebuilt from its ONNX | 3 · 2 · 3 · 3 · 3 · 3 · 3 · 3 (every 50 its) |
- **The cross-body start works for the zombie too** when the task builds 4096 envs and starts on the base recipe; on
  the final recipes it re-learns "fast command = do not run", as the log of 2026-09-29 and g2_v4 already said.
- **`tools/onnx_to_checkpoint.py`** rebuilds a checkpoint from a deployed brain + its critic (actor, both normalisers,
  value net; action std chosen; ctrl vs the brain 5e-7) — a way back to a brain whose run directory is gone.
  z2_v20 from it fell 8 -> 3/10 within 50 iterations of small steps and stayed there: z2_v11 it 400 is an isolated
  good point, as z2_v12-v14 found in September.
- **Deployed `zombie_rung2` today: 8/10** — two misses, a 2.76 m/s run with a -0.42 rad/s turn (0.21) and a slow
  diagonal side-step (0.2, -0.17: 0.17). New report-only field `lin_rms_stride` in the gate JSON (the velocity error
  vector averaged over one stride, as `yaw_rms_stride`): its speed error is 0.64 of the per-tick one (median), the
  side-step passes, the fast run with a turn stays over (0.18) -> **9/10 under a stride-averaged speed bar**. That
  last one is a real shortfall at 92 % of the zombie's envelope while turning, not measurement noise.
- **Not deployed:** nothing of these runs; `zombie_rung2.onnx` is unchanged. Open choices are the user's: judge the
  zombie's speed per stride (as its yaw since 2026-09-29), lower the G1 sprint top for it (GRANDMA's 2.8 m/s
  precedent), or accept 8/10.
- **08:00 — queue empty, GRANDMA's deployment finished.** Odds refitted on 640 fresh heats with g2_v9 it 1200 (Events
  5, 9-13, 19, 22; `odds_model.json`), Unity "Sync Contract + Brains" (headless) and `ParityGateTests` 39/39, brain
  manifest 25 brains / 0 problems, pytest 90/90, `check_bodies.ps1` clean. KIM (poolympic-c8's jobs): k2f_v1 keeps
  G1 Rung 2 10/10 at it 100-300; ks_v1 (stance from her walking brain, 1500 its) has no stance drill until it 1499,
  where the flamingo drill is 10/10 and walking 6/10. My rs_v9 (the same recipe on mattbio) was dropped unrun.

## 2026-10-12 10:45 · MATT stance drills: march is 10/10 (the 9 was the stopwatch), torso 7/10 is the joint range; zombie z2_v21
- **z2_v21** (`Zombie-Rung2-Omni-CapBands`, new: base rewards + lateral cap, 2.83 m/s top, bands, symmetric runner;
  from z2b_v1 it 1999): 1 · 2 · 1 · 1 · 0 of 10. Seven valid zombie runs now, none above the deployed 8/10.
- **rs_v9** (`MattBio-RungS-Stance-v8`, 1200 its, from the DEPLOYED rs_v8 it 1199 rebuilt with
  `onnx_to_checkpoint.py` on a KIM stance checkpoint as template): squat 10 throughout; march 8 · 7 · 9 · 9 · 5 · 5 ·
  0 · 5; torso 7 · 7 · 8 · 8 · 7 · 7 · 5 · 7. Specialists from rs_v9 it 600 (new IDs `MattBio-RungS-March` / `-Torso`,
  65 % of the commands on one skill): rsm_v1 march 4 · 5 · 8 · 7 · 3 · 5; rst_v1 torso 7 at every gate (march 9).
- **March: the one failing seed was the measurement.** Seed 1006 commands 1.294 Hz; the drill counted lifts over its
  8 s window, which can only read multiples of 0.125 Hz — 1.25 or 1.375, the second 6 % off against the 5 %
  tolerance. `evaluate_stance.drill_march` now times the lifts (first to last; the count is kept as
  `cadence_count_hz`); bars unchanged. The deployed march brain reads 1.295 Hz there: **rs_v8 it 1199 march 10/10**
  (rst_v1 it 200 too; rs_v6 it 2550 stays 0/10: it does not lift). pytest 90/90.
- **Torso: 7/10 is the body, not the brain.** The three failing seeds each contain one aim with |yaw| 0.93-1.0 rad at
  a pitch of -0.15 to +0.07; the error there is 0.20-0.21 rad after 2 s and after 6 s, identical for rs_v8 and the
  specialist. Chest yaw against the pelvis comes from `abdomen_twist`, range +-45 deg = 0.785 rad; the drill's
  approved range is +-1.047 rad (60 deg). Leaning forward (pitch 0.6) reaches 0.94 by compounding the bend; upright
  it cannot. No training closes that: either the torso-yaw range of the drill / event comes down to what the joint
  allows (about +-0.75 rad), or the body gets a wider twist (new body ID, every brain retrained). User decision, like
  the reach radius (open since 2026-10-02).
- **Per-event stance brains today:** squat 10/10 (rs_v6 it 2550 / rs_v8 it 1199), flamingo 10/10 (rs_v6 it 2550),
  march 10/10 (rs_v8 it 1199), torso 7/10 (range), reach 0-1/10 (radius). Nothing new deployed.

## 2026-10-12 (evening) · Crowd contact: collision re-check, pack rule for Event 19, crowd fine-tune (nothing deployed)
- **Collision re-check (user: "verify players collide and cannot intersect each other's mesh").** Detection: audit
  32/32 scene lineups, Unity `CrowdContactTests` 15/15, roster / mattbio scenes 100 % of cross-lane part pairs.
  Depth (new, scratch measurement): the colliders sit inside the skin (38-49 % of skin vertices outside; median 8-19
  mm, p95 26-55 mm, worst 51-115 mm at shoulders / GRANDMA's hands) and contacts are MuJoCo-default soft: a standing
  athlete thrown sideways into another overlaps 17 / 24 / 34 mm (median) at 1 / 2 / 6 m/s, worst 59-72 mm (forearm on
  forearm), no part fully through another below 12 m/s. Left as is (a fix = new model version).
- **`tools/crowd_heats.py` (new):** CPU heats of 19 and 11 per lineup, each seed with crowd contact on and off; falls,
  finishers, pair-seconds by body pair. Old lane break (lines at half their distance to lane 1), 6 heats: all-MATT 2.3
  falls / heat with contact, 0 without; mixed lineups 0 falls and 0.1-0.4 pair-s. The cause is speed, not width: the
  field splits into same-body pairs (MATT 4.0, KIM 3.7, zombie 2.9, GRANDMA 2.8 m/s) that run 4 lanes = 2.4 m apart.
  Slalom: 0-0.2 falls / heat and < 1 pair-s in every lineup (the tactician yields) — left alone.
- **Pack rule (Event 19, `track.pack_target` = C# `TrackRaceEvent.BreakSteer`):** past the break a runner lines up
  outside every racing runner of a lower lane within 1.5 m of it along the track, 0.953 x the two half-widths apart
  (half-width = widest collider at the start pose; two MATTs 0.61 m as before). Falls / heat and pair-s, deployed
  brains: all-MATT 3.2 (11.5), MATT / zombie 2.0 (10.4), + GRANDMA 2.3 (27), + KIM 1.5 (11); 0 with contact off. Every
  body is floored by a bump; they get up (7.2-8 of 8 finish). Unity: plays, half-width 0.320 m = Python, leaders pack
  ~0.6 m apart. Pre-existing, seen in both: a runner that falls strays 10-24 m off the track after getting up.
- **Crowd fine-tune (`matt_env.add_crowd_bumps`, tasks `PoOlympic-<Body>-Rung2-Crowd`):** the deployed Rung 2 recipe +
  mjlab `apply_body_impulse` on each forearm and the chest (up to +-300 N x mass scale per axis, 0.03-0.25 s, every
  1.5-4 s each; measured race bumps: peak 170-500 N, 0.03-0.25 s, 35 % on forearms), KL 0.004, entropy 0.001, 400 its.
  This is a proxy for 2-4 colliding athletes per env, judged on real 8-athlete heats.
  - **MATT r2c_v1** (from rung2.onnx via onnx_to_checkpoint): G1 10/10 at every checkpoint but it 300 (9/10). All-MATT
    sprint falls / heat 3.2 -> 2.5 (it 100) -> 1.7 (it 200) -> **0.7 (it 350)**. Cost at it 350 (event heats vs the
    deployed brain): peak 4.04 -> 3.89 m/s, inverted sprint 14.9 -> 15.5 s, crab 18.3 -> 21.2 s, slalom 24.8 -> 27.4 s.
  - **KIM k2c_v1:** G1 10/10 at it 50 / 100, then 8 -> 3/10 by it 399 (no falls, speed tracking). it 100 in the mixed
    sprint: 1 -> 0 falls in 6 heats (noise level).
  - **GRANDMA g2c_v1:** G1 10/10 at it 100, 4/10 by it 300 (stopped at the user's request at it ~340). it 100: 6 -> 7
    falls, no gain.
  - **Zombie z2c_v1, MATT r2c_v2 (second stage):** queued, not run (`#stop`).
- **Not deployed.** Open (user): MATT r2c_v1 it 350 trades ~4 % top speed for 4-5x fewer contact falls; KIM / GRANDMA
  need a gentler bump (their gate degrades after 100 its) or true multi-athlete envs.
- **Android deploy (23:50, Pixel 9 Pro): lit surfaces black, fixed.** Two causes. (1) `UberPost` "stripped from the
  build": the committed `UniversalRenderPipelineGlobalSettings.asset` had an EMPTY `m_RuntimeSettings` list (left by a
  `git checkout` of that file after an earlier build); the build repopulates it, and the next build from the populated
  file ships the post-processing shaders — WRONG, see the correction below. (2) The decal renderer feature (BroadcastFx scuff
  marks, 2026-10-12) on technique Automatic takes the DBuffer path on the phone and every lit material renders black
  (crowd / hair / particles still show). `Mobile_Renderer` decal technique = Screen Space (`EnsureDecalFeature` sets
  it). Verified on device: Event 1 renders, 60 fps.
  Correction (00:10): the stripped `UberPost` came back on the 4th build with the populated file. What the two bad
  builds share: they were started with the editor on the Windows target (the build switched platform itself); the
  three good ones started on Android. `AndroidBuild.BuildScenes` now switches first and asks for a second call.
- **Athlete under-light (user, phone: "shadows very dark, under the chin").** `StadiumLook.AthleteFill` adds a third
  shadow-free light on the athlete rendering layer, pointing up from the front (pitch -50, 0.5); menu "Update athlete
  fill lights (all scenes)" added it to the 15 event scenes (bakes untouched). Before / after on the phone:
  `SCREENSHOTS/android_lighting_2026-10-12/report.html`.
- **More light (2026-10-13, user: "fix more").** `AthleteFill`: fourth athlete fill (`Fill_30`, 0.7), `Fill_300` 0.6 ->
  0.9, `Fill_Up` 0.5 -> 0.7, and two shadow-free directional lights on the default layer (`Arena_45` / `Arena_225`,
  0.35 each) in the 15 event scenes; realtime, bakes untouched. Phone: faces lit on both sides, 60 fps (report image 4).

## 2026-10-13 · Goal "every behaviour works 100 %": event completion by the deployed brains
- **Survey (3 CPU heats of each of the 15 events, deployed brains):** 100 % of the athletes finish 9, 10, 11, 12, 13,
  19, 27; elimination events 1 / 5 / 6 end by design; short of 100 %: 7 Cadence March 5 of 24 DONE, 3 Deep Squat 7 of
  24 DONE, 8 / 23 crawls (MATT 11 of 72 DNF, GRANDMA 3 of 20 in the trench), 22 one MATT DQ (its seeded late brake:
  the rule, not a failure).
- **7 Cadence March -> 24 of 24, no training.** The brain sees velocities, never a position, so over 36 s it drifts
  past the 0.6 m rule. Spot hold in the event (`march.hold_command` = C# `CadenceMarchEvent.HoldCommand`): velocity
  command towards the spot, gain 1.0 /s, cap 0.2 m/s; the march brain follows it while marching. DONE 5 -> 24 of 24,
  drift 0.60 -> 0.10 m (median max), points 74.6 -> 93.3; gains 0.5-2.0 all 24 of 24. The same command makes a
  squatting athlete step (0-14 of 24): not used there.
- **3 Deep Squat -> 80 of 80, new brain `rsq_v2_it500`.** rs_v6 it 3299 does not step: its feet slide apart ~1-2 cm
  per rep (stance 0.18 -> 0.43 m over the 12 reps) until one is 20 cm from its mark. rsq_v1 (`RungS-Squat`: 65 % squat
  commands, 80 % of them metronome reps in the event's depth / tempo range, foot-speed penalty): depth error 19 -> 4
  mm but stance 0.72 m, 24 of 24 STEPPED — a speed penalty does not see a centimetre per second. rsq_v2
  (`RungS-Squat2`: + stance-width reward, 0.18 m, std 0.06, from rsq_v1 it 400): it 300 / 400 / 500 / 599 all 24 of 24;
  10 heats: **it 300 80 of 80, it 500 80 of 80**, 114.9 points (rs_v6: 86.0), squat drill 10/10. Event 3 = rsq_v2 it
  500 (critic exported, manifest, `EventScenes.SquatBrain`, test asserts 8 of 8 DONE).
- **Crawls: not bumps.** `*-Crawl-Crowd` (Crawl3 + pack bumps at half force; mcc_v1, gcc_v1): 80 heats, 7.10 -> 7.20
  of 8 finish = no change. Contact off: 100 % finish. Traces: MATT, GRANDMA and KIM crawl SIDEWAYS. `crawl_heading`
  (pelvis x + z axes) turns away from the spine when the pelvis rolls, and the Crawl3 brains travel along it with the
  spine 92 / 65 / 100 deg off (zombie: straight). They fill 1.5 m of a 1.1 m lane and swing 1.3 / 0.9 / 1.7 m sideways
  when they set off. Start-rule tweaks in the event (slow until straight, turn during the countdown) halve the swing
  alone but not the DNFs: discarded. `*-Crawl-Square` = Crawl3 + `mdp.crawl_square` (spine along the travel direction,
  w 2). mcs_v1 (from the deployed brain): spine -92 -> -81 / -76 / -70 deg at it 200 / 400 / 600 — too slow; mcs_v2
  starts from the ZOMBIE's straight crawl instead (cross-body warm start).
- **Crawls -> 512 of 512, new crawl brains for MATT, KIM, GRANDMA.** Cross-body start from the zombie's straight crawl:
  mcs_v2 (`Matt-Crawl-Square`, 1000 its) crawls with the spine 19 deg off travel and 30 m in 26.5 s, but no longer
  steers (+-0.4 rad/s command -> +-0.03; lane error 10 m). `*-Crawl-Square2` = Square + yaw-rate reward w 1 -> 3, kernel
  halved: **mcs_v3 it 999** (from mcs_v2): steering +0.41 / -0.33, start swing 1.31 -> 0.18 m, 30 m in 24.6 s (26.1);
  **kcs_v2 it 1499** (KIM, from the zombie brain): spine -80 -> -18 deg, swing 1.73 -> 0.11 m, +-0.41, 27.3 s (28.6);
  **gcs_v2 it 1200** (GRANDMA, same): spine 65 -> 0 deg, swing 0.93 -> 0.08 m, +0.43 / -0.40, 27.1 s. Own-brain starts
  (kcs_v1, gcs_v1) did not move the gait. Heats with crowd contact, 8 per event and lineup (4 lineups, events 8 + 23):
  **8.00 of 8 finish in all 64 heats, 0 tumbles** (before: 6.9-7.9); the trench takes GRANDMA 14.9 s (30.6) and KIM 14.6
  s (31.2). Deployed as `crawl_matt / crawl_kim / crawl_grandma.onnx` + critics; odds refitted for 8 / 23, confidence
  for the crawl groups (GRANDMA / KIM: no "danger" tick left in the fit data -> uncalibrated, as before for stance).
- **Unity:** brains synced, `Event_DeepSquat` rewired to rsq_v2 it 500. Play-throughs: squat 8 of 8 DONE (113-116 pts),
  march 8 of 8 DONE (drift 0.05-0.28 m), All Fours 8 of 8 in 25 s, Trench 8 of 8 in 13 s, 0 tumbles. ParityGateTests
  39/39, pytest 92/92.
- **Survey after (3 heats per event):** every athlete completes 3, 7, 8, 9, 10, 11, 12, 13, 19, 23, 27. Left as they
  are: 1 / 5 / 6 end by elimination (the event), 22 one MATT over the line (its seeded late brake). Not event
  completion, still open: contact falls in the 19 sprint (they get up and finish; MATT candidate r2c_v1), the fallen
  sprinter's 10-24 m detour, stance events MATT-only, zombie G1 8/10, torso 7/10 and reach 0-1/10 (joint range / reach
  radius: user decisions).


## 2026-10-13 (afternoon) · Crowd2 bump fine-tunes: trained, gates hold, no gain in mixed heats (nothing deployed)

- **Runs (400 its each, ~15 min, Balanced plan, GPU 86-88 C):** G1 Rung 2 pass rate, 0 falls everywhere. MATT r2c_v3
  10/10 at it 100 / 200 / 300 / 399. KIM k2c_v2 10 / 10 / 9 / 7 of 10. GRANDMA g2c_v2 10 / 10 / 10 / 9 of 10 (g2c_v1: 4
  of 10 by it 300). Zombie z2c_v2 9/10 at it 50, then 3 / 3 / 7 / 5 / 8 / 4 / 4 of 10: unstable.
- **Event 19 heats, contact on, 12 heats (`crowd_heats.py`, `parity/crowd_heats_base.json`, `_crowd2*.json`), falls per
  heat (finish of 8):** all-MATT 3.25 (8.00) -> **1.08** (8.00) with r2c_v3 it 399. Mixed mzgkmzgk, deployed 1.83 (7.92):
  all four candidates 3.17 (7.33); MATT's only **3.25 (6.83)**; KIM it 200 only 1.50 (7.83); GRANDMA it 300 only 1.42
  (7.92); zombie it 50 only 1.25 (7.75); KIM + GRANDMA + zombie 1.75 (7.75).
- **Reading:** the MATT candidate helps among MATTs and hurts the mixed race: matt-matt contact 2.2 -> 7.7 pair-s,
  grandma-matt 0 -> 2.6, MATT falls 0 -> 6, GRANDMA 9 -> 14 (speeds not checked; guess: the slower brain stays in the
  pack). The other three are within noise of the deployed brains. **Not deployed.** The forearm / chest impulse proxy
  does not carry over to mixed packs: next would be true multi-athlete envs, not a third bump recipe.

## 2026-10-13 (night) · Events 2 and 4 playable (no training); Rung 4 slope tasks for Event 14

- **User: "create a playable scene from the missing numbers", then 8 hours unattended.** 13 of the 15 missing events need a
  skill that no brain has; Events 2 and 4 had brains and two open range questions. Decided in the event, not the
  contract: the targets stay inside what the body reaches.
- **Probes (solo mattbio, CPU):** rs_v8 it 1199 holds every torso aim within 2 deg for yaw up to 45 deg at every pitch
  (-20 .. 40 deg), settles in < 1.2 s; at 50 deg it is 5 deg short, at 60 deg 12-15 deg (the spine's 45 deg twist).
  rs_reach_v1 it 1199 (restored from git c8ed003; removed in the 12aad03 clean-up): hand error 1-9 cm for targets up
  to 0.78 of the reach radius and elevation <= 45 deg, 10-29 cm beyond (0.82-0.95, or az > 60 deg with el > 30 deg).
- **`events/targets.py` = Unity `TargetEvent`** (one module, kind archer / reach; scenes `scene_archer8` / `reach8_mattbio`
  = the venues' 2 x 4 station grids via compose_mixed): 12 rounds, 2.0 -> 1.0 s. Archer: the aim flies to a new one
  (yaw +-45, pitch -15 .. 35 deg, >= 30 deg apart) over 60 % of the round, error over the whole round, zero points at
  0.25 rad. Reach: alternating arms, 0.50 -> 0.78 of the radius, error over the last 40 %, zero points at 15 cm. Out =
  FELL / STEPPED (20 cm), balance bonus 10. CPU heats (3 seeds): archer 8 of 8 DONE, 71-80 of 130; reach 8 of 8 DONE,
  94-107. Unity play-through: archer winner 71.4, reach 104.3, 8 of 8 DONE each; menu 17 events, hub venues 2 / 4 set.
  The reach spread is ~2 points per heat: traits barely separate the field (open: tighter tolerance or shorter holds).
- **Event 14 The Alpine Ramp, plan:** the ramp = the flat plane with gravity tilted by the slope (`matt_env.add_slope`,
  tasks `PoOlympic-<Body>-Rung4-Slope8` / `-Slope15`, each body's deployed Rung 2 recipe, forward commands to half the
  flat envelope, start facing uphill). Same physics in the ramp's frame, no terrain, no observation change; the event
  (`track.py` mode `ramp`, 10 m = the venue's ramp, `tools/ramp_heats.py`) and Unity use the same tilted gravity.
  Baseline, deployed flat-ground brains on 15 deg: 0 of 24 finish (MATT / KIM / GRANDMA fall, the zombie stands still).

## 2026-10-14 (night) · Event 14 The Alpine Ramp: slope brains for all four bodies, scene playable

- **Recipe (`matt_env.add_slope`):** each body's deployed Rung 2 recipe on the flat plane with gravity tilted by the slope
  (uphill = +x), start facing uphill, staged command curriculum dropped, 4096 envs, entropy 0.001. `climb=True`: forward
  commands 5 % .. `top` x the flat top speed, half the stand-still commands. Tasks `-Rung4-Slope8` (fast range 0-50 %),
  `-Slope15` (same), `-Slope11c` / `-Slope15c` (climb, top 30 %), `-Slope15s` (climb, top 15 %).
- **Heats (`tools/ramp_heats.py`, 9 m up 15 deg, mixed lineup, 2 per body):** flat brains 0 of 24 (MATT / KIM / GRANDMA
  fall at once, the zombie stands). MATT: Slope8 400 its -> 8 deg 10 m in 8.4 s; Slope15 1000 its -> 15 deg 4 of 4, 21 s;
  his brain climbs fastest when asked for 1.0-1.5 m/s (17.5 s), stalls at 3. KIM: Slope8 ok, **Slope15 stood still** (0 of
  4 at it 200-999; no falls, episode length 1000, track_lin 0.5: survival paid, a speed she cannot reach paid the same
  standing as creeping); Slope11c 4 of 4 on 11.5 deg; Slope15c still 0 of 4 (0.1-0.2 m/s); **Slope15s** 6 of 6, 26 s.
  GRANDMA and the zombie went 8 -> 11.5c -> 15s directly: 12 of 12 each. MATT Slope15c (from Slope15 it 999): 20.7 s.
  GRANDMA's Slope15s gets slower with training (it 200 26.8 s, it 799 28.8 s; picked by heats).
- **Deployed (`slope_<body>.onnx` + critics, manifest role "slope"):** MATT ms15c_v1 it 200, KIM ks15s_v1 it 799,
  GRANDMA gs15s_v1 it 200, zombie zs15s_v1 it 799. Race command 0.6 (MATT units = 15 % of each body's top speed), 45 s
  limit. Mixed field, 6 heats: **48 of 48 finish, 0 falls** (MATT 20.7 s, GRANDMA 26.8, KIM 28.0, zombie 34.4).
- **Unity:** `TrackRaceEvent.Mode.Ramp` tilts the model gravity after the fingerprint check; runners hold the start spot
  during the countdown (a standing slope brain creeps 0.6-0.9 m downhill) and the top after the finish.
  `EventScenes.BuildAlpineRamp`: stadium, sun, fill lights, probes turned -15 deg about the ramp foot (shifted 2.2 m so the
  foot is 0.2 m behind the start line; at 1.8 m the athletes floated 10 cm above it); `RampView` keeps the broadcast
  camera's position and aim and turns its up vector by -15 deg (turning the whole camera about the focus athlete put the
  head-on shots inside the ramp). Mixed play-through: 8 of 8 home by 34 s. Menu 18 events, hub venue 14 set.
- One CUDA "unspecified launch failure" killed ks15_v1 at it 764; the queue resumed it from it 700.
- Review page: `parity/tb/slope/review.html`. Open: the athletes climb at 0.3-0.6 m/s (a person walks up 15 deg at about
  1 m/s); MATT's brain does 0.8 m/s when asked for more, the others stall above their 15 %.
