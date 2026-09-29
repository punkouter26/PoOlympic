# Proposal — contract v4 "stance skills" (events 2, 3, 4, 6, 7)

Status: **approved** (user, 2026-09-29): option A (one shared brain), MATT first, ranges and physics limits as below. tasks.md calls this
"contract v3"; v3 is already taken by the gait-clock change of 2026-09-28, so this is **v4**.

## Why

Five catalogue events need behaviours the current command (vx, vy, ωz) cannot ask for:

| # | Event | Needs |
|---|---|---|
| 2 | Torso Archer | aim the upper body at a moving target, feet planted |
| 3 | Deep Squat Endurance | a pelvis height target (rhythmic squats) |
| 4 | Precision Javelin Reach | a hand target (one arm, stance anchored) |
| 6 | The Flamingo Classic | stand on one chosen leg |
| 7 | Cadence March | march in place at a given cadence (gait clock running with zero velocity) |

## Proposed interface: an 11-dim "skill command" block

Appended **after** the current 84 obs, so every existing offset stays the same (obs 84 → 95). Zero block = today's
behaviour.

| Term | Dim | Range | Used by |
|---|---|---|---|
| pelvis height target (relative to standing default) | 1 | −0.45 … 0 m (× λ per body) | 3 |
| lift foot one-hot (left, right) | 2 | {0,1} | 6 |
| march cadence, knee lift | 2 | 0 or 0.8–2.0 Hz; 0.10–0.35 m | 7 |
| torso aim: chest yaw, pitch vs pelvis heading | 2 | ±60°, −20…+40° | 2 |
| hand target xyz (heading frame) + arm select | 4 | reach sphere 0.75 m from the shoulder; −1 L / +1 R / 0 none | 4 |

Gait clock: when the march cadence is > 0, the clock advances at that cadence even at zero velocity (today it freezes
below 0.1). All other terms leave the clock as in v3.

Parity: ObservationBuilder appends the block (G2 covers it), ONNX metadata gets `contract_version: 4` + the block layout;
Unity refuses v3 brains in v4 scenes and vice versa, as it does today for fingerprints.

## Training plan (recommended: option A)

**A — one "stance skills" brain (Rung S), MATT first.** Warm start from `rung2.onnx` with the new input columns
zero-initialised, so the first iteration behaves exactly like Rung 2. Each episode samples one skill (20 % each) or plain
locomotion (20 %, to keep Rung 2 skills). Per-skill tracking rewards plus the Rung 0/2 stability terms. One G1 drill set
per event. Estimate: 1 run of ~2–3 h plus gate iterations.

**B — five small event brains.** Simpler reward tuning, but 5 runs, 5 parity sets and 5 brains to ship.

Recommendation: **A**. Every event scene already loads one brain per body, and one shared brain keeps parity work and
APK size down.

## Decisions (user, 2026-09-29)

1. **Option A** — one shared "stance skills" brain (Rung S).
2. **MATT first**; the zombie gets its own run with the size-scaled ranges later (until then zombie lanes cannot enter
   events 2, 3, 4, 6, 7).
3. **Ranges as proposed.**
4. **Physics limits accepted**: wrists stay welded, no fingers; the Javelin Reach target is the forearm tip.

Next (no training): contract v4 + obs builder in Python and C#, G2 tests, the Rung S task scaffolding and G1 drills;
then the training run.
