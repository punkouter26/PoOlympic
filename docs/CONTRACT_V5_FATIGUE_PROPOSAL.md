# Proposal — contract v5 "fatigue block" + torque-speed and fatigue limits in the physics

Status: **proposal, not approved** (written 2026-10-11 with the training audit). Nothing deployed uses it: contracts v3
and v4, `contract*.json`, every MJCF and every exported brain are unchanged. The Python side below exists and is
tested; Unity has **nothing** of it yet. One staged task (`PoOlympic-Kim-Rung2-Bio`), not trained.

## Why

Two things real athletes have and ours do not:

| Gap | Today | Proposed |
|---|---|---|
| Torque falls with joint speed | only a reward penalty (`mdp.torque_speed_excess`, recipe v5); the physics keeps the flat cap | the actuator's force range shrinks with joint speed, on the same curve |
| Tiring | none: the 40th second of a sprint is as strong as the first | a per-joint-group fatigue level that lowers the force cap to 60 % at worst and recovers at rest |

A brain can only pace itself if it can feel how tired it is, so the six fatigue levels become observations. That is
an interface change, hence a contract version.

## Interface: a 6-value block after the 84 v3 observations (84 → 90)

Same pattern as the v4 skill block: appended, every v3 offset unchanged, the brain's input size / sidecar
`contract_version` picks the layout (84 = v3, 95 = v4, **90 = v5**). v4 + fatigue (101 inputs) is not defined yet.

| Offset | Term | Range |
|---|---|---|
| 84 | fatigue level, abdomen | 0 fresh … 1 exhausted |
| 85 | shoulder (both sides share one level) | |
| 86 | elbow | |
| 87 | hip | |
| 88 | knee | |
| 89 | ankle | |

A joint belongs to the group its actuator name starts with (`abdomen_*`, `shoulder_*`, `elbow_*`, `hip_*`, `knee_*`,
`ankle_*`). Left and right share a level, so the mirror map for PPO symmetry leaves the block as it is. No observation
noise on the block. Actions are unchanged (23 joint targets, same ONNX output mapping).

A v5 brain's own ONNX metadata carries a `fatigue_block` entry with the layout and every constant below
(`contract.fatigue_block()`); `contract.json` gets the block only once this is approved.

## The physics hook (one definition: `training/poolympic/biomech.py`)

Constants (all four fatigue numbers are proposals to tune):

| Name | Value | Meaning |
|---|---|---|
| `HILL_W_MAX` | abdomen 8, shoulder 20, elbow 20, hip 15, knee 22, ankle 18 rad/s | joint speed at which the torque available *with* the motion reaches 0 (the table the reward already uses; MATT units for every body) |
| `FATIGUE_EFFORT_REF` | 0.25 | effort (share of the cap) that exhausts a fresh group in `TAU_LOAD` seconds |
| `FATIGUE_TAU_LOAD` | 10 s | |
| `FATIGUE_TAU_REST` | 30 s | recovery time constant at rest |
| `FATIGUE_FLOOR` | 0.6 | force-cap scale of an exhausted group |

Once per control tick (dt = timestep × decimation = 0.02 s), per athlete, with `base_lo/base_hi` = the actuator's
force range as loaded (after the lane's strength trait), `cap = max(|base_lo|, |base_hi|)`, `tau` =
`mjData.actuator_force` as the previous tick's last substep left it, `qd` = the joint's velocity now:

    effort_g = mean over the group's actuators of |tau_i| / cap_i
    x_g      = effort_g / 0.25
    level_g  = clamp(level_g + 0.02 * (x_g * x_g / 10 - level_g / 30), 0, 1)
    f_i      = 1 - (1 - 0.6) * level_g(i)                      # fatigue: 1 … 0.6
    h_i      = clamp(1 - |qd_i| / w_max_g(i), 0, 1)            # torque-speed: 1 at rest … 0 at w_max
    lo_i     = base_lo_i * f_i * (h_i if qd_i < 0 else 1)      # h only on the side that pushes WITH the motion;
    hi_i     = base_hi_i * f_i * (h_i if qd_i > 0 else 1)      # braking torque keeps the full (fatigued) cap
    mjModel.actuator_forcerange[i] = (lo_i, hi_i)

A sustained effort `e` settles at level `min(1, 3 (e / 0.25)²)`, 63 % of the way after 30 s.

### Tick position

    read_state → **bio_limits** → advance_phase → build_obs (v3 terms, then the 6 levels) → infer → write_ctrl
               → apply_disturbance → mj_step × 4

The first tick after a reset does nothing in `bio_limits`: levels 0, force range = base. (Training applies the same
update at the end of the previous environment step — mjlab "step" event — so both sides use the state at the tick
start and the previous tick's force; a training reset also starts fresh, or at a random level, see below.)

## What Unity's PolicyRunner would have to mirror

1. **State per athlete:** `double level[6]`, `bool fresh`, `base_lo/base_hi[23]` captured once after the strength trait
   is applied; `ResetToDefault` sets levels to 0, `fresh = true` and writes the base range back.
2. **`BioLimits()` at the tick position above**, in doubles, operations in the order written (the Python reference is
   `biomech.BioLimits.tick`, 12 lines).
3. **ObservationBuilder:** append the 6 levels at offset 84 when the brain's sidecar says `contract_version` 5; refuse
   a v5 brain in a scene that does not run `BioLimits`, and the reverse (same rule as fingerprints and v3 / v4).
4. **Model fingerprint (G0):** `actuator_forcerange` is fingerprinted. Dump the fingerprint before the first
   `BioLimits` write, or compare against the stored base.
5. **Group / w_max tables:** by actuator-name prefix, from the brain's `fatigue_block` metadata (do not hard-code).
6. **Brain confidence:** the staged task trains a privileged critic (98 inputs: v5 obs + 8 values only the simulator
   knows). `tools/export_critic.py` now refuses such a critic. A v5 brain needs either a second critic fitted on the
   actor's observation or no confidence bar.

Parity work if approved: a v5 random test brain + reference run (G2 with the block, G3 / G5 with the limits on), as
`random_brain_v4` did for v4. `poolympic/reference.py` and `closed_loop.py` do not know the block yet.

## What exists today (Python, opt-in)

- `poolympic/biomech.py` — constants, the formulas (numpy and torch run the same code), `BioLimits` (CPU reference).
- `poolympic/contract.py` — `FATIGUE_VERSION` 5, `FATIGUE_LAYOUT`, `OBS_DIM_V5` 90, `build_obs(fatigue=…)`,
  `fatigue_block()`.
- Training: `mdp.bio_force_limits` (step + reset event), `mdp.obs_fatigue`, `matt_env.add_bio_limits`; mirror map in
  `tasks/symmetry.py`. Training episodes start with each group at a random level U(0, 0.8) — a 20 s episode is too
  short to get tired by effort alone; play / CPU / Unity start fresh.
- CPU gates: `evaluate.Sim` and the event lanes (`events/iron_pedestal._Lane`) switch the limits on for a 90-input
  brain. `POOLYMPIC_BIO=1` forces them on for any brain (probe an existing brain under the limits).
- Export: `tools/export_brain.py` stamps `contract_version` 5 on a 90-input checkpoint.
- Tests: `tests/test_audit.py` (fatigue rises under load, recovers, never below the floor; one-sided torque-speed
  scale; numpy == torch; a v5 brain in the CPU gate; mirror map).

## Measured before any training (KIM, `kim_rung2` = k2_v3 it 600, CPU, 2026-10-11)

- Effort in the G1 Rung 2 drill (action log, seed 1002, 55 s): mean per group 0.03-0.10 of the cap, 0.10-0.13 for hip /
  knee / ankle while sprinting; 99th percentile per actuator 0.07-0.42, ankle dorsiflexion 0.87-0.92.
- Fatigue levels reached in that drill with the constants above: abdomen 0.36, shoulder 0.13, elbow 0.07, hip 0.25,
  knee 0.39, ankle 0.60 (cap scale 0.76 at the ankle).
- Torque-speed scale at the fastest tick: hip 0.59, knee 0.50, ankle 0.48.
- **The current brain, never trained with the limits, passes G1 Rung 2 10/10 with both limits on** (10/10 without).
  With these constants the limits are mild for her; whether they should bite harder is the decision below.

## Training plan (staged, not run)

`PoOlympic-Kim-Rung2-Bio` = her production recipe (k2_v3, `-Bands2`) + the limits + the fatigue block + privileged
critic. One change at a time: shoves and DR stay k2_v3's.

    cd training ; $env:POOLYMPIC_BODY = "kim"
    uv run python tools/expand_obs.py runs/kim_rung2/2026-10-11_02-42-46_k2_v3/model_600.pt runs/kim_rung2/kbio_init --actor-zeros 6 --critic-zeros 14
    # queue line: kim | PoOlympic-Kim-Rung2-Bio | kbio_v1 | kim_rung2/kbio_init | 800 | --gates 2 --its 200,400,600,799
    uv run python tools/smoke_task.py PoOlympic-Kim-Rung2-Bio      # builds, steps, mirrors (done 2026-10-11)

The gate (`--gates 2`) runs the usual Rung 2 drill; a 90-input brain turns the limits on by itself.

## Decisions needed

1. Go / no-go for a v5 contract at all (Unity work: items 1-6 above, plus parity references).
2. The four fatigue constants: mild as proposed, or strong enough to change who wins a 45 s event?
3. Per-side fatigue (11 values instead of 6) — more realistic for the Flamingo and one-arm reaches, costs a mirror map
   and five more inputs.
4. Should `HILL_W_MAX` be Froude-scaled per body (× 1/√λ, like the joint-speed bar)? The reward term never was.
5. Stance brains (v4) with fatigue: a 101-input layout, or fatigue for locomotion brains only.
