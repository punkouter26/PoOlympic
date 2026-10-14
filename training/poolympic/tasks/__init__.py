"""PoOlympics mjlab tasks. Loaded through the `mjlab.tasks` entry point (pyproject.toml).
A process registers the tasks of its body only ($POOLYMPIC_BODY, default matt — contract/mdp load that body)."""

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .. import bodies


def _patch_rsl_code_state() -> None:
    """rsl_rl stores `git diff` in every run with strict UTF-8; parallel sessions' uncommitted files (non-UTF-8 bytes)
    made that crash the launch (2026-09-30). Same file, invalid bytes replaced."""
    import builtins
    import rsl_rl.utils.logger as L

    orig = L.Logger._store_code_state
    if getattr(orig, "_poolympic", False):
        return

    def store(self):
        real_open = builtins.open

        def lenient(file, mode="r", *a, **kw):
            if "b" not in mode and str(file).endswith(".diff"):
                kw["errors"] = "replace"
            return real_open(file, mode, *a, **kw)
        builtins.open = lenient
        try:
            return orig(self)
        finally:
            builtins.open = real_open
    store._poolympic = True
    L.Logger._store_code_state = store


_patch_rsl_code_state()

def _low_entropy(rl, coef: float = 0.001):
    rl.algorithm.entropy_coef = coef
    return rl


def _register_fast(task_id: str, fn, exp: str, its: int, runner, entropy: float | None = None) -> None:
    """`<task_id>-Fast` (STAGED 2026-10-11, not trained): the same recipe on the fast training sim (matt_env.fast_sim:
    8192 envs, one pool cube, nconmax 64 / njmax 256) with the 8-mini-batch PPO cfg; `its` = half the 4096-env run's
    (twice the samples per iteration), checkpoints every 50. The buffers hold for the self-colliding bodies too: CPU
    MuJoCo peaks of the deployed Rung 0 / Rung 2 brains in the G1 drills, 4-cube scene, are 30-34 contacts / 120-136
    rows per world on MATT, zombie, GRANDMA and KIM alike (12 contacts / 48 rows of that are the 3 parked cubes)."""
    from .matt_env import fast_sim, matt_ppo_v5_cfg
    rl = matt_ppo_v5_cfg(exp, max_iterations=its)
    if entropy is not None:
        rl.algorithm.entropy_coef = entropy
    register_mjlab_task(task_id=task_id + "-Fast", env_cfg=fast_sim(fn()), play_env_cfg=fn(play=True), rl_cfg=rl,
                        runner_cls=runner)


if bodies.current().name == "matt":

    from .matt_env import (matt_getup2_env_cfg, matt_getup3_env_cfg, matt_getup4_env_cfg, matt_getup5_env_cfg, matt_getup_env_cfg, matt_pedestal2_env_cfg, matt_pedestal_env_cfg, matt_ppo_cfg, matt_rung0_env_cfg, matt_rung1_env_cfg,
                           matt_rung2_env_cfg, matt_rung2_flight2_env_cfg, matt_rung2_flight3_env_cfg, matt_rung2_flight_env_cfg, matt_rung2_sym2_env_cfg, matt_rung2_sym3_env_cfg, matt_rung2_sym4_env_cfg, matt_rung2_sym5_env_cfg,
                           matt_rung2_sym_env_cfg)
    from .symmetry import SymmetricRunner

    register_mjlab_task(
        task_id="PoOlympic-Matt-Getup",
        env_cfg=matt_getup_env_cfg(),
        play_env_cfg=matt_getup_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("matt_getup", max_iterations=3000),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Matt-Getup2",
        env_cfg=matt_getup2_env_cfg(),
        play_env_cfg=matt_getup2_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("matt_getup", max_iterations=3000),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Matt-Getup3",
        env_cfg=matt_getup3_env_cfg(),
        play_env_cfg=matt_getup3_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("matt_getup", max_iterations=3000),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Matt-Getup4",
        env_cfg=matt_getup4_env_cfg(),
        play_env_cfg=matt_getup4_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("matt_getup", max_iterations=2500),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Matt-Getup5",
        env_cfg=matt_getup5_env_cfg(),
        play_env_cfg=matt_getup5_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("matt_getup", max_iterations=2500),
        runner_cls=MjlabOnPolicyRunner,
    )

    from .matt_env import matt_ppo_v5_cfg, matt_rung2_v5_env_cfg

    # recipe v5 on the current MATT body (staged): r2_v8 + fast sim + bio rewards; warm start r2_v8 it 600
    register_mjlab_task(
        task_id="PoOlympic-Matt-Rung2-V5",
        env_cfg=matt_rung2_v5_env_cfg(),
        play_env_cfg=matt_rung2_v5_env_cfg(play=True),
        rl_cfg=matt_ppo_v5_cfg("matt_rung2", max_iterations=600),
        runner_cls=SymmetricRunner,
    )

    from .stance_env import matt_stance_env_cfg

    # Rung S (contract v4 stance skills, events 2/3/4/6/7): warm start = rung2 (r2_v8) expanded by tools/expand_obs.py
    register_mjlab_task(
        task_id="PoOlympic-Matt-RungS-Stance",
        env_cfg=matt_stance_env_cfg(),
        play_env_cfg=matt_stance_env_cfg(play=True),
        rl_cfg=matt_ppo_v5_cfg("matt_stance", max_iterations=1000),   # 8192 envs: ~2x samples per iteration
        runner_cls=SymmetricRunner,
    )

    from .crawl_env import matt_crawl2_env_cfg, matt_crawl3_env_cfg, matt_crawl_env_cfg

    register_mjlab_task(
        task_id="PoOlympic-Matt-Crawl3",
        env_cfg=matt_crawl3_env_cfg(),
        play_env_cfg=matt_crawl3_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("matt_crawl", max_iterations=2500),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Matt-Crawl2",
        env_cfg=matt_crawl2_env_cfg(),
        play_env_cfg=matt_crawl2_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("matt_crawl", max_iterations=2500),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Matt-Crawl",
        env_cfg=matt_crawl_env_cfg(),
        play_env_cfg=matt_crawl_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("matt_crawl", max_iterations=2500),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Matt-Rung0-Stand",
        env_cfg=matt_rung0_env_cfg(),
        play_env_cfg=matt_rung0_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("matt_rung0", max_iterations=3000),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Matt-Rung1-Run",
        env_cfg=matt_rung1_env_cfg(),
        play_env_cfg=matt_rung1_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("matt_rung1", max_iterations=3000),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Matt-Rung2-Omni",
        env_cfg=matt_rung2_env_cfg(),
        play_env_cfg=matt_rung2_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("matt_rung2", max_iterations=2000),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Matt-Pedestal",
        env_cfg=matt_pedestal_env_cfg(),
        play_env_cfg=matt_pedestal_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("matt_pedestal", max_iterations=1000),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Matt-Rung2-Omni-Sym",
        env_cfg=matt_rung2_sym_env_cfg(),
        play_env_cfg=matt_rung2_sym_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("matt_rung2", max_iterations=1500),
        runner_cls=SymmetricRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Matt-Pedestal2",
        env_cfg=matt_pedestal2_env_cfg(),
        play_env_cfg=matt_pedestal2_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("matt_pedestal", max_iterations=1200),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Matt-Rung2-Omni-Sym2",
        env_cfg=matt_rung2_sym2_env_cfg(),
        play_env_cfg=matt_rung2_sym2_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("matt_rung2", max_iterations=2000),
        runner_cls=SymmetricRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Matt-Rung2-Omni-Sym3",
        env_cfg=matt_rung2_sym3_env_cfg(),
        play_env_cfg=matt_rung2_sym3_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("matt_rung2", max_iterations=2000),
        runner_cls=SymmetricRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Matt-Rung2-Omni-Sym4",
        env_cfg=matt_rung2_sym4_env_cfg(),
        play_env_cfg=matt_rung2_sym4_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("matt_rung2", max_iterations=2000),
        runner_cls=SymmetricRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Matt-Rung2-Omni-Sym5",
        env_cfg=matt_rung2_sym5_env_cfg(),
        play_env_cfg=matt_rung2_sym5_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("matt_rung2", max_iterations=1500),
        runner_cls=SymmetricRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Matt-Rung2-Flight",
        env_cfg=matt_rung2_flight_env_cfg(),
        play_env_cfg=matt_rung2_flight_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("matt_rung2", max_iterations=1000),
        runner_cls=SymmetricRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Matt-Rung2-Flight2",
        env_cfg=matt_rung2_flight2_env_cfg(),
        play_env_cfg=matt_rung2_flight2_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("matt_rung2", max_iterations=800),
        runner_cls=SymmetricRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Matt-Rung2-Flight3",
        env_cfg=matt_rung2_flight3_env_cfg(),
        play_env_cfg=matt_rung2_flight3_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("matt_rung2", max_iterations=800),
        runner_cls=SymmetricRunner,
    )

    # 2026-10-11 audit (opt-in variants; nothing above changed): Rung 0 with DESIGN §5's foot-slip / CoM terms, Rung 2
    # with the success-driven shove ramp + contact solref DR + privileged critic
    from .matt_env import matt_rung0_full_env_cfg, matt_rung2_ramp_env_cfg

    register_mjlab_task(task_id="PoOlympic-Matt-Rung0-Full", env_cfg=matt_rung0_full_env_cfg(),
                        play_env_cfg=matt_rung0_full_env_cfg(play=True),
                        rl_cfg=matt_ppo_cfg("matt_rung0", max_iterations=400), runner_cls=MjlabOnPolicyRunner)
    register_mjlab_task(task_id="PoOlympic-Matt-Rung2-Ramp", env_cfg=matt_rung2_ramp_env_cfg(),
                        play_env_cfg=matt_rung2_ramp_env_cfg(play=True),
                        rl_cfg=matt_ppo_cfg("matt_rung2", max_iterations=600), runner_cls=SymmetricRunner)
    _register_fast("PoOlympic-Matt-Rung0-Full", matt_rung0_full_env_cfg, "matt_rung0", 200, MjlabOnPolicyRunner)
    _register_fast("PoOlympic-Matt-Rung2-Ramp", matt_rung2_ramp_env_cfg, "matt_rung2", 300, SymmetricRunner)

if bodies.current().name == "matt":
    # Get-up for the roster MATT (2026-10-11: fall recovery + Event 27 need it; the 2026-09-29 Matt-Getup..5 tasks never
    # stood up). The ladder recipe that solved mattbio / GRANDMA / KIM, on MATT's own model and contract v3.
    from .getup_env import matt_getup_ladder_e_env_cfg, matt_getup_prone_env_cfg

    for _tid, _fn, _its in (("PoOlympic-Matt-Getup-Ladder", matt_getup_ladder_e_env_cfg, 1500),
                            ("PoOlympic-Matt-Getup-Prone", matt_getup_prone_env_cfg, 800)):
        register_mjlab_task(task_id=_tid, env_cfg=_fn(), play_env_cfg=_fn(play=True),
                            rl_cfg=matt_ppo_cfg("matt_getup", max_iterations=_its), runner_cls=MjlabOnPolicyRunner)

if bodies.current().name == "mattbio":
    # STAGED (not trained; needs the user's OK — DESIGN §2 body change): MATT with full self-collision + bio torque caps
    # (bodies.BIO_TORQUE_CAPS). Same contract, so every MATT checkpoint warm-starts directly. POOLYMPIC_BODY=mattbio.
    from .matt_env import matt_ppo_v5_cfg, matt_rung0_v5_env_cfg, matt_rung2_flight_v5_env_cfg, matt_rung2_v5_env_cfg
    from .getup_env import (matt_getup_ladder_b_env_cfg, matt_getup_ladder_c_env_cfg, matt_getup_ladder_d_env_cfg,
                            matt_getup_ladder_e_env_cfg, matt_getup_ladder_env_cfg, matt_getup_prone_env_cfg, matt_getup_rev_env_cfg)
    from .crawl_env import matt_crawl_v5_env_cfg
    from .stance_env import matt_stance_env_cfg, matt_stance_reach2_env_cfg, matt_stance_reach_env_cfg, matt_stance_v6_env_cfg, matt_stance_v7_env_cfg
    from .symmetry import SymmetricRunner

    for task_id, fn, exp, its, runner in (
            ("PoOlympic-MattBio-RungS-Reach", matt_stance_reach_env_cfg, "mattbio_stance", 1200, SymmetricRunner),
            ("PoOlympic-MattBio-RungS-Reach2", matt_stance_reach2_env_cfg, "mattbio_stance", 1200, SymmetricRunner),
            ("PoOlympic-MattBio-Rung0-Stand", matt_rung0_v5_env_cfg, "mattbio_rung0", 400, MjlabOnPolicyRunner),
            ("PoOlympic-MattBio-Rung2-Omni", matt_rung2_v5_env_cfg, "mattbio_rung2", 800, SymmetricRunner),
            ("PoOlympic-MattBio-RungS-Stance", matt_stance_env_cfg, "mattbio_stance", 1000, SymmetricRunner),
            ("PoOlympic-MattBio-RungS-Stance-v6", matt_stance_v6_env_cfg, "mattbio_stance", 1000, SymmetricRunner),
            ("PoOlympic-MattBio-RungS-Stance-v7", matt_stance_v7_env_cfg, "mattbio_stance", 1500, SymmetricRunner),
            ("PoOlympic-MattBio-RungS-Stance-v8", matt_stance_v7_env_cfg, "mattbio_stance", 1200, SymmetricRunner),
            ("PoOlympic-MattBio-Rung2-Flight", matt_rung2_flight_v5_env_cfg, "mattbio_rung2", 400, SymmetricRunner),
            ("PoOlympic-MattBio-Getup-Rev", matt_getup_rev_env_cfg, "mattbio_getup", 1500, MjlabOnPolicyRunner),
            ("PoOlympic-MattBio-Getup-Ladder", matt_getup_ladder_env_cfg, "mattbio_getup", 1500, MjlabOnPolicyRunner),
            ("PoOlympic-MattBio-Getup-Ladder-b", matt_getup_ladder_b_env_cfg, "mattbio_getup", 1500, MjlabOnPolicyRunner),
            ("PoOlympic-MattBio-Getup-Ladder-c", matt_getup_ladder_c_env_cfg, "mattbio_getup", 1500, MjlabOnPolicyRunner),
            ("PoOlympic-MattBio-Getup-Ladder-d", matt_getup_ladder_d_env_cfg, "mattbio_getup", 1500, MjlabOnPolicyRunner),
            ("PoOlympic-MattBio-Getup-Ladder-e", matt_getup_ladder_e_env_cfg, "mattbio_getup", 1500, MjlabOnPolicyRunner),
            ("PoOlympic-MattBio-Getup-Prone", matt_getup_prone_env_cfg, "mattbio_getup", 1000, MjlabOnPolicyRunner),
            ("PoOlympic-MattBio-Crawl", matt_crawl_v5_env_cfg, "mattbio_crawl", 600, MjlabOnPolicyRunner)):
        rl = matt_ppo_v5_cfg(exp, max_iterations=its)
        if task_id.endswith("Stance-v7"):
            rl.algorithm.entropy_coef = 0.0025   # rs_v3-v6 passed this on the CLI; the queue has no train-args field
        if task_id.endswith("Stance-v8") or task_id.endswith(("RungS-Reach", "RungS-Reach2")):
            # rs_v8 = rs_v7's env, entropy 0.0025 -> 0.001 (+ plain_init --max-std 0.3): the action std rose 0.44 -> 0.54
            # over rs_v6 + rs_v7 while track_ang fell 0.98 -> 0.71; on GRANDMA's Rung 2 this change stopped both.
            rl.algorithm.entropy_coef = 0.001
        register_mjlab_task(task_id=task_id, env_cfg=fn(), play_env_cfg=fn(play=True), rl_cfg=rl, runner_cls=runner)

    # 2026-10-12: per-event specialists (user decision 2026-10-01: one stance brain per event). rs_v9 (Stance-v8 on from
    # the deployed rs_v8 it 1199, rebuilt with tools/onnx_to_checkpoint.py) peaked at march 9/10 + torso 8/10 (it 450 /
    # 600) and then lost the march (0/10 at it 1050): the five skills share one network and trade against each other.
    # Same env, 65 % of the commands on ONE skill (locomotion 15 %, the other four 5 % each so they are not forgotten
    # outright), entropy 0.001. Warm start: rs_v9 it 600 (plain_init --max-std 0.3).
    def _specialist(mode: int):
        def make(play: bool = False):
            cfg = matt_stance_v7_env_cfg(play=play)
            probs = [0.15, 0.05, 0.05, 0.05, 0.05, 0.05]
            probs[mode] = 0.65
            cfg.commands["athlete"].mode_probs = tuple(probs)
            return cfg
        return make

    # 2026-10-13: Event 3 specialist on event-like squat reps (stance_env.matt_stance_squat_env_cfg)
    from .stance_env import matt_stance_squat_env_cfg
    _rq = matt_ppo_v5_cfg("mattbio_stance", max_iterations=600)
    _rq.algorithm.entropy_coef = 0.001
    register_mjlab_task(task_id="PoOlympic-MattBio-RungS-Squat", env_cfg=matt_stance_squat_env_cfg(),
                        play_env_cfg=matt_stance_squat_env_cfg(play=True), rl_cfg=_rq, runner_cls=SymmetricRunner)
    from .stance_env import matt_stance_squat2_env_cfg
    register_mjlab_task(task_id="PoOlympic-MattBio-RungS-Squat2", env_cfg=matt_stance_squat2_env_cfg(),
                        play_env_cfg=matt_stance_squat2_env_cfg(play=True), rl_cfg=_rq, runner_cls=SymmetricRunner)

    for task_id, mode in (("PoOlympic-MattBio-RungS-March", 3), ("PoOlympic-MattBio-RungS-Torso", 4)):
        fn = _specialist(mode)
        rl = matt_ppo_v5_cfg("mattbio_stance", max_iterations=600)
        rl.algorithm.entropy_coef = 0.001
        register_mjlab_task(task_id=task_id, env_cfg=fn(), play_env_cfg=fn(play=True), rl_cfg=rl, runner_cls=SymmetricRunner)

if bodies.current().name == "kim":
    # KIM (2026-10-10): the GRANDMA chain on her body — Rung 0 / 1, Rung 2 (base -> cap -> bands), crawl, get-up ladder
    # (supine, then face down: prone = press-up recipe, roll = the roll ladder if her arms cannot press her up)
    from .crawl_env import grandma_crawl2_env_cfg, grandma_crawl3_env_cfg
    from .getup_env import body_getup_ladder_env_cfg, body_getup_prone_env_cfg, body_getup_roll_env_cfg
    from .kim_env import (kim_rung0_env_cfg, kim_rung0_tight_env_cfg, kim_rung1_env_cfg, kim_rung2_bands2_env_cfg,
                          kim_rung2_bands3_env_cfg, kim_rung2_bands_env_cfg, kim_rung2_base_env_cfg, kim_rung2_cap_env_cfg)
    from .matt_env import matt_ppo_cfg
    from .symmetry import SymmetricRunner

    for _tid, _fn, _exp, _its, _runner, _ent in (
            ("PoOlympic-Kim-Rung0-Stand", kim_rung0_env_cfg, "kim_rung0", 1500, MjlabOnPolicyRunner, None),
            ("PoOlympic-Kim-Rung1-Run", kim_rung1_env_cfg, "kim_rung1", 3000, MjlabOnPolicyRunner, None),
            ("PoOlympic-Kim-Rung2-Omni-Base", kim_rung2_base_env_cfg, "kim_rung2", 2000, MjlabOnPolicyRunner, None),
            ("PoOlympic-Kim-Rung2-Omni-Cap", kim_rung2_cap_env_cfg, "kim_rung2", 1200, MjlabOnPolicyRunner, 0.001),
            ("PoOlympic-Kim-Rung2-Omni-Bands", kim_rung2_bands_env_cfg, "kim_rung2", 1200, SymmetricRunner, 0.001),
            ("PoOlympic-Kim-Rung2-Omni-Bands2", kim_rung2_bands2_env_cfg, "kim_rung2", 1000, SymmetricRunner, 0.001),
            ("PoOlympic-Kim-Rung2-Omni-Bands3", kim_rung2_bands3_env_cfg, "kim_rung2", 600, SymmetricRunner, 0.001),
            ("PoOlympic-Kim-Rung0-Tight", kim_rung0_tight_env_cfg, "kim_rung0", 500, MjlabOnPolicyRunner, 0.001),
            ("PoOlympic-Kim-Crawl2", grandma_crawl2_env_cfg, "kim_crawl", 1500, MjlabOnPolicyRunner, None),
            ("PoOlympic-Kim-Crawl3", grandma_crawl3_env_cfg, "kim_crawl", 1000, MjlabOnPolicyRunner, None),
            ("PoOlympic-Kim-Getup", body_getup_ladder_env_cfg, "kim_getup", 2000, MjlabOnPolicyRunner, None),
            ("PoOlympic-Kim-Getup-Prone", body_getup_prone_env_cfg, "kim_getup", 600, MjlabOnPolicyRunner, None),
            ("PoOlympic-Kim-Getup-Roll", body_getup_roll_env_cfg, "kim_getup", 800, MjlabOnPolicyRunner, None)):
        _rl = matt_ppo_cfg(_exp, max_iterations=_its)
        if _ent is not None:
            _rl.algorithm.entropy_coef = _ent      # GRANDMA's Rung 2 finding (g2_v2b): 0.005 out-pulls the task
        register_mjlab_task(task_id=_tid, env_cfg=_fn(), play_env_cfg=_fn(play=True), rl_cfg=_rl, runner_cls=_runner)

    # MATT's extra skills on her body (2026-10-11): Rung S stance skills (squat / flamingo / march / torso / reach: the
    # rs_v7 recipe on her Rung 2 base, contract v4, 8192 envs) and the Event 13 flight brain
    from .kim_env import kim_rung2_flight_env_cfg
    from .matt_env import matt_ppo_v5_cfg
    from .stance_env import matt_stance_v7_env_cfg

    register_mjlab_task(task_id="PoOlympic-Kim-RungS-Stance", env_cfg=matt_stance_v7_env_cfg(),
                        play_env_cfg=matt_stance_v7_env_cfg(play=True),
                        rl_cfg=_low_entropy(matt_ppo_v5_cfg("kim_stance", max_iterations=1500)), runner_cls=SymmetricRunner)
    register_mjlab_task(task_id="PoOlympic-Kim-Rung2-Flight", env_cfg=kim_rung2_flight_env_cfg(),
                        play_env_cfg=kim_rung2_flight_env_cfg(play=True),
                        rl_cfg=_low_entropy(matt_ppo_cfg("kim_rung2", max_iterations=400)), runner_cls=SymmetricRunner)

    # 2026-10-11 audit (opt-in variants; nothing above changed). Rung2-Bio = the contract v5 proposal (fatigue block)
    from .kim_env import kim_rung0_full_env_cfg, kim_rung2_bio_env_cfg, kim_rung2_ramp_env_cfg

    register_mjlab_task(task_id="PoOlympic-Kim-Rung0-Full", env_cfg=kim_rung0_full_env_cfg(),
                        play_env_cfg=kim_rung0_full_env_cfg(play=True),
                        rl_cfg=_low_entropy(matt_ppo_cfg("kim_rung0", max_iterations=400)), runner_cls=MjlabOnPolicyRunner)
    register_mjlab_task(task_id="PoOlympic-Kim-Rung2-Ramp", env_cfg=kim_rung2_ramp_env_cfg(),
                        play_env_cfg=kim_rung2_ramp_env_cfg(play=True),
                        rl_cfg=_low_entropy(matt_ppo_cfg("kim_rung2", max_iterations=600)), runner_cls=SymmetricRunner)
    register_mjlab_task(task_id="PoOlympic-Kim-Rung2-Bio", env_cfg=kim_rung2_bio_env_cfg(),
                        play_env_cfg=kim_rung2_bio_env_cfg(play=True),
                        rl_cfg=_low_entropy(matt_ppo_cfg("kim_rung2", max_iterations=800)), runner_cls=SymmetricRunner)
    _register_fast("PoOlympic-Kim-Rung0-Full", kim_rung0_full_env_cfg, "kim_rung0", 200, MjlabOnPolicyRunner, 0.001)
    _register_fast("PoOlympic-Kim-Rung2-Ramp", kim_rung2_ramp_env_cfg, "kim_rung2", 300, SymmetricRunner, 0.001)
    _register_fast("PoOlympic-Kim-Rung2-Bio", kim_rung2_bio_env_cfg, "kim_rung2", 400, SymmetricRunner, 0.001)

if bodies.current().name == "grandma":
    from .grandma_env import (grandma_rung0_env_cfg, grandma_rung1_env_cfg, grandma_rung2_base_env_cfg,
                              grandma_rung2_cap28_env_cfg, grandma_rung2_cap_env_cfg, grandma_rung2_env_cfg,
                              grandma_rung2_bands_env_cfg, grandma_rung2_sharp_env_cfg)
    from .matt_env import matt_ppo_cfg
    from .symmetry import SymmetricRunner

    for _tid, _fn, _exp, _its, _runner in (
            ("PoOlympic-Grandma-Rung1-Run", grandma_rung1_env_cfg, "grandma_rung1", 3000, MjlabOnPolicyRunner),
            ("PoOlympic-Grandma-Rung2-Omni-Base", grandma_rung2_base_env_cfg, "grandma_rung2", 2000, MjlabOnPolicyRunner),
            ("PoOlympic-Grandma-Rung2-Omni-Cap", grandma_rung2_cap_env_cfg, "grandma_rung2", 1200, MjlabOnPolicyRunner),
            ("PoOlympic-Grandma-Rung2-Omni-Cap28", grandma_rung2_cap28_env_cfg, "grandma_rung2", 1200, MjlabOnPolicyRunner),
            ("PoOlympic-Grandma-Rung2-Omni-Cap28-Sharp", grandma_rung2_sharp_env_cfg, "grandma_rung2", 1200, MjlabOnPolicyRunner),
            ("PoOlympic-Grandma-Rung2-Omni-Cap28-Bands", grandma_rung2_bands_env_cfg, "grandma_rung2", 1200, SymmetricRunner),
            ("PoOlympic-Grandma-Rung2-Omni", grandma_rung2_env_cfg, "grandma_rung2", 2000, SymmetricRunner)):
        _rl = matt_ppo_cfg(_exp, max_iterations=_its)
        if _tid in ("PoOlympic-Grandma-Rung2-Omni", "PoOlympic-Grandma-Rung2-Omni-Cap", "PoOlympic-Grandma-Rung2-Omni-Cap28",
                    "PoOlympic-Grandma-Rung2-Omni-Cap28-Sharp", "PoOlympic-Grandma-Rung2-Omni-Cap28-Bands"):
            # g2_v2b (entropy 0.005): action std 0.51 -> 0.57 and mean reward 112 -> 102 in 400 its, speed at a 2.5 m/s
            # command 2.36 -> 1.25: the entropy bonus out-pulled the task. g2_v3 onwards: 0.001.
            _rl.algorithm.entropy_coef = 0.001
        register_mjlab_task(task_id=_tid, env_cfg=_fn(), play_env_cfg=_fn(play=True), rl_cfg=_rl, runner_cls=_runner)

    from .crawl_env import grandma_crawl2_env_cfg, grandma_crawl3_env_cfg, grandma_crawl_env_cfg
    from .getup_env import body_getup_ladder_env_cfg, body_getup_prone_env_cfg

    from .getup_env import body_getup_roll_env_cfg

    register_mjlab_task(
        task_id="PoOlympic-Grandma-Getup-Roll",
        env_cfg=body_getup_roll_env_cfg(),
        play_env_cfg=body_getup_roll_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("grandma_getup", max_iterations=800),
        runner_cls=MjlabOnPolicyRunner,
    )

    from .getup_env import body_getup_roll2_env_cfg

    register_mjlab_task(
        task_id="PoOlympic-Grandma-Getup-Roll2",
        env_cfg=body_getup_roll2_env_cfg(),
        play_env_cfg=body_getup_roll2_env_cfg(play=True),
        # entropy 0.005 -> 0.001: ggetup_roll_v1's action std rose 0.44 -> 0.93 in 620 its while the 120 deg stage sat at 45-48 %
        rl_cfg=_low_entropy(matt_ppo_cfg("grandma_getup", max_iterations=1000)),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Grandma-Getup-Prone",
        env_cfg=body_getup_prone_env_cfg(),
        play_env_cfg=body_getup_prone_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("grandma_getup", max_iterations=600),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Grandma-Crawl3",
        env_cfg=grandma_crawl3_env_cfg(),
        play_env_cfg=grandma_crawl3_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("grandma_crawl", max_iterations=1000),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Grandma-Crawl2",
        env_cfg=grandma_crawl2_env_cfg(),
        play_env_cfg=grandma_crawl2_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("grandma_crawl", max_iterations=1500),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Grandma-Crawl",
        env_cfg=grandma_crawl_env_cfg(),
        play_env_cfg=grandma_crawl_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("grandma_crawl", max_iterations=1500),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Grandma-Getup",
        env_cfg=body_getup_ladder_env_cfg(),
        play_env_cfg=body_getup_ladder_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("grandma_getup", max_iterations=2000),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Grandma-Rung0-Stand",
        env_cfg=grandma_rung0_env_cfg(),
        play_env_cfg=grandma_rung0_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("grandma_rung0", max_iterations=1500),
        runner_cls=MjlabOnPolicyRunner,
    )

    # 2026-10-11 audit (opt-in variants; nothing above changed)
    from .grandma_env import grandma_rung0_full_env_cfg, grandma_rung2_ramp_env_cfg

    register_mjlab_task(task_id="PoOlympic-Grandma-Rung0-Full", env_cfg=grandma_rung0_full_env_cfg(),
                        play_env_cfg=grandma_rung0_full_env_cfg(play=True),
                        rl_cfg=matt_ppo_cfg("grandma_rung0", max_iterations=400), runner_cls=MjlabOnPolicyRunner)
    register_mjlab_task(task_id="PoOlympic-Grandma-Rung2-Ramp", env_cfg=grandma_rung2_ramp_env_cfg(),
                        play_env_cfg=grandma_rung2_ramp_env_cfg(play=True),
                        rl_cfg=_low_entropy(matt_ppo_cfg("grandma_rung2", max_iterations=600)), runner_cls=SymmetricRunner)
    _register_fast("PoOlympic-Grandma-Rung0-Full", grandma_rung0_full_env_cfg, "grandma_rung0", 200, MjlabOnPolicyRunner)
    _register_fast("PoOlympic-Grandma-Rung2-Ramp", grandma_rung2_ramp_env_cfg, "grandma_rung2", 300, SymmetricRunner, 0.001)

if bodies.current().name == "zombie":
    from .matt_env import matt_ppo_cfg
    from .symmetry import SymmetricRunner
    from .zombie_env import (zombie_rung0_env_cfg, zombie_rung1_env_cfg, zombie_rung2_base_env_cfg, zombie_rung2_env_cfg,
                             zombie_rung2_sym3_env_cfg, zombie_rung2_sym3yaw_env_cfg,
                             zombie_rung2_yawfilt_cap3_env_cfg, zombie_rung2_yawfilt_cap3_lin3_env_cfg, zombie_rung2_yawfilt_env_cfg, zombie_rung2_wobble_env_cfg,
                             zombie_rung2_wobble_sharp_env_cfg, zombie_rung2_wobble_sprint_env_cfg,
                             zombie_rung2_yawgrad_env_cfg)

    # 2026-10-12: GRANDMA's / KIM's Bands recipe on the zombie (z2_v16; see zombie_rung2_bands_env_cfg)
    from .zombie_env import zombie_rung2_bands_env_cfg
    _zb = matt_ppo_cfg("zombie_rung2", max_iterations=1200)
    _zb.algorithm.entropy_coef = 0.001
    register_mjlab_task(task_id="PoOlympic-Zombie-Rung2-Omni-Bands", env_cfg=zombie_rung2_bands_env_cfg(),
                        play_env_cfg=zombie_rung2_bands_env_cfg(play=True), rl_cfg=_zb, runner_cls=SymmetricRunner)
    from .zombie_env import zombie_rung2_bands2_env_cfg, zombie_rung2_polish_env_cfg
    register_mjlab_task(task_id="PoOlympic-Zombie-Rung2-Omni-Bands2", env_cfg=zombie_rung2_bands2_env_cfg(),
                        play_env_cfg=zombie_rung2_bands2_env_cfg(play=True), rl_cfg=_zb, runner_cls=SymmetricRunner)
    from .zombie_env import zombie_rung2_capbands_env_cfg
    register_mjlab_task(task_id="PoOlympic-Zombie-Rung2-Omni-CapBands", env_cfg=zombie_rung2_capbands_env_cfg(),
                        play_env_cfg=zombie_rung2_capbands_env_cfg(play=True), rl_cfg=_zb, runner_cls=SymmetricRunner)
    _zp = matt_ppo_cfg("zombie_rung2", max_iterations=400)
    _zp.algorithm.entropy_coef, _zp.algorithm.desired_kl, _zp.save_interval = 0.001, 0.004, 50
    register_mjlab_task(task_id="PoOlympic-Zombie-Rung2-Omni-Wobble-Polish", env_cfg=zombie_rung2_polish_env_cfg(),
                        play_env_cfg=zombie_rung2_polish_env_cfg(play=True), rl_cfg=_zp, runner_cls=SymmetricRunner)

    # The zombie's September tasks below build ONE environment unless the CLI passes --env.scene.num-envs, and the
    # queue has no train-args field: z2_v15, z1_v2 and z0_v2 (2026-10-12) each trained a single env through it.
    # "-4096" = the same task with the env count in the config, for the queue.
    def _envs4096(fn):
        def make(play: bool = False):
            cfg = fn(play=play)
            cfg.scene.num_envs = 4096
            return cfg
        return make

    for _tid, _fn, _exp, _its, _runner in (
            ("PoOlympic-Zombie-Rung0-Stand-4096", zombie_rung0_env_cfg, "zombie_rung0", 1500, MjlabOnPolicyRunner),
            ("PoOlympic-Zombie-Rung1-Run-4096", zombie_rung1_env_cfg, "zombie_rung1", 3000, MjlabOnPolicyRunner),
            ("PoOlympic-Zombie-Rung2-Omni-Base-4096", zombie_rung2_base_env_cfg, "zombie_rung2", 2000, MjlabOnPolicyRunner),
            ("PoOlympic-Zombie-Rung2-Omni-Wobble-4096", zombie_rung2_wobble_env_cfg, "zombie_rung2", 1000, SymmetricRunner)):
        _fn = _envs4096(_fn)
        register_mjlab_task(task_id=_tid, env_cfg=_fn(), play_env_cfg=_fn(play=True),
                            rl_cfg=matt_ppo_cfg(_exp, max_iterations=_its), runner_cls=_runner)

    register_mjlab_task(
        task_id="PoOlympic-Zombie-Rung0-Stand",
        env_cfg=zombie_rung0_env_cfg(),
        play_env_cfg=zombie_rung0_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("zombie_rung0", max_iterations=1500),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Zombie-Rung1-Run",
        env_cfg=zombie_rung1_env_cfg(),
        play_env_cfg=zombie_rung1_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("zombie_rung1", max_iterations=3000),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Zombie-Rung2-Omni-Base",
        env_cfg=zombie_rung2_base_env_cfg(),
        play_env_cfg=zombie_rung2_base_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("zombie_rung2", max_iterations=2000),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Zombie-Rung2-Omni",
        env_cfg=zombie_rung2_env_cfg(),
        play_env_cfg=zombie_rung2_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("zombie_rung2", max_iterations=3000),
        runner_cls=SymmetricRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Zombie-Rung2-Omni-Sym3",
        env_cfg=zombie_rung2_sym3_env_cfg(),
        play_env_cfg=zombie_rung2_sym3_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("zombie_rung2", max_iterations=2000),
        runner_cls=SymmetricRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Zombie-Rung2-Omni-Sym3Yaw",
        env_cfg=zombie_rung2_sym3yaw_env_cfg(),
        play_env_cfg=zombie_rung2_sym3yaw_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("zombie_rung2", max_iterations=1500),
        runner_cls=SymmetricRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Zombie-Rung2-Omni-YawGrad",
        env_cfg=zombie_rung2_yawgrad_env_cfg(),
        play_env_cfg=zombie_rung2_yawgrad_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("zombie_rung2", max_iterations=1500),
        runner_cls=SymmetricRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Zombie-Rung2-Omni-YawFilt",
        env_cfg=zombie_rung2_yawfilt_env_cfg(),
        play_env_cfg=zombie_rung2_yawfilt_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("zombie_rung2", max_iterations=1500),
        runner_cls=SymmetricRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Zombie-Rung2-Omni-YawFiltCap3",
        env_cfg=zombie_rung2_yawfilt_cap3_env_cfg(),
        play_env_cfg=zombie_rung2_yawfilt_cap3_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("zombie_rung2", max_iterations=1500),
        runner_cls=SymmetricRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Zombie-Rung2-Omni-YawFiltCap3Lin3",
        env_cfg=zombie_rung2_yawfilt_cap3_lin3_env_cfg(),
        play_env_cfg=zombie_rung2_yawfilt_cap3_lin3_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("zombie_rung2", max_iterations=1000),
        runner_cls=SymmetricRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Zombie-Rung2-Omni-Wobble",
        env_cfg=zombie_rung2_wobble_env_cfg(),
        play_env_cfg=zombie_rung2_wobble_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("zombie_rung2", max_iterations=1000),
        runner_cls=SymmetricRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Zombie-Rung2-Omni-WobbleSharp",
        env_cfg=zombie_rung2_wobble_sharp_env_cfg(),
        play_env_cfg=zombie_rung2_wobble_sharp_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("zombie_rung2", max_iterations=600),
        runner_cls=SymmetricRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Zombie-Rung2-Omni-WobbleSprint",
        env_cfg=zombie_rung2_wobble_sprint_env_cfg(),
        play_env_cfg=zombie_rung2_wobble_sprint_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("zombie_rung2", max_iterations=600),
        runner_cls=SymmetricRunner,
    )

    from .crawl_env import zombie_crawl2_env_cfg, zombie_crawl3_env_cfg, zombie_crawl4_env_cfg, zombie_crawl5_env_cfg, zombie_crawl_env_cfg
    from .getup_env import body_getup_ladder_env_cfg, body_getup_prone_env_cfg

    from .getup_env import zombie_getup2_env_cfg

    register_mjlab_task(
        task_id="PoOlympic-Zombie-Getup2",
        env_cfg=zombie_getup2_env_cfg(),
        play_env_cfg=zombie_getup2_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("zombie_getup", max_iterations=1500),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Zombie-Getup-Prone",
        env_cfg=body_getup_prone_env_cfg(),
        play_env_cfg=body_getup_prone_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("zombie_getup", max_iterations=600),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Zombie-Getup",
        env_cfg=body_getup_ladder_env_cfg(),
        play_env_cfg=body_getup_ladder_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("zombie_getup", max_iterations=2000),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Zombie-Crawl5",
        env_cfg=zombie_crawl5_env_cfg(),
        play_env_cfg=zombie_crawl5_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("zombie_crawl", max_iterations=1500),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Zombie-Crawl4",
        env_cfg=zombie_crawl4_env_cfg(),
        play_env_cfg=zombie_crawl4_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("zombie_crawl", max_iterations=2500),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Zombie-Crawl3",
        env_cfg=zombie_crawl3_env_cfg(),
        play_env_cfg=zombie_crawl3_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("zombie_crawl", max_iterations=2500),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Zombie-Crawl2",
        env_cfg=zombie_crawl2_env_cfg(),
        play_env_cfg=zombie_crawl2_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("zombie_crawl", max_iterations=2500),
        runner_cls=MjlabOnPolicyRunner,
    )

    register_mjlab_task(
        task_id="PoOlympic-Zombie-Crawl",
        env_cfg=zombie_crawl_env_cfg(),
        play_env_cfg=zombie_crawl_env_cfg(play=True),
        rl_cfg=matt_ppo_cfg("zombie_crawl", max_iterations=2500),
        runner_cls=MjlabOnPolicyRunner,
    )

    # 2026-10-11 audit (opt-in variants; nothing above changed)
    from .zombie_env import zombie_rung0_full_env_cfg, zombie_rung2_ramp_env_cfg

    register_mjlab_task(task_id="PoOlympic-Zombie-Rung0-Full", env_cfg=zombie_rung0_full_env_cfg(),
                        play_env_cfg=zombie_rung0_full_env_cfg(play=True),
                        rl_cfg=matt_ppo_cfg("zombie_rung0", max_iterations=400), runner_cls=MjlabOnPolicyRunner)
    register_mjlab_task(task_id="PoOlympic-Zombie-Rung2-Ramp", env_cfg=zombie_rung2_ramp_env_cfg(),
                        play_env_cfg=zombie_rung2_ramp_env_cfg(play=True),
                        rl_cfg=matt_ppo_cfg("zombie_rung2", max_iterations=600), runner_cls=SymmetricRunner)
    _register_fast("PoOlympic-Zombie-Rung0-Full", zombie_rung0_full_env_cfg, "zombie_rung0", 200, MjlabOnPolicyRunner)
    _register_fast("PoOlympic-Zombie-Rung2-Ramp", zombie_rung2_ramp_env_cfg, "zombie_rung2", 300, SymmetricRunner)


if bodies.current().name in ("matt", "zombie", "grandma", "kim"):
    # 2026-10-12 crowd fine-tune: each roster body's DEPLOYED Rung 2 recipe + the bumps of a race in a pack
    # (matt_env.add_crowd_bumps). Small steps as the zombie polish (KL 0.004, entropy 0.001), a checkpoint every 50.
    def _crowd():
        from .matt_env import add_crowd_bumps, matt_ppo_cfg
        from .symmetry import SymmetricRunner
        body = bodies.current()
        if body.name == "matt":
            from .matt_env import matt_rung2_sym5_env_cfg as base
        elif body.name == "zombie":
            from .zombie_env import zombie_rung2_polish_env_cfg as base
        elif body.name == "grandma":
            from .grandma_env import grandma_rung2_bands_env_cfg as base
        else:
            from .kim_env import kim_rung2_bands2_env_cfg as base
        rl = matt_ppo_cfg(f"{body.name}_rung2", max_iterations=400)
        rl.algorithm.entropy_coef, rl.algorithm.desired_kl, rl.save_interval = 0.001, 0.004, 50
        cfg = add_crowd_bumps(base(), body.mass_scale)
        cfg.scene.num_envs = 4096              # MATT's September recipe builds one env unless the CLI says otherwise
        # Crowd2 (2026-10-13): r2c_v1 made MATT robust (sprint falls 3.2 -> 0.7 per heat) but ~4 % slower, and KIM's /
        # GRANDMA's G1 fell from 10/10 to 3-4/10 after 100 its: a bump on one of three sites every ~1 s is far more than
        # a race gives (a runner is bumped a few times per heat). Same recipe, 2/3 of the force, each site every 4-10 s.
        cfg2 = add_crowd_bumps(base(), body.mass_scale, force=200.0, cooldown_s=(4.0, 10.0))
        cfg2.scene.num_envs = 4096
        register_mjlab_task(task_id=f"PoOlympic-{body.name.capitalize()}-Rung2-Crowd2", env_cfg=cfg2,
                            play_env_cfg=base(play=True), rl_cfg=rl, runner_cls=SymmetricRunner)
        register_mjlab_task(task_id=f"PoOlympic-{body.name.capitalize()}-Rung2-Crowd", env_cfg=cfg,
                            play_env_cfg=base(play=True), rl_cfg=rl, runner_cls=SymmetricRunner)
    _crowd()

    # 2026-10-13 Rung 4 slope (event 14 The Alpine Ramp, 15 deg): each roster body's deployed Rung 2 recipe on a tilted-
    # gravity ramp (matt_env.add_slope). Two steps from the walking brain: Slope8 (8 deg), then Slope15 from it.
    def _slope():
        from .matt_env import add_slope, matt_ppo_cfg
        from .symmetry import SymmetricRunner
        body = bodies.current()
        if body.name == "matt":
            from .matt_env import matt_rung2_sym5_env_cfg as base
        elif body.name == "zombie":
            from .zombie_env import zombie_rung2_polish_env_cfg as base
        elif body.name == "grandma":
            from .grandma_env import grandma_rung2_bands_env_cfg as base
        else:
            from .kim_env import kim_rung2_bands2_env_cfg as base
        for deg, its in ((8, 400), (15, 1000)):
            rl = matt_ppo_cfg(f"{body.name}_slope", max_iterations=its)
            rl.algorithm.entropy_coef, rl.save_interval = 0.001, 100
            cfg = add_slope(base(), deg)
            cfg.scene.num_envs = 4096
            register_mjlab_task(task_id=f"PoOlympic-{body.name.capitalize()}-Rung4-Slope{deg}", env_cfg=cfg,
                                play_env_cfg=add_slope(base(play=True), deg), rl_cfg=rl, runner_cls=SymmetricRunner)
        # 2026-10-14 climb ladder (KIM's Slope15 learned to stand still): 11.5 deg between, climb commands
        for name, deg, its, top in (("Slope11c", 11.5, 400, 0.3), ("Slope15c", 15, 800, 0.3), ("Slope15s", 15, 800, 0.15)):
            rl = matt_ppo_cfg(f"{body.name}_slope", max_iterations=its)
            rl.algorithm.entropy_coef, rl.save_interval = 0.001, 100
            cfg = add_slope(base(), deg, climb=True, top=top)
            cfg.scene.num_envs = 4096
            register_mjlab_task(task_id=f"PoOlympic-{body.name.capitalize()}-Rung4-{name}", env_cfg=cfg,
                                play_env_cfg=add_slope(base(play=True), deg, climb=True, top=top), rl_cfg=rl, runner_cls=SymmetricRunner)
    _slope()

if bodies.current().name in ("matt", "grandma"):
    # 2026-10-13 crawl in a pack. Events 8 / 23 (12 heats each, mixed lineups): MATT does not finish 11 of 72 crawls and
    # GRANDMA 3 of 20 trench crawls — they tumble against the neighbour (lanes 1.1 m) and do not come back; with
    # athlete-athlete contact off everyone finishes. The crawl recipes have no disturbance at all. The deployed Crawl3
    # recipe + the pack bumps at half the running force (crawlers meet at ~1 m/s), small steps, 4096 envs.
    def _crawl_crowd():
        from .matt_env import add_crowd_bumps, matt_ppo_cfg
        body = bodies.current()
        if body.name == "matt":
            from .crawl_env import matt_crawl3_env_cfg as base
        else:
            from .crawl_env import grandma_crawl3_env_cfg as base
        cfg = add_crowd_bumps(base(), body.mass_scale, force=150.0, play=False)
        cfg.scene.num_envs = 4096
        rl = matt_ppo_cfg(f"{body.name}_crawl", max_iterations=400)
        rl.algorithm.entropy_coef, rl.algorithm.desired_kl, rl.save_interval = 0.001, 0.004, 50
        register_mjlab_task(task_id=f"PoOlympic-{body.name.capitalize()}-Crawl-Crowd", env_cfg=cfg,
                            play_env_cfg=base(play=True), rl_cfg=rl, runner_cls=MjlabOnPolicyRunner)
    _crawl_crowd()

if bodies.current().name in ("matt", "grandma", "kim"):
    # 2026-10-13 straight crawl. The bump fine-tune above changed nothing (10 heats x 2 lineups x 2 events: 7.1 -> 7.2
    # of 8 finish): the tangles are not bumps. MATT, GRANDMA and KIM crawl SIDEWAYS — the spine 100 / 65 / 100 deg off
    # the direction of travel (mdp.crawl_square) — so each fills 1.5 m of a 1.1 m lane and swings 0.8-1.7 m into the
    # neighbours when it sets off. Crawl3 + a reward for the spine along the travel direction (w 2), 4096 envs.
    def _crawl_square():
        from mjlab.managers.reward_manager import RewardTermCfg
        from . import mdp
        from .matt_env import matt_ppo_cfg
        body = bodies.current()
        if body.name == "matt":
            from .crawl_env import matt_crawl3_env_cfg as base
        else:
            from .crawl_env import grandma_crawl3_env_cfg as base      # KIM trains GRANDMA's crawl recipe on her body

        def make(play: bool = False):
            cfg = base(play=play)
            cfg.rewards["crawl_square"] = RewardTermCfg(func=mdp.crawl_square, weight=2.0)
            cfg.scene.num_envs = 4096 if not play else cfg.scene.num_envs
            return cfg
        register_mjlab_task(task_id=f"PoOlympic-{body.name.capitalize()}-Crawl-Square", env_cfg=make(),
                            play_env_cfg=make(play=True), rl_cfg=matt_ppo_cfg(f"{body.name}_crawl", max_iterations=800),
                            runner_cls=MjlabOnPolicyRunner)

        # Square2: started from the zombie's straight crawl (cross-body), MATT's mcs_v2 crawls with the spine 19 deg off
        # the travel direction (own-brain starts stayed at 70-80 deg) and finishes 30 m in 26.5 s — but it no longer
        # steers: a +-0.4 rad/s command turns it +-0.03 rad/s (deployed MATT +0.25 / -0.37, zombie +-0.46), lane error
        # 10 m. The yaw-rate reward is 1 of ~12 reward points with a 0.5 rad/s kernel: here weight 3, kernel halved.
        def make2(play: bool = False):
            cfg = make(play)
            cfg.rewards["track_yaw"].weight = 3.0
            cfg.rewards["track_yaw"].params["std"] *= 0.5
            return cfg
        register_mjlab_task(task_id=f"PoOlympic-{body.name.capitalize()}-Crawl-Square2", env_cfg=make2(),
                            play_env_cfg=make2(play=True), rl_cfg=matt_ppo_cfg(f"{body.name}_crawl", max_iterations=1500),
                            runner_cls=MjlabOnPolicyRunner)
    _crawl_square()

# ---- registry hygiene -------------------------------------------------------------------------------------------------
# Task IDs whose recipe a later ID of the same body and skill replaced, or that were tried and dropped. They stay
# registered (old runs, the queue and the log refer to them) and build exactly what they always built; do not start new
# runs on them. Not listed = current: the recipe behind a deployed brain, a stage of a warm-start chain (Rung 0 -> 1 ->
# Rung2-Omni-Base ...), or still open work. `uv run python tools/dump_task_cfgs.py --list` prints both groups.
SUPERSEDED = frozenset("PoOlympic-" + t for t in (
    # MATT: get-up v1-v4 (-> Getup5, then the mattbio ladder), crawl v1-v2 (-> Crawl3 = crawl_matt), Rung 2 r2_v3-v7
    # (-> Sym5 = rung2), pedestal v1 (-> Pedestal2), flight v1-v2 (-> Flight3 = r2f_v3)
    "Matt-Getup", "Matt-Getup2", "Matt-Getup3", "Matt-Getup4", "Matt-Crawl", "Matt-Crawl2",
    "Matt-Rung2-Omni-Sym", "Matt-Rung2-Omni-Sym2", "Matt-Rung2-Omni-Sym3", "Matt-Rung2-Omni-Sym4",
    "Matt-Pedestal", "Matt-Rung2-Flight", "Matt-Rung2-Flight2",
    # mattbio: Rung S rs_v5-v7 (-> Stance-v8 = rs_v7's env at entropy 0.001, by design), reach v1 (-> Reach2), get-up
    # reverse curriculum and ladder v2-v2d (-> Getup-Ladder-e)
    "MattBio-RungS-Stance", "MattBio-RungS-Stance-v6", "MattBio-RungS-Stance-v7", "MattBio-RungS-Reach",
    "MattBio-Getup-Rev", "MattBio-Getup-Ladder", "MattBio-Getup-Ladder-b", "MattBio-Getup-Ladder-c", "MattBio-Getup-Ladder-d",
    # zombie: Rung 2 z2_v1-v10 (-> Wobble = z2_v11 = zombie_rung2), z2_v13 / v14 (no gain), crawl v1-v2
    "Zombie-Rung2-Omni", "Zombie-Rung2-Omni-Sym3", "Zombie-Rung2-Omni-Sym3Yaw", "Zombie-Rung2-Omni-YawGrad",
    "Zombie-Rung2-Omni-YawFilt", "Zombie-Rung2-Omni-YawFiltCap3", "Zombie-Rung2-Omni-YawFiltCap3Lin3",
    "Zombie-Rung2-Omni-WobbleSharp", "Zombie-Rung2-Omni-WobbleSprint", "Zombie-Crawl", "Zombie-Crawl2",
    # GRANDMA: Rung 2 g2_v2b-v6 (-> Cap28-Bands = g2_v7 / v8 = grandma_rung2), crawl v1, the press-up get-up (-> Getup-Roll)
    "Grandma-Rung2-Omni", "Grandma-Rung2-Omni-Cap", "Grandma-Rung2-Omni-Cap28", "Grandma-Rung2-Omni-Cap28-Sharp",
    "Grandma-Crawl", "Grandma-Getup-Prone",
    # KIM: Rung 2 Cap (skipped), Bands (-> Bands2 = k2_v3 = kim_rung2), Bands3 (k2_v4: no gain)
    "Kim-Rung2-Omni-Cap", "Kim-Rung2-Omni-Bands", "Kim-Rung2-Omni-Bands3",
))
