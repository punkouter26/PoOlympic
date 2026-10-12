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

if bodies.current().name == "zombie":
    from .matt_env import matt_ppo_cfg
    from .symmetry import SymmetricRunner
    from .zombie_env import (zombie_rung0_env_cfg, zombie_rung1_env_cfg, zombie_rung2_base_env_cfg, zombie_rung2_env_cfg,
                             zombie_rung2_sym3_env_cfg, zombie_rung2_sym3yaw_env_cfg,
                             zombie_rung2_yawfilt_cap3_env_cfg, zombie_rung2_yawfilt_cap3_lin3_env_cfg, zombie_rung2_yawfilt_env_cfg, zombie_rung2_wobble_env_cfg,
                             zombie_rung2_wobble_sharp_env_cfg, zombie_rung2_wobble_sprint_env_cfg,
                             zombie_rung2_yawgrad_env_cfg)

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
