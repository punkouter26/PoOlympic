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
    from .stance_env import matt_stance_env_cfg

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

if bodies.current().name == "mattbio":
    # STAGED (not trained; needs the user's OK — DESIGN §2 body change): MATT with full self-collision + bio torque caps
    # (bodies.BIO_TORQUE_CAPS). Same contract, so every MATT checkpoint warm-starts directly. POOLYMPIC_BODY=mattbio.
    from .matt_env import matt_ppo_v5_cfg, matt_rung0_v5_env_cfg, matt_rung2_flight_v5_env_cfg, matt_rung2_v5_env_cfg
    from .getup_env import matt_getup_rev_env_cfg
    from .crawl_env import matt_crawl_v5_env_cfg
    from .stance_env import matt_stance_env_cfg
    from .symmetry import SymmetricRunner

    for task_id, fn, exp, its, runner in (
            ("PoOlympic-MattBio-Rung0-Stand", matt_rung0_v5_env_cfg, "mattbio_rung0", 400, MjlabOnPolicyRunner),
            ("PoOlympic-MattBio-Rung2-Omni", matt_rung2_v5_env_cfg, "mattbio_rung2", 800, SymmetricRunner),
            ("PoOlympic-MattBio-RungS-Stance", matt_stance_env_cfg, "mattbio_stance", 1000, SymmetricRunner),
            ("PoOlympic-MattBio-Rung2-Flight", matt_rung2_flight_v5_env_cfg, "mattbio_rung2", 400, SymmetricRunner),
            ("PoOlympic-MattBio-Getup-Rev", matt_getup_rev_env_cfg, "mattbio_getup", 1500, MjlabOnPolicyRunner),
            ("PoOlympic-MattBio-Crawl", matt_crawl_v5_env_cfg, "mattbio_crawl", 600, MjlabOnPolicyRunner)):
        register_mjlab_task(task_id=task_id, env_cfg=fn(), play_env_cfg=fn(play=True),
                            rl_cfg=matt_ppo_v5_cfg(exp, max_iterations=its), runner_cls=runner)

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
