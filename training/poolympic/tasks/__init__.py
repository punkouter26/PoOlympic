"""PoOlympics mjlab tasks. Loaded through the `mjlab.tasks` entry point (pyproject.toml).
A process registers the tasks of its body only ($POOLYMPIC_BODY, default matt — contract/mdp load that body)."""

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .. import bodies

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
