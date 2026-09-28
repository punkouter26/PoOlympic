"""PoOlympics mjlab tasks. Loaded through the `mjlab.tasks` entry point (pyproject.toml)."""

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .matt_env import (matt_pedestal2_env_cfg, matt_pedestal_env_cfg, matt_ppo_cfg, matt_rung0_env_cfg, matt_rung1_env_cfg,
                       matt_rung2_env_cfg, matt_rung2_sym2_env_cfg, matt_rung2_sym3_env_cfg,
                       matt_rung2_sym_env_cfg)
from .symmetry import SymmetricRunner

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
