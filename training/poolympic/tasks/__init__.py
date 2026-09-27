"""PoOlympics mjlab tasks. Loaded through the `mjlab.tasks` entry point (pyproject.toml)."""

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .matt_env import matt_ppo_cfg, matt_rung0_env_cfg, matt_rung1_env_cfg

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
