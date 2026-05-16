from isaaclab.utils import configclass

from legged_obstacle_rl.tasks.manager_based.locomotion.velocity.go1.rough.env_cfg import (
    Go1RoughEnvCfg_v0,
    Go1RoughEnvCfg_v0_PLAY,
    Go1RoughEnvCfg_v0_PLAY_ICRA,
)


@configclass
class Go1RoughLongHistoryEnvCfg_v0(Go1RoughEnvCfg_v0):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 15
        self.rewards.flat_orientation_l2.weight = -0.5


@configclass
class Go1RoughLongHistoryEnvCfg_v0_PLAY(Go1RoughEnvCfg_v0_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 15


@configclass
class Go1RoughLongHistoryEnvCfg_v0_PLAY_ICRA(Go1RoughEnvCfg_v0_PLAY_ICRA):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 15
