from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG

from legged_obstacle_rl.tasks.manager_based.locomotion.locomotion_env_cfg import LocomotionRLEnvCfg


@configclass
class Go1ArgoEnvCfg(LocomotionRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.events.physics_material.params["num_buckets"] = 50
        self.events.physics_material.params["static_friction_range"] = (0.2, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.2, 1.0)

        self.events.base_com = None
        self.events.base_external_force_torque = None

        self.rewards.undesired_contacts = None
        self.rewards.terrain_levels_mean = None
        self.rewards.track_height.weight = -20

        self.curriculum.terrain_levels = None

        self.observations.policy.height_scan = None

        self.commands.base_velocity.ranges.lin_vel_x = (0, 0)
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.ang_vel_z = (0, 0)


@configclass
class Go1ArgoEnvCfg_PLAY(Go1ArgoEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class Go1ArgoHEnvCfg(Go1ArgoEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.actions.history_length = 15


@configclass
class Go1ArgoHEnvCfg_PLAY(Go1ArgoHEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
