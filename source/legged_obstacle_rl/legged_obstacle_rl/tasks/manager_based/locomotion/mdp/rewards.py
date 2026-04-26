from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_base_height_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height difference from a command height using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target_height = env.command_manager.get_command(command_name)[:, 0]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        target_height += torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)


def terrain_levels_mean(env: ManagerBasedRLEnv):
    return env.scene.terrain.terrain_levels.mean(dtype=torch.float32).item()
