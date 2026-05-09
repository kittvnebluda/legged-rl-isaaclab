from typing import Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter


def terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    vpr_threshold: float = 0.2,
) -> torch.Tensor:
    """Curriculum based on the v_pr success threshold for direction commands."""

    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_direction")  # [cos, sin, turn_dir]

    displacement = asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    distance = torch.norm(displacement, dim=1)

    move_up = torch.sum(displacement * command[env_ids, :2], dim=1) > vpr_threshold
    move_up *= distance > terrain.cfg.terrain_generator.size[0] * 0.4

    move_down = distance < terrain.cfg.terrain_generator.size[0] * 0.1

    # Don't demote if the robot is currently commanded to turn in place
    is_not_turning = torch.abs(command[env_ids, 2]) < 0.1
    move_down *= is_not_turning

    terrain.update_env_origins(env_ids, move_up, move_down)

    return torch.mean(terrain.terrain_levels.float())
