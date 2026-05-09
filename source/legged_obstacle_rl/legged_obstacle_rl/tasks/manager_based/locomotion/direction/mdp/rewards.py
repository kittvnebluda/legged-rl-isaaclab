from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_linear_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Command shape: (num_envs, 3) -> [cos(yaw), sin(yaw), turn_dir]
    command = env.command_manager.get_command("base_direction")
    cmd_dir = command[:, :2]  # Extract horizontal direction vector
    cmd_norm = torch.norm(cmd_dir, dim=1)
    is_standing_cmd = cmd_norm < 0.1

    vel_xy_b = asset.data.root_lin_vel_b[:, :2]
    v_pr = torch.sum(vel_xy_b * cmd_dir, dim=1)

    rew = torch.where(v_pr >= 0.6, 1.0, 0.0)
    rew = torch.where(v_pr < 0.6, torch.exp(-2.0 * torch.square(v_pr - 0.6)), rew)
    rew = torch.where(is_standing_cmd, 0.0, rew)

    return rew


def track_angular_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    command = env.command_manager.get_command("base_direction")
    cmd_turn = command[:, 2]  # Discrete turning command: -1, 0, 1
    vel_yaw_b = asset.data.root_ang_vel_b[:, 2]

    w_pr = cmd_turn * vel_yaw_b

    return torch.where(w_pr >= 0.6, 1.0, torch.exp(-1.5 * torch.square(w_pr - 0.6)))


def base_motion_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Command shape: (num_envs, 3) -> [cos(yaw), sin(yaw), turn_dir]
    command = env.command_manager.get_command("base_direction")
    cmd_dir = command[:, :2]

    lv_xy = asset.data.root_lin_vel_b[:, :2]
    av_xy = asset.data.root_ang_vel_b[:, :2]

    v_pr = torch.sum(lv_xy * cmd_dir, dim=1)
    v_o = torch.norm(lv_xy - v_pr.unsqueeze(1) * cmd_dir, dim=1)

    return torch.exp(-1.5 * torch.square(v_o)) + torch.exp(-1.5 * torch.sum(torch.square(av_xy), dim=1))


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

    is_commanded_to_move = torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= is_commanded_to_move

    return reward
