from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from .commands_cfg import UniformDirectionCommandCfg

logger = logging.getLogger(__name__)


class UniformDirectionCommand(CommandTerm):
    r"""Command generator that produces a directional command vector as described in Lee et al. (2020).

    The command comprises a target horizontal direction in the robot's base frame
    and a discrete turning direction. Mathematically, it is defined as:
        command = < cos(ψ_T), sin(ψ_T), ω̂_T >
    where ψ_T is the commanded yaw angle and ω̂_T ∈ {-1, 0, 1} is the turning direction.
    A standing command is represented as <0.0, 0.0, 0.0>.

    Unlike velocity tracking commands, this generator only prescribes a heading and turning
    intent, allowing the policy to autonomously determine an appropriate speed based on terrain.
    """

    cfg: UniformDirectionCommandCfg

    def __init__(self, cfg: UniformDirectionCommandCfg, env):
        """Initialize the command generator."""
        super().__init__(cfg, env)

        # obtain the robot asset
        self.robot: Articulation = env.scene[cfg.asset_name]

        # command buffer: [cos_yaw, sin_yaw, turn_dir]
        self.dir_command_b = torch.zeros(self.num_envs, 3, device=self.device)

        # standing env tracking
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # metrics
        self.metrics["error_dir"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_turn"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["v_pr"] = torch.zeros(self.num_envs, device=self.device)  # velocity projection metric from paper

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformDirectionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tYaw range: {self.cfg.ranges.yaw}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired direction command in the base frame. Shape is (num_envs, 3)."""
        return self.dir_command_b

    def _update_metrics(self):
        """Log tracking metrics and the velocity projection (v_pr) used in the paper's reward."""
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        # Actual velocity direction in base frame
        vel_xy_b = self.robot.data.root_lin_vel_b[:, :2]
        vel_norm = torch.norm(vel_xy_b, dim=-1, keepdim=True)
        vel_dir_b = torch.where(vel_norm > 1e-5, vel_xy_b / vel_norm, torch.zeros_like(vel_xy_b))

        # Commanded direction
        cmd_dir = self.dir_command_b[:, :2]

        # 1. Direction alignment error (angle between commanded and actual direction)
        dot_prod = torch.sum(cmd_dir * vel_dir_b, dim=-1).clamp(-1.0, 1.0)
        dir_error = torch.acos(dot_prod)
        self.metrics["error_dir"] += dir_error / max_command_step

        # 2. Turning alignment error
        # We compare commanded turn dir with sign of actual angular velocity
        actual_turn_dir = torch.sign(self.robot.data.root_ang_vel_b[:, 2])
        turn_error = torch.abs(self.dir_command_b[:, 2] - actual_turn_dir)
        self.metrics["error_turn"] += turn_error / max_command_step

        # 3. Velocity projection (v_pr) - core metric from the paper's reward function
        # v_pr = (base_velocity) · (commanded_direction)
        self.metrics["v_pr"] += torch.sum(self.robot.data.root_lin_vel_b[:, :2] * cmd_dir, dim=-1) / max_command_step

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample direction commands for the specified environments."""
        r = torch.empty(len(env_ids), device=self.device)

        # Sample yaw angle uniformly from range
        yaw = r.uniform_(*self.cfg.ranges.yaw)
        self.dir_command_b[env_ids, 0] = torch.cos(yaw)
        self.dir_command_b[env_ids, 1] = torch.sin(yaw)

        # Sample discrete turning direction: {-1, 0, 1}
        # torch.randint(low, high) is exclusive on high, so -1 to 2 gives -1, 0, 1
        self.dir_command_b[env_ids, 2] = torch.randint(-1, 2, (len(env_ids),), device=self.device).float()

        # Update standing environments
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Post-processes the command. Enforces zero command for standing environments."""
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.dir_command_b[standing_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set visibility of debug visualization markers."""
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Callback to update debug visualization markers."""
        if not self.robot.is_initialized:
            return

        # Base position slightly above the robot for visibility
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        # Resolve goal direction and current velocity direction to arrow visuals
        goal_dir = self.command[:, :2]
        goal_scale, goal_quat = self._resolve_xy_direction_to_arrow(goal_dir)

        curr_vel = self.robot.data.root_lin_vel_b[:, :2]
        curr_scale, curr_quat = self._resolve_xy_direction_to_arrow(curr_vel)

        self.goal_vel_visualizer.visualize(base_pos_w, goal_quat, goal_scale)
        self.current_vel_visualizer.visualize(base_pos_w, curr_quat, curr_scale)

    def _resolve_xy_direction_to_arrow(self, xy_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts an XY direction vector to arrow scale and orientation in world frame."""
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_vec.shape[0], 1)

        norm = torch.linalg.norm(xy_vec, dim=1)
        # Scale arrow by vector magnitude, clamped to prevent vanishing arrows on stop commands
        arrow_scale[:, 0] *= torch.clamp(norm * 3.0, min=0.15, max=1.5)

        # Compute heading angle from XY components
        heading_angle = torch.atan2(xy_vec[:, 1], xy_vec[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = quat_from_euler_xyz(zeros, zeros, heading_angle)

        # Rotate from base frame to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
