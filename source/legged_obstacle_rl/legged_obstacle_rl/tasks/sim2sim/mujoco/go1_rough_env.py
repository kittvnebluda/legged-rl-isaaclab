from copy import copy
from importlib.resources import files
from typing import Literal

import mujoco
import numpy as np
import torch
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
from isaaclab.actuators import ActuatorNetMLP
from isaaclab.utils.types import ArticulationActions
from isaaclab_assets.robots.unitree import GO1_ACTUATOR_CFG
from numpy.typing import NDArray

from legged_obstacle_rl.tasks.sim2sim.mujoco.utils import (
    GRAVITY_VEC,
    isaac_home_jpos,
    isaac_joint_names,
    isaac_to_mujoco_joints,
    mujoco_to_isaac_joints,
    quat_apply_inverse,
)


class Go1RoughEnv(MujocoEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        frame_skip: int = 4,
        device: Literal["cpu", "cuda"] = "cpu",
        **kwargs,
    ):
        xml_file = str(files("legged_obstacle_rl").joinpath("tasks/sim2sim/mujoco/unitree_go1/scene.xml"))
        MujocoEnv.__init__(self, xml_file, frame_skip, observation_space=None, **kwargs)

        self.metadata = {"render_modes": ["human"], "render_fps": int(np.round(1.0 / self.dt))}
        self.device = device
        self.action_scale = 0.25

        self._main_body = 1
        self._step_counter = 0

        self.actuators = ActuatorNetMLP(
            GO1_ACTUATOR_CFG, joint_names=isaac_joint_names, joint_ids=slice(None), num_envs=1, device=self.device
        )

        # Initialize constants
        self.MAX_ACTIONS_LEN = 15
        self.HS_RESOLUTION = 0.1
        self.HS_SIZE = (1.6, 1.0)
        self.HS_OFFSET_Z = 20.0

        x_range = np.arange(-self.HS_SIZE[0] / 2, self.HS_SIZE[0] / 2 + self.HS_RESOLUTION, self.HS_RESOLUTION)
        y_range = np.arange(-self.HS_SIZE[1] / 2, self.HS_SIZE[1] / 2 + self.HS_RESOLUTION, self.HS_RESOLUTION)
        self.hs_xv, self.hs_yv = np.meshgrid(x_range, y_range)

        self.vel_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # vx,vy,wz
        self.z_cmd = 0.3

        self.action_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        self.obs_size = 235
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32)

        self.reset_model()

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], np.float32, bool, bool, dict[str, np.float32]]:

        xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        self.do_simulation(isaac_home_jpos + action * self.action_scale, self.frame_skip)
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()

        xy_vel = (xy_position_after - xy_position_before) / self.dt
        x_vel, y_vel = xy_vel

        obs = self._get_obs()

        wz_cmd_err = self.vel_cmd[2] - self.data.qvel[5]
        info = {
            "step/vx_cmd_error": abs(self.vel_cmd[0] - x_vel),
            "step/vy_cmd_error": abs(self.vel_cmd[1] - y_vel),
            "step/wz_cmd_error": abs(wz_cmd_err),
            "step/body_height_cmd_error": abs(self.z_cmd - self.data.qpos[2]),
        }

        self.actions.append(action.copy())
        if len(self.actions) > self.MAX_ACTIONS_LEN:
            del self.actions[0]

        if self.render_mode == "human":
            self.render()

        return obs, np.float32(0.0), False, False, info

    def do_simulation(self, ctrl, n_frames) -> None:
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}")

        for _ in range(n_frames):
            q = self.data.qpos[7:]
            v = self.data.qvel[6:]

            target_articulation = self.actuators.compute(
                ArticulationActions(joint_positions=torch.from_numpy(ctrl).float().unsqueeze(0)),
                torch.from_numpy(q[mujoco_to_isaac_joints]).float().unsqueeze(0),
                torch.from_numpy(v[mujoco_to_isaac_joints]).float().unsqueeze(0),
            )

            if target_articulation.joint_efforts is None:
                raise ValueError("ActuatorNetMLP returned None in joint_efforts")

            efforts_mj = target_articulation.joint_efforts.squeeze(0).detach().numpy()[isaac_to_mujoco_joints]
            self.data.ctrl[:] = efforts_mj

            mujoco.mj_step(self.model, self.data)

    def _get_obs(self):
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()
        base_ang_vel = qvel[3:6]

        obs = np.concatenate(
            (
                qpos[7:][mujoco_to_isaac_joints] - isaac_home_jpos,
                self.base_lin_vel(),
                base_ang_vel,
                qvel[6:][mujoco_to_isaac_joints],
                self.projected_gravity(),
                self.vel_cmd,
                self.actions[-1],
                self.height_scan(),
            )
        ).astype(np.float32)

        assert len(obs) == self.obs_size, f"{len(obs)} does not equal to {self.obs_size}"
        return obs

    def reset_model(self):
        self.actions = [isaac_home_jpos.copy()]
        self._ep_start_time = copy(self.data.time)

        qpos = np.concatenate([np.array([0, 0, 0.4, 1, 0, 0, 0]), isaac_home_jpos[isaac_to_mujoco_joints]])
        qvel = np.zeros(len(qpos) - 1)
        self.set_state(qpos, qvel)

        return self._get_obs()

    def print_debug(self):
        lv = self.base_lin_vel()
        lines = [
            "------------ DEBUG INFO ------------",
            f"Time  : {self.data.time:8.3f} s",
            "-------",
            f"CMD VX: {self.vel_cmd[0]:8.3f} m/s    ACTUAL VX: {lv[0]:8.3f} m/s",
            f"CMD VY: {self.vel_cmd[1]:8.3f} m/s    ACTUAL VY: {lv[1]:8.3f} m/s",
            f"CMD WZ: {self.vel_cmd[2]:8.3f} rad/s  ACTUAL WZ: {lv[2]:8.3f} rad/s",
            f"CMD Z : {self.z_cmd:8.3f} m      ACTUAL Z : {self.data.qpos[2]:8.3f} m",
            "------------------------------------",
            "",
        ]
        print("\n".join(lines))

    def base_lin_vel(self):
        base_quat = self.data.qpos[3:7]
        return quat_apply_inverse(base_quat, self.data.qvel[:3])

    def projected_gravity(self):
        q = self.data.qpos[3:7]  # (w, x, y, z)
        return quat_apply_inverse(q, GRAVITY_VEC)

    def height_scan(self):
        body_pos = self.data.xpos[self._main_body]
        body_mat = self.data.xmat[self._main_body].reshape(3, 3)

        yaw = np.arctan2(body_mat[1, 0], body_mat[0, 0])
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        yaw_mat = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])

        local_origins = np.stack(
            [self.hs_xv.flatten(), self.hs_yv.flatten(), np.full_like(self.hs_xv.flatten(), self.HS_OFFSET_Z)], axis=-1
        )

        world_origins = body_pos + local_origins @ yaw_mat.T

        world_direction = np.array([0, 0, -1.0])

        distances = []
        geom_id = np.zeros(1, dtype=np.int32)
        for origin in world_origins:
            dist = mujoco.mj_ray(
                self.model,
                self.data,
                origin,
                world_direction,
                np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
                1,
                -1,
                geom_id,
            )

            val = dist - self.HS_OFFSET_Z
            distances.append(np.clip(val, -1.0, 1.0))

        return np.array(distances)
