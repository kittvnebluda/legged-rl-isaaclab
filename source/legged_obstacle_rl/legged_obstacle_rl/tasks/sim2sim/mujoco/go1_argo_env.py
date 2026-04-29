from copy import copy
from importlib.resources import files
from typing import Literal

import torch

import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
from isaaclab.actuators import ActuatorNetMLP
from isaaclab.utils.types import ArticulationActions
from isaaclab_assets.robots.unitree import GO1_ACTUATOR_CFG
from numpy.typing import NDArray


def normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalize an array along the last dimension."""
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def quat_apply_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Apply an inverse quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)

    xyz = quat[:, 1:]
    t = np.cross(xyz, vec, axis=-1) * 2
    return (vec - quat[:, 0:1] * t + np.cross(xyz, t, axis=-1)).reshape(shape)


def cor(*, target, source):
    res = []
    for name in target:
        for i, iname in enumerate(source):
            if iname == name:
                res.append(i)
    return res


# fmt: off
# Joint Order in MuJoCo
mujoco_joint_names = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
# Joint Order in IsaacLab
isaac_joint_names = [
    "FL_hip_joint",   "FR_hip_joint",   "RL_hip_joint",   "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint",  "FR_calf_joint",  "RL_calf_joint",  "RR_calf_joint",
]
# Home joint positions in IsaacLab
isaac_home_jpos = np.array([
     0.1, -0.1,  0.1, -0.1, # hips
     0.8,  0.8,  1.0,  1.0, # thighs
    -1.5, -1.5, -1.5, -1.5, # calves
])
# fmt: on
isaac_to_mujoco_joints = cor(target=mujoco_joint_names, source=isaac_joint_names)
mujoco_to_isaac_joints = cor(target=isaac_joint_names, source=mujoco_joint_names)

ZERO_ACTION = np.zeros(12, dtype=np.float32)
GRAVITY_VEC = normalize(np.array([[0.0, 0.0, -9.81]], dtype=np.float32)).squeeze(0)


class Go1ArgoEnv(MujocoEnv):
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

        self.vel_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # vx,vy,wz
        self.z_cmd = 0.3

        self.action_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        self.obs_size = 49
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
        base_quat = qpos[3:7]
        base_lin_vel = quat_apply_inverse(base_quat, qvel[:3])
        base_ang_vel = qvel[3:6]

        obs = np.concatenate(
            (
                qpos[7:][mujoco_to_isaac_joints] - isaac_home_jpos,
                base_lin_vel,
                base_ang_vel,
                qvel[6:][mujoco_to_isaac_joints],
                self.projected_gravity(),
                self.vel_cmd,
                (self.z_cmd,),
                self.actions[-1],
            )
        ).astype(np.float32)

        assert len(obs) == self.obs_size
        return obs

    def reset_model(self):
        self.actions = [isaac_home_jpos.copy()]
        self._ep_start_time = copy(self.data.time)

        qpos = np.concatenate([np.array([0, 0, 0.4, 1, 0, 0, 0]), isaac_home_jpos[isaac_to_mujoco_joints]])
        qvel = np.zeros(len(qpos) - 1)
        self.set_state(qpos, qvel)

        return self._get_obs()

    def print_debug(self):
        lines = [
            "------------ DEBUG INFO ------------",
            f"Time  : {self.data.time:8.3f} s",
            "-------",
            f"CMD VX: {self.vel_cmd[0]:8.3f} m/s    ACTUAL VX: {self.data.qvel[0]:8.3f} m/s",
            f"CMD VY: {self.vel_cmd[1]:8.3f} m/s    ACTUAL VY: {self.data.qvel[1]:8.3f} m/s",
            f"CMD WZ: {self.vel_cmd[2]:8.3f} rad/s  ACTUAL WZ: {self.data.qvel[5]:8.3f} rad/s",
            f"CMD Z : {self.z_cmd:8.3f} m      ACTUAL Z : {self.data.qpos[2]:8.3f} m",
            "------------------------------------",
            "",
        ]
        print("\n".join(lines))

    def projected_gravity(self):
        q = self.data.qpos[3:7]  # (w, x, y, z)
        return quat_apply_inverse(q, GRAVITY_VEC)


class Go1ArgoHEnv(Go1ArgoEnv):
    def __init__(self, frame_skip: int = 10, device: Literal["cpu", "cuda"] = "cpu", **kwargs):
        super().__init__(frame_skip, device, **kwargs)
        self.obs_size = 217
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32)
        self.reset_model()

    def _get_obs(self):
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()
        base_quat = qpos[3:7]
        base_lin_vel = quat_apply_inverse(base_quat, qvel[:3])
        base_ang_vel = qvel[3:6]

        obs = np.concatenate(
            (
                -qpos[7:][mujoco_to_isaac_joints] + isaac_home_jpos,
                base_lin_vel,
                base_ang_vel,
                qvel[6:][mujoco_to_isaac_joints],
                self.projected_gravity(),
                self.vel_cmd,
                (self.z_cmd,),
                np.concatenate(self.actions[-15:]),
            )
        ).astype(np.float32)

        return obs

    def reset_model(self):
        self.actions = [ZERO_ACTION.copy() for _ in range(15)]
        self._ep_start_time = copy(self.data.time)

        qpos = np.concatenate([np.array([0, 0, 0.4, 1, 0, 0, 0]), isaac_home_jpos[isaac_to_mujoco_joints]])
        qvel = np.zeros(len(qpos) - 1)
        self.set_state(qpos, qvel)

        return self._get_obs()
