# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import config

##
# Register Gym environments.
##


gym.register(
    id="LORL-Go1Argo-RL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{config.__name__}.go1.argo_rl_env_cfg:Go1ArgoEnvCfg",
        "skrl_cfg_entry_point": f"{config.__name__}.go1.agents:skrl_argo_ppo_cfg.yaml",
    },
)

gym.register(
    id="LORL-Go1Argo-RL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{config.__name__}.go1.argo_rl_env_cfg:Go1ArgoEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{config.__name__}.go1.agents:skrl_argo_ppo_cfg.yaml",
    },
)

gym.register(
    id="LORL-Go1ArgoH-RL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{config.__name__}.go1.argo_rl_env_cfg:Go1ArgoHEnvCfg",
        "skrl_cfg_entry_point": f"{config.__name__}.go1.agents:skrl_argo_ppo_cfg.yaml",
    },
)

gym.register(
    id="LORL-Go1ArgoH-RL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{config.__name__}.go1.argo_rl_env_cfg:Go1ArgoHEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{config.__name__}.go1.agents:skrl_argo_ppo_cfg.yaml",
    },
)
gym.register(
    id="LORL-Go1Rough-RL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{config.__name__}.go1.rough_rl_env_cfg:Go1RoughEnvCfg_v0",
        "skrl_cfg_entry_point": f"{config.__name__}.go1.agents:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="LORL-Go1Rough-RL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{config.__name__}.go1.rough_rl_env_cfg:Go1RoughEnvCfg_v0_PLAY",
        "skrl_cfg_entry_point": f"{config.__name__}.go1.agents:skrl_rough_ppo_cfg.yaml",
    },
)
