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
    id="LORL-Go1ArgoFlat-RL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{config.__name__}.go1.argo_flat_rl_env_cfg:Go1ArgoFlatEnvCfg",
        "skrl_cfg_entry_point": f"{config.__name__}.go1.agents:skrl_argo_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="LORL-Go1ArgoFlat-RL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{config.__name__}.go1.argo_flat_rl_env_cfg:Go1ArgoFlatEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{config.__name__}.go1.agents:skrl_argo_flat_ppo_cfg.yaml",
    },
)
