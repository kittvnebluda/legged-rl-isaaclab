# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What

I want to train robots (dogs, humanoids) locomotion with obstacle avoidance.
I have been able to train go1 with as simple architecture as just one MLP
that accepts proprioception and a height map. I have read papers Miki et al
and Lee et al, and I have successfully implemented their robust architectures.
My current training stack is isaaclab + skrl/rsl_rl.

## Commands

### See envs in the repo

```bash
python scripts/list_envs.py
```

### Lint / Format

```bash
pre-commit run --all-files          # run all hooks (ruff lint+format, codespell, etc.)
ruff check --fix .                  # lint only
ruff format .                       # format only
```

## Architecture

### Two parallel env stacks

**IsaacLab (training)** — `tasks/manager_based/`  
Uses IsaacLab's `ManagerBasedRLEnvCfg` with composable config dataclasses (`@configclass`). Hierarchy:

```
LocomotionRLEnvCfg          # scene, actions, events, terminations
  └── VelocityRLEnvCfg      # velocity commands, observations, rewards, curriculum
        └── Go1RoughEnvCfg  # robot asset, architecture variants (Long/Wide/Argo)
```

Each `__init__.py` registers gym IDs (e.g. `LORL-Go1Rough-RL-v0`). Agent configs live in `go1/agents/` as YAML/Hydra entry points.

**MuJoCo (sim2sim deployment)** — `tasks/mujoco/`  
Uses Gymnasium's `MujocoEnv`. Three-layer structure:

```
LocomotionEnv (MujocoEnv, ABC)   # locomotion_env.py: physics loop, joint reorder,
                                 #   reset, base_lin_vel / projected_gravity.
                                 #   abstract: get_obs, compute_ctrl, inject_teleop
  ├── go1/go1_env.py    Go1Env       # compute_ctrl via ActuatorNetMLP (torque)
  │     ├── argo_env.py     Go1ArgoEnv / Go1ArgoHEnv   # obs 49 / 217 (15-step history)
  │     └── velocity_env.py Go1VelocityFlatEnv / ...HFieldEnv  # obs 235 (+ height scan)
  └── aliengo/aliengo_env.py AliengoEnv   # compute_ctrl passthrough (XML position actuators)
        └── direction_env.py AliengoDirectionProprioEnv (+ ICRA Flat/Sloped)  # obs 45
```

Command mixins live in `commandables.py`: `VelocityCommandable` / `ArgoCommandable` /
`DirectionCommandable` supply the command buffer (`vel_cmd` / `dir_cmd`) and `inject_teleop`.

**MRO gotcha:** mixins MUST be listed *first* in the bases, e.g.
`class Go1ArgoEnv(ArgoCommandable, Go1Env)`. `LocomotionEnv` declares `inject_teleop`
`@abstractmethod`; if the mixin comes second it sits after `LocomotionEnv` in the MRO,
the abstract version wins, and the class can't be instantiated.

### Critical joint reordering

MuJoCo and IsaacLab use different joint orderings. `joints.py` defines `isaac_to_mujoco_joints` / `mujoco_to_isaac_joints` index arrays (built by `map_indexes`) plus `isaac_home_jpos`. All joint position/velocity reads and writes must go through these mappings — easy to miss when adding new obs terms.

### compute_ctrl / ActuatorNetMLP

`LocomotionEnv.do_simulation()` calls the abstract `compute_ctrl()` every physics substep (args/return in IsaacLab joint order), then writes the result to `data.ctrl` before `mj_step`.

- **Go1** (`Go1Env`): `compute_ctrl` runs `ActuatorNetMLP.compute()` (`GO1_ACTUATOR_CFG` from `isaaclab_assets`) to convert target joint positions → torques. Core of sim2sim fidelity.
- **Aliengo** (`AliengoEnv`): `compute_ctrl` is a passthrough — the scene XML uses `<position>` actuators, so MuJoCo's built-in PD turns the target positions into torques. Keep XML `kp`/`kv` matched to the IsaacLab `ALIENGO_CFG` stiffness/damping.

### Terrain curriculum

IsaacLab training uses 8 procedurally generated sub-terrain types in `terrains/`. `ROUGH_TERRAINS_CFG` wires them together. Curriculum is activated automatically when `CurriculumCfg.terrain_levels` is set; `LocomotionRLEnvCfg.__post_init__` enables `terrain_generator.curriculum` accordingly.

## Key file locations

| Purpose | Path |
|---|---|
| IsaacLab base env config | `tasks/manager_based/locomotion/locomotion_env_cfg.py` |
| Velocity command env config | `tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` |
| Go1 rough env variants | `tasks/manager_based/locomotion/velocity/go1/rough/env_cfg.py` |
| MuJoCo base env | `tasks/mujoco/locomotion_env.py` |
| MuJoCo joint maps + home pose | `tasks/mujoco/joints.py` |
| MuJoCo command mixins | `tasks/mujoco/commandables.py` |
| MuJoCo gym registrations | `tasks/mujoco/__init__.py` |
| MuJoCo sim2sim deploy script | `scripts/rsl_rl/deploy_mujoco.py` |
| Terrain definitions | `terrains/height_field/hf_terrains.py`, `terrains/trimesh/mesh_terrains.py` |
| Teleop (evdev) | `legged_obstacle_rl/teleop.py` |
| MuJoCo scene XML | `tasks/mujoco/go1/unitree_go1/scene.xml`, `tasks/mujoco/aliengo/unitree_aliengo/scene.xml` |
