# Legged Obstacle RL in IsaacLab

## Overview

Reinforcement learning for the Unitree Go1 and AlienGo quadrupeds on rough terrain, using [IsaacLab](https://isaac-sim.github.io/IsaacLab) for training with two interchangeable RL libraries: [skrl](https://skrl.readthedocs.io) and [rsl_rl](https://github.com/leggedrobotics/rsl_rl). Includes a MuJoCo sim-to-sim transfer pipeline and keyboard teleop for interactive evaluation.

**Features:**

- Velocity-commanded and direction-commanded locomotion policies
- PPO training with skrl or rsl_rl, plus teacher-student distillation (rsl_rl)
- Curriculum over 8 generated terrain types
- MuJoCo sim2sim deployment
- Keyboard teleop
- TensorBoard logging

**Keywords:** unitree, go1, aliengo, isaaclab, rsl_rl, skrl, legged-robotics, sim2sim, mujoco

## Installation

Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

Clone this repository outside the `IsaacLab` directory, then install in editable mode:

```bash
python -m pip install -e source/legged_obstacle_rl
```

Verify by listing available environments:

```bash
python scripts/list_envs.py
```

### Cache Robot Assets Locally (optional, avoids Nucleus timeouts)

Robot USDs default to NVIDIA's remote Nucleus server, so every launch streams
them over the network. Download them once to a local cache to start offline
and avoid server timeouts:

```bash
python scripts/download_assets.py            # downloads Aliengo + Go1 to ~/.cache/legged_obstacle_rl/assets
# or a custom location:
python scripts/download_assets.py --dest /data/lorl_assets
export LORL_ASSETS_DIR=/data/lorl_assets     # if you used --dest, point training at it
```

After this, `ALIENGO_CFG` / `GO1_CFG` resolve to the local copies automatically.
If an asset is missing locally, they fall back to the Nucleus
URL --- nothing breaks before the download. Re-run with `--force` to refresh.

## Scripts

Training, play and MuJoCo sim2sim commands live in
**[docs/scripts.md](docs/scripts.md)**:

- Train --- [skrl](docs/scripts.md#train-skrl) / [teacher-student (rsl_rl)](docs/scripts.md#teacherstudent-training-rsl_rl)
- Play --- [skrl](docs/scripts.md#play--skrl) / [rsl_rl](docs/scripts.md#play-rsl_rl)
- [Deploy to MuJoCo](docs/scripts.md#deploy-to-mujoco)

## Teleop Controls

Available in both `play.py` (`--teleop`) and `deploy_mujoco.py` (`--teleop`).
Requires Linux evdev python package.

| Key | Action | Range |
|-----|--------|-------|
| I / K | Forward velocity +/- | [-1.5, 1.5] m/s |
| J / L | Lateral velocity +/- | [-1.5, 1.5] m/s |
| U / O | Yaw rate +/- | [-1.5, 1.5] rad/s |
| Y / H | Body height +/- | [0.1, 0.5] m |
| Ctrl+L | Toggle command lock | — |
| ESC | Stop | — |

## MuJoCo Setup

```bash
pip install gymnasium[mujoco]
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

Add the extension path to `.vscode/settings.json`:

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/legged_obstacle_rl"
    ]
}
```

### Pylance Crash

If Pylance runs out of memory from indexing too many Omniverse packages, exclude unused ones in `.vscode/settings.json`:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"
"<path-to-isaac-sim>/extscache/omni.kit.*"
"<path-to-isaac-sim>/extscache/omni.graph.*"
"<path-to-isaac-sim>/extscache/omni.services.*"
```
