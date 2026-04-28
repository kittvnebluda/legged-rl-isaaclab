"""Script to deploy a checkpoint of an RL agent from skrl in MuJoCo."""

"""Parse args first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deploy a checkpoint of an RL agent from skrl in MuJoCo.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--config", type=str, default=None, help="Path to model's config.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)

args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os
import random
import time

import gymnasium as gym
import skrl
import torch
from packaging import version

MIN_SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(MIN_SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={MIN_SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

import legged_obstacle_rl.tasks  # noqa: F401
from skrl.envs.wrappers.torch import wrap_env

from isaaclab.utils.io.yaml import load_yaml


def main():
    cfg = load_yaml(args_cli.config)
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    args_cli.seed = random.randint(0, 10000)

    cfg["seed"] = args_cli.seed if args_cli.seed is not None else cfg["seed"]

    log_root_path = os.path.join("logs", "skrl", cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.checkpoint and ("agent" in args_cli.checkpoint or "policy" in args_cli.checkpoint):
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        print("[ERROR] Cannot load provided checkpoint")

    with gym.make(args_cli.task, render_mode="human") as env:
        try:
            dt = env.dt
        except AttributeError:
            dt = env.unwrapped.dt

        env = wrap_env(env, "gymnasium")

        cfg["trainer"]["close_environment_at_exit"] = False
        cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
        cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
        runner = Runner(env, cfg)

        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        print(f"[INFO] Joint names: {[env.model.joint(i).name for i in range(13)]}")
        runner.agent.load(resume_path)
        runner.agent.set_running_mode("eval")

        print("[DEBUG] State preprocessor in config: ", cfg["agent"]["state_preprocessor"])
        print("[DEBUG] State preprocessor:", getattr(runner.agent, "_state_preprocessor", None))
        print("[DEBUG] Time delta:", dt)

        obs, _ = env.reset()
        while 1:
            start_time = time.time()

            with torch.inference_mode():
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])
                # actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                obs, _, _, _, _ = env.step(actions)

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
