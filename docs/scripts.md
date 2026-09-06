# Scripts

Command-line entry points for training, evaluation and deployment.
Back to the [README](../README.md).

## Train (skrl)

```bash
python scripts/skrl/train.py \
    --task=LORL-Go1Rough-RL-v0 \
    [--num_envs 4096] \
    [--checkpoint PATH] \
    [--max_iterations 1500] \
    [--video] [--video_length 200] \
    [--seed 42] \
    [--algorithm PPO]
```

Logs to `logs/skrl/` and `outputs/`. TensorBoard metrics include velocity
tracking, terrain level, and custom hyperparameters.

## Teacher–Student Training (rsl_rl)

Two-phase privileged-learning + distillation for the AlienGo direction task.
Phase A trains a privileged teacher with PPO; Phase B distills it into
a proprioception-only GRU student via DAgger. Both phases log under `logs/rsl_rl/aliengo_direction/`.

**Phase A — teacher (privileged PPO, symmetry augmentation, clean observations):**

```bash
python scripts/rsl_rl/train.py \
    --task LORL-AlienGoDirection-RL-v0 \
    --num_envs 4096 \
    --max_iterations 1500 \
    --run_name teacher \
    --headless
```

Resume a teacher:

```bash
  python scripts/rsl_rl/train.py \
      --task LORL-AlienGoDirection-RL-v0 \
      --num_envs 8192 \
      --max_iterations 1500 \
      --run_name teacher \
      --headless \
      --resume \
      --load_run 2026-06-16_14-25-03_teacher_kadupul \
      --checkpoint model_1499.pt \
      --seed -1
```

Writes checkpoints to `logs/rsl_rl/aliengo_direction/<timestamp>_teacher/`.

**Phase B — student (GRU DAgger distillation, noisy proprioception):**

```bash
python scripts/rsl_rl/train.py \
    --task LORL-AlienGoDirection-RL-Distill-v0 \
    --agent rsl_rl_distillation_cfg_entry_point \
    --num_envs 4096 \
    --max_iterations 1000 \
    --load_run <timestamp>_teacher \
    --checkpoint model_1499.pt \
    --run_name student \
    --headless
```

`--agent rsl_rl_distillation_cfg_entry_point` selects the distillation runner; `--load_run`
and `--checkpoint` point at the Phase A teacher (resolved within the shared
`aliengo_direction` experiment root). The student imitates the teacher's actions
while acting on corrupted proprioception.

## Play  (skrl)

```bash
python scripts/skrl/play.py \
    --task=LORL-Go1Rough-RL-Play-v0 \
    --checkpoint PATH \
    [--num_envs 50] \
    [--teleop] \
    [--real-time]
```

`--teleop` enables keyboard control (see [Teleop Controls](../README.md#teleop-controls)).

## Play (rsl_rl)

Teacher:

```bash
python scripts/rsl_rl/play.py
    --task LORL-AlienGoDirection-RL-Play-v0 \
    --checkpoint logs/rsl_rl/aliengo_direction/<date-time>_teacher/model_X.pt \
    [--num_envs 50] \
    [--teleop] \
    [--real-time]
```

Student:

```bash
python scripts/rsl_rl/play.py
    --task LORL-AlienGoDirection-RL-Play-v0 \
    --agent rsl_rl_distillation_cfg_entry_point \
    --checkpoint logs/rsl_rl/aliengo_direction/<date-time>_student/model_X.pt \
    [--num_envs 50] \
    [--teleop] \
    [--real-time]
```

`--teleop` enables keyboard control (see [Teleop Controls](../README.md#teleop-controls)).

## Deploy to MuJoCo

skrl:

```bash
python scripts/skrl/deploy_mujoco.py \
    --task=LORL-Go1Rough-MJ-v0 \
    --checkpoint PATH \
    --teleop \
    --real-time \
    [--config path/to/agent_cfg.yaml]
```

rsl_rl:

```bash
python scripts/rsl_rl/deploy_mujoco.py \
    --task LORL-Aliengo-Direction-MJ-v0 \
    --checkpoint logs/rsl_rl/aliengo_direction/<date-time>_student/exported/policy.pt \
    --real-time \
    --teleop
```
