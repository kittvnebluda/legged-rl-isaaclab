# tasks/ architecture

Two parallel stacks. Same robot assets, no shared runtime code.

---

## IsaacLab stack (`manager_based/`)

### Config inheritance

```
ManagerBasedRLEnvCfg [isaaclab]
  └── LocomotionRLEnvCfg          scene/actions/events/terminations, decimation=4, ep=20s
        ├── VelocityRLEnvCfg      cmds=vel+height, obs=115D flat, curriculum=terrain_levels_vel
        │     ├── Go1RoughEnvCfg_v0          robot=GO1_CFG trunk, SKRL PPO, 3 arch variants
        │     ├── Go1RoughLongHistoryEnvCfg  history_length=15
        │     ├── Go1ArgoEnvCfg              flat terrain, obs=49/217D
        │     └── AlienGoRoughEnvCfg_vel     robot=ALIENGO_CFG base
        └── DirectionRLEnvCfg     cmds=[cos,sin,turn], curriculum=terrain_levels_dir
              ├── go1/Go1RoughEnvCfg_v0      policy=45D, priv=186D, RSL-RL PPO+Distill
              └── aliengo/AlienGoRoughEnvCfg_v0  policy=45D, priv=186D, RSL-RL PPO+Distill
```

### MDP import chain

```
isaaclab.envs.mdp
  └── isaaclab_tasks.locomotion.velocity.mdp   (randomize_actuator_gains, randomize_joint_parameters)
        └── manager_based/mdp                  (custom rewards, UniformDirectionCmd, UniformVelocityCmd)
              └── direction/mdp                (track_linear_velocity_ramp, terrain_levels_dir, symmetry)
```

### Direction vs velocity obs

| Group | Velocity | Direction |
|---|---|---|
| Policy | 115D (includes base_lin_vel, height scan) | 45D (no lin vel, no scan) |
| Privileged | none (flat critic) | 186D: base_lin_vel(3)+foot_contacts(4)+4×foot_scan(36)+actuator_gains(24)+forces(3)+torques(3)+actuator_delay(1) |
| Trainer | SKRL | RSL-RL (PPO + DAgger distill) |

### Direction privileged obs = 186D breakdown

`3 + 4 + 4×36 + 24 + 3 + 3 + 1 = 186`

### Body name difference

- Go1: body `trunk` (base class default — no override needed)
- Aliengo: body `base` (must override `add_base_mass`, `base_com`, `base_external_force_torque` in `__post_init__`)

### Domain randomization (direction tasks)

Both robots get startup DR:

- `randomize_actuator_gains`: stiffness+damping scale (0.7, 1.3)
- `randomize_joint_parameters`: friction abs (0.0, 0.4), armature abs (0.0, 0.05)

PLAY variants pin them to nominal: stiffness/damping=(1.0,1.0), friction=0.2, armature=0.02

### Gym IDs

```
LORL-Go1Rough-RL-{v0,Play-v0,Play-ICRA-v0,LongArch-*,WideArch-*}     SKRL
LORL-Go1RoughLongHistory-RL-{v0,Play-v0,Play-ICRA-v0}                  SKRL
LORL-Go1Argo-RL-{v0,Play-v0,H15-v0,H15-Play-v0}                        SKRL
LORL-Go1Direction-RL-{v0,Play-v0,Distill-v0,Play-ICRA-v0}              RSL-RL
LORL-AlienGoRough-RL-{v0,Play-v0}                                       SKRL
LORL-AlienGoDirection-RL-{v0,Play-v0,Play-ICRA-v0,Distill-v0}          RSL-RL
```

---

## MuJoCo stack (`mujoco/`)

### Class hierarchy

```
MujocoEnv [gymnasium]
  └── LocomotionEnv (ABC)     physics loop, joint reorder, reset, base_lin_vel, projected_gravity
        ├── Go1Env (ABC)      compute_ctrl = ActuatorNetMLP (PD targets → torques)
        │     ├── VelocityCommandable + Go1Env → Go1VelocityFlatEnv      obs=235
        │     ├── VelocityCommandable + Go1Env → Go1VelocityHFieldEnv    obs=235
        │     ├── ArgoCommandable    + Go1Env → Go1ArgoEnv               obs=49
        │     ├── ArgoCommandable    + Go1Env → Go1ArgoHEnv              obs=217
        │     └── DirectionCommandable + Go1Env → Go1DirectionEnv        obs=45
        └── AliengoEnv (ABC)  compute_ctrl = passthrough (XML position actuators)
              └── DirectionCommandable + AliengoEnv → AliengoDirectionProprioEnv  obs=45
                    ├── AliengoDirectionProprioIcraFlatEnv
                    └── AliengoDirectionProprioIcraSlopedEnv
```

### MRO rule — CRITICAL

Commandable mixin MUST be first in bases:

```python
class Go1DirectionEnv(DirectionCommandable, Go1Env)  # correct
class Go1DirectionEnv(Go1Env, DirectionCommandable)  # WRONG — abstract inject_teleop wins, can't instantiate
```

### Go1 vs Aliengo actuator model

- **Go1** `compute_ctrl`: runs `ActuatorNetMLP(GO1_ACTUATOR_CFG)` → torques. Core of sim2sim fidelity.
- **Aliengo** `compute_ctrl`: identity passthrough. MuJoCo XML `<position>` actuators do PD internally. Keep XML kp/kv matched to `ALIENGO_CFG` stiffness/damping.

### Joint ordering (`joints.py`)

Single file for both robots — joint names are identical (FR/FL/RR/RL × hip/thigh/calf).
`mujoco_to_isaac_joints` permutes MuJoCo order → IsaacLab order before assembling obs.
`isaac_home_jpos` = Go1 defaults (hips=±0.1, thighs=0.8/1.0, calves=-1.5).

```
