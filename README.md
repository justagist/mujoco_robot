# MuJoCo Robot

[![CI](https://github.com/justagist/mujoco_robot/actions/workflows/ci.yml/badge.svg)](https://github.com/justagist/mujoco_robot/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/mujoco-robot)](https://pypi.org/project/mujoco-robot/)
[![Python versions](https://img.shields.io/pypi/pyversions/mujoco-robot)](https://pypi.org/project/mujoco-robot/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A general Python interface for robot simulations using [MuJoCo](https://mujoco.org). Its public API
mirrors [`pybullet_robot`](https://github.com/justagist/pybullet_robot) so that controllers written
against that backend can be run here with minimal changes. `MujocoRobot` is a single object to
load, introspect, control, and read state from a robot, implemented directly on `MjModel` +
`MjData`.

## Installation

> Requires Python >= 3.10 and `mujoco >= 3.2`.

```bash
pip install mujoco_robot
pip install "mujoco_robot[descriptions]"   # fetch robots by name (robot_descriptions)
pip install "mujoco_robot[ik]"             # optional whole-body IK backend (mink)
```

The core needs only `mujoco`, `numpy` and `scipy`. The Quickstart and most examples load robots
with `robot_loader_utils`, which needs the `[descriptions]` extra.

From source:

```bash
git clone https://github.com/justagist/mujoco_robot && cd mujoco_robot
pip install -e .
```

## Quickstart

```python
import numpy as np
from mujoco_robot import MujocoRobot
from mujoco_robot.utils.robot_loader_utils import get_mjcf_from_awesome_robot_descriptions

robot = MujocoRobot(
    mjcf_path=get_mjcf_from_awesome_robot_descriptions("panda_mj_description"),
    ee_names=["hand"],
    run_async=False,   # step manually
    render=True,       # launch the passive viewer (needs a display)
)
robot.set_position_control_mode()

target = robot.get_actuated_joint_positions()
target[0] += 0.5
while robot.is_viewer_running():
    robot.set_joint_positions(target)
    robot.step()
```

## Wrapping an existing world

When the robot is one part of a larger simulation that already exists as `(model, data)` and is
stepped by its host (other actuated joints, other free bodies, the host's own pre-step logic),
scope `MujocoRobot` to the robot's joints:

```python
robot = MujocoRobot(model=sim.model, data=sim.data,
                    joint_names=[f"arm/{j}" for j in AUBO_JOINTS],
                    ee_sites=["arm/tcp"], run_async=False)
# host steps: sim.step(); robot only reads and writes ctrl / qfrc_applied for its joints
```

**Ownership rule:** the host steps; the robot reads state and writes `ctrl` / `qfrc_applied` for
its own joints only.

- `joint_names` become `actuated_joint_names`, in that order. Every other joint, actuator and body
  is foreign: commanding or resetting it raises, and nothing writes to it. Actuators are matched
  to joints by their transmission, so their names need not match the joint names.
- The root body (`root_body`, or by default the common ancestor of the joints) gives the base pose
  and velocity; the base is floating iff the root carries a free joint. The links are the root's
  subtree, so Jacobians, link poses, contacts, transparency and dynamics editing cover the robot
  alone.
- Passing `model`/`data` changes nothing on construction unless `default_joint_positions` or
  `default_base_position`/`default_base_orientation` are given. In this mode `run_async` and
  `place_on_ground` default to False, and `managed_pd` defaults to False: a command to a joint
  with no actuator raises instead of quietly starting a PD loop on `qfrc_applied`. With
  `managed_pd=True`, the host calls `robot.apply_managed_control()` before each `mj_step`.
- `step()` is only for a robot that owns stepping; do not call it when the host steps.
- Torque mode on a scoped robot clamps only its own actuators' force range to zero, so the rest
  of the world keeps its actuation.
- Arm-only IK: `differential_ik(model, data, "arm/tcp", target, joint_names=robot.actuated_joint_names)`.

`mujoco_robot.utils.model_builder` has helpers for building such worlds with `MjSpec`:

```python
from mujoco_robot.utils import model_builder as mb

world = mujoco.MjSpec()
arm = mb.load_spec(urdf_path="aubo_i10.urdf")
mb.attach_robot(world, arm, prefix="arm/", pos=[0.0, 0.0, 0.8], quat_xyzw=[0, 0, 0, 1])
mb.add_position_actuators(world, AUBO_JOINTS, kp=2000.0, kv=100.0, prefix="arm/")
model = world.compile()
```

`add_position_actuators` adds a `general` actuator per joint with `gaintype=FIXED` and
`biastype=AFFINE`, so URDF robots (which ship no actuators) are position-controlled through
`ctrl`. Note that `MjSpec` actuators default to `biastype=NONE`: setting `biasprm` alone silently
gives a pure gain, not a position servo.

## Features

- **Load** an MJCF/URDF file, or wrap an existing `(model, data)`, optionally scoped to one robot
  inside a larger host-stepped world; assemble a world (ground, light, props, force-torque
  sensors, attached robots, position actuators) with the `MjSpec` helpers in
  `mujoco_robot.utils.model_builder`.
- **Introspect**: name<->id maps for bodies, joints, actuators, sites, geoms, and sensors, with
  per-joint `qpos`/`dof` addressing.
- **Lifecycle**: manual `step()` or a background stepping thread, an optional passive viewer,
  joint/base resets, and place-on-ground.
- **State**: joint, base, and link/site pose and velocity (quaternions as `[x, y, z, w]`), the
  geometric Jacobian, gravity-compensation torques, and a `get_robot_states` snapshot.
- **Control**: position control via the model's actuators, and torque/impedance control via
  `data.qfrc_applied`, with a PVT-PD command and position/torque mode switching.
- **Contacts and force-torque** sensing, plus **cosmetics / domain-randomisation** helpers
  (transparency, per-geom collision, dynamics editing).
- **Inverse kinematics**: `differential_ik` for a single frame (`from mujoco_robot.ik import
  differential_ik`); [`mink`](https://github.com/kevinzakka/mink) for whole-body (optional `[ik]`).

## Examples

Runnable demos in `examples/` (each takes `--headless` to run without the viewer):

### Joint Position Control

KUKA iiwa14 follows a joint-space sinusoid (position control).

![Joint-space position control demo](https://media.githubusercontent.com/media/justagist/_assets/refs/heads/main/mujoco_robot/mujoco_robot_jctrl.gif)

### Task Space Control

Panda traces a circle with Cartesian impedance control.

![Task-space control demo](https://media.githubusercontent.com/media/justagist/_assets/refs/heads/main/mujoco_robot/mujoco_robot_tsctrl.gif)

### Measuring Contact Forces using FT Sensor

Contact wrench visualised as the robot presses on a block.

![FT sensor demo](https://media.githubusercontent.com/media/justagist/_assets/refs/heads/main/mujoco_robot/mujoco_robot_ft_demo.gif)

### Single arm Inverse Kinematics

UR5e tracks a moving target with `differential_ik`.

![IK demo](https://media.githubusercontent.com/media/justagist/_assets/refs/heads/main/mujoco_robot/mujoco_robot_ik.gif)

### Whole body IK (via `mink` -- needs `pip install "mujoco_robot[ik]"`)

Go1 whole-body squat demo by doing IK for all four feet and body simultaneously.

![WB IK demo](https://media.githubusercontent.com/media/justagist/_assets/refs/heads/main/mujoco_robot/mujoco_robot_wb_ik.gif)

## Design notes

- **Quaternions:** the external API uses `[x, y, z, w]` (scipy convention); values are converted to
  MuJoCo's `[w, x, y, z]` only at the boundary.
- **Angles:** joint positions, velocities, and limits are in radians (a compiled model always
  stores radians, regardless of the MJCF `compiler angle` setting).
- **Control:** writes `data.ctrl` when the joint has an actuator, otherwise falls back to
  `data.qfrc_applied` with a Python PD loop (`managed_pd`; on by default only when loading from a
  path).
- **Fixed vs floating base:** a structural free joint (added via `MjSpec`), not a load-time flag.
- **Sensors:** force-torque sensors are declared up front (`ft_sensor_links`); the toggle only
  gates reading, since MuJoCo sensors are compile-time.

## Development

```bash
pip install -e ".[dev]"   # test + lint + build tooling
ruff check .
pytest -q
```

## License

MIT (c) Saif Sidhik
