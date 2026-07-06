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
pip install "mujoco_robot[ik]"   # optional whole-body IK backend (mink)
```

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

## Features

- **Load** an MJCF/URDF file, or wrap an existing `(model, data)`; assemble a world (ground, light,
  props, force-torque sensors) with the `MjSpec` helpers in `mujoco_robot.utils.model_builder`.
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
- **Control:** writes `data.ctrl` when the model has actuators, otherwise falls back to
  `data.qfrc_applied` with a Python PD loop.
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
