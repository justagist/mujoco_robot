# MuJoCo Robot

> 🚧 **Work in progress** (pre-release).

A general Python interface for robot simulations using [MuJoCo](https://mujoco.org). Its public
API mirrors [`pybullet_robot`](https://github.com/justagist/pybullet_robot) so that controllers
written against that backend can be run here with minimal changes.

`MujocoRobot` gives you a single object to load, introspect, control, and read state from a robot,
implemented directly on `MjModel` + `MjData`.

## Features

- **Load** a robot from an MJCF/URDF file, or wrap an existing `(model, data)` pair. A minimal
  world (ground plane, light) and force-torque sensors can be assembled with the `MjSpec` helpers
  in `mujoco_robot.utils.model_builder`.
- **Introspect**: name↔id maps for bodies, joints, actuators, sites, geoms, and sensors, with
  per-joint `qpos`/`dof` addressing.
- **Lifecycle**: manual `step()` or a background stepping thread, an optional passive viewer,
  timestep/gravity access, joint/base resets, and place-on-ground.
- **State**: joint positions/velocities/efforts, base and link/site pose and velocity (quaternions
  as `[x, y, z, w]`), the geometric Jacobian, gravity-compensation torques, and a `get_robot_states`
  snapshot.
- **Control**: position control through the model's actuators (`data.ctrl`), and torque/impedance
  control through `data.qfrc_applied`, with a PVT-PD command and position/torque mode switching.
- **Contacts and force-torque**: per-link contact state and net contact force, plus compile-time
  force-torque sensors.
- **Cosmetics / domain randomisation**: visual transparency, per-geom collision toggling, and
  live-safe dynamics randomisation.

## Installation

> Requires Python >= 3.10 and `mujoco >= 3.2`.

```bash
git clone https://github.com/justagist/mujoco_robot
cd mujoco_robot
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

## Examples

Runnable demos live in `examples/` (each takes `--headless` to run without the viewer):

- `demo_joint_position_control.py`: the arm follows a joint-space sinusoid under position control.
- `demo_task_space_control.py`: Cartesian impedance control traces a circle with the end-effector.
- `demo_ft_sensor.py`: the hand presses onto a block and the measured force-torque wrench is shown.
- `demo_reference_ghost.py`: a translucent, collision-free robot posed kinematically as a reference.

## Design notes

- **Quaternions:** the external API uses `[x, y, z, w]` (scipy convention); values are converted to
  MuJoCo's `[w, x, y, z]` only at the boundary.
- **Angles:** joint positions, velocities, and limits are in radians. A compiled model always
  stores radians regardless of the MJCF `compiler angle` setting, so no conversion is done.
- **Backend:** pure `MjModel` + `MjData` (no client-server). Pass an existing `(model, data)` to
  wrap a robot that is already part of a scene.
- **End-effector frame:** a body by default; a site can be used optionally.
- **Control:** writes `data.ctrl` when the model has actuators, otherwise falls back to
  `data.qfrc_applied` with a Python PD loop.
- **Fixed vs floating base:** a structural free joint (added via `MjSpec`), not a load-time flag.
- **Sensors:** MuJoCo sensors are compile-time, so force-torque sensors are declared up front
  (`ft_sensor_links`) and the toggle only gates reading, rather than adding/removing at runtime.

## Development

```bash
pip install -e ".[dev]"   # test + lint + build tooling
ruff check .
pytest -q
```

## License

MIT (c) Saif Sidhik
