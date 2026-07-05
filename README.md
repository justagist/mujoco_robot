# MuJoCo Robot

> 🚧 **Work in progress.**

A general Python interface for robot simulations using [MuJoCo](https://mujoco.org). Its public
API mirrors [`pybullet_robot`](https://github.com/justagist/pybullet_robot) so that controllers
written against that backend can be run here with minimal changes.

`MujocoRobot` gives you a single object to load, introspect, control, and read state from a robot,
implemented directly on `MjModel` + `MjData`.

## Installation

> Requires Python >= 3.10 and `mujoco >= 3.2`.

### From source

```bash
git clone https://github.com/justagist/mujoco_robot
cd mujoco_robot
pip install -e .
```

## Development

```bash
pip install -e ".[dev]"   # test + lint + build tooling
ruff check .
pytest -q
```

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

## Status

Work in progress, built up in phases. Panda (from `robot_descriptions`) is the reference test
robot.

## License

MIT (c) Saif Sidhik
