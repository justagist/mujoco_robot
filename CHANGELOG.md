# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Entries tagged **[BREAKING]** can change the behaviour of
existing code.

## [0.2.1] - 2026-10-16

### Added

- `robot.frame(name)` returns a typed link-or-site handle (`Frame`) that `get_jacobian` accepts.

### Changed

- Methods bound to one kind of element (links, sites, joints, geoms, bodies) accept a name or an
  id, including numpy integer ids. Geom ids make dynamics editing work on unnamed geoms.
- `get_jacobian` raises `TypeError` for a bare id, which cannot say whether it is a body or a
  site; pass a name or a `Frame`.

## [0.2.0] - 2026-10-16

### Added

- Wrap one robot inside a larger, host-stepped world: `MujocoRobot(model=, data=,
  joint_names=[...], root_body=...)`. Only the given joints are controlled; the rest of the world
  is never written.
- `managed_pd=` and `apply_managed_control()` for joints without actuators.
- `differential_ik(..., joint_names=[...])` to solve over a subset of joints.
- `model_builder.add_position_actuators` and `model_builder.attach_robot`.

### Changed

- **[BREAKING]** `robot-descriptions` is now an optional extra. Install
  `mujoco_robot[descriptions]` to keep using `robot_loader_utils`.
- **[BREAKING]** Passing `model` no longer changes the simulation on construction: `run_async`,
  `place_on_ground` and `managed_pd` default to False, and joints, base and `ctrl` are only reset
  if `default_joint_positions` or `default_base_position`/`default_base_orientation` are given.
  Loading from a path is unchanged.
- **[BREAKING]** The floating base is the free joint on the robot's root body, not the last free
  joint in the model. Pass `root_body=` if the robot is not the first body in the world.
- **[BREAKING]** `set_joint_positions`, `set_joint_velocities` and `set_joint_torques` raise
  `ValueError` instead of `KeyError` for joints that are not the robot's.
- **[BREAKING]** Free joints passed to those methods or to `reset_joints` now raise instead of
  being written to.

## [0.1.1] - 2026-07-06

### Added

- Pre- and post-step callables.
- `get_actuated_joint_accelerations`, `get_contact_info` and `get_ft_reading` (with optional
  smoothing).
- `include_coriolis` option for `get_gravity_compensation_torques`.
- `viewer_utils` for drawing points and frames in the viewer.

## [0.1.0] - 2026-07-06

### Added

- `differential_ik`, plus an optional `[ik]` extra (mink) for whole-body IK.
- Transparency, per-geom collision and dynamics-editing (domain randomisation) methods.
- `reset_actuated_joint_positions` and `model_builder.add_site`.

## [0.0.1] - 2026-07-05

### Added

- First release: `MujocoRobot` and `MujocoObject` with loading, stepping, state getters,
  position/torque control, contacts and force-torque sensors.

[0.2.1]: https://github.com/justagist/mujoco_robot/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/justagist/mujoco_robot/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/justagist/mujoco_robot/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/justagist/mujoco_robot/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/justagist/mujoco_robot/releases/tag/v0.0.1
