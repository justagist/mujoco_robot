"""Shared types, conventions and conversions for mujoco_robot.

Quaternion convention: everything exposed by this package uses ``[x, y, z, w]`` (scipy
convention). MuJoCo internally uses ``[w, x, y, z]``; conversion happens ONLY at the boundary via
:func:`wxyz_to_xyzw` / :func:`xyzw_to_wxyz`.

Angle convention: all joint quantities exposed here (positions, velocities, limits) are in
radians. A compiled ``MjModel`` always stores angles in radians regardless of the MJCF
``compiler angle`` setting (which only affects how the XML *source* is parsed), so no runtime
conversion is needed or performed.
"""

from typing import List, Literal, Optional, TypeAlias
from dataclasses import dataclass

import numpy as np

QuatType: TypeAlias = np.ndarray
"""Numpy array representing a quaternion in format [x,y,z,w] (scipy convention)."""
Vector3D: TypeAlias = np.ndarray
"""Numpy array representing a 3D cartesian vector in format [x,y,z]."""


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap the provided angle(s) (radians) to be within -pi and pi.

    Args:
        angle (float | np.ndarray): Input angle (or array of angles) in radians.

    Returns:
        float | np.ndarray: Output after wrapping.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def wxyz_to_xyzw(quat_wxyz: np.ndarray) -> QuatType:
    """Convert a MuJoCo quaternion ``[w,x,y,z]`` to the external ``[x,y,z,w]`` convention.

    This is the single conversion point when reading orientations out of MuJoCo.

    Args:
        quat_wxyz (np.ndarray): Quaternion in MuJoCo order ``[w, x, y, z]``.

    Returns:
        QuatType: Quaternion in ``[x, y, z, w]`` order.
    """
    q = np.asarray(quat_wxyz, dtype=float)
    return np.array([q[1], q[2], q[3], q[0]])


def xyzw_to_wxyz(quat_xyzw: QuatType) -> np.ndarray:
    """Convert an external ``[x,y,z,w]`` quaternion to MuJoCo's ``[w,x,y,z]`` convention.

    This is the single conversion point when writing orientations into MuJoCo.

    Args:
        quat_xyzw (QuatType): Quaternion in ``[x, y, z, w]`` order.

    Returns:
        np.ndarray: Quaternion in MuJoCo order ``[w, x, y, z]``.
    """
    q = np.asarray(quat_xyzw, dtype=float)
    return np.array([q[3], q[0], q[1], q[2]])


@dataclass
class MujocoJointInfo:
    """Static information about a single joint (from ``MjModel``)."""

    joint_id: int
    joint_name: str
    joint_type: int
    """MuJoCo joint type (``mujoco.mjtJoint``): free / ball / slide / hinge."""
    qpos_adr: int
    """Start index of this joint in ``data.qpos``."""
    qpos_dim: int
    """Number of ``qpos`` entries (free=7, ball=4, slide/hinge=1)."""
    dof_adr: int
    """Start index of this joint in ``data.qvel`` / dof-indexed arrays."""
    dof_dim: int
    """Number of dof entries (free=6, ball=3, slide/hinge=1)."""
    body_id: int
    """Id of the body this joint moves."""
    body_name: str
    lower_limit: float
    upper_limit: float
    limited: bool
    """Whether the joint has enforced range limits."""
    axis: Vector3D
    """Joint axis in the local frame."""
    damping: float
    """dof damping (nan for multi-dof joints)."""
    actuator_id: Optional[int]
    """Id of the JOINT-transmission actuator driving this joint, or None."""


@dataclass
class MujocoRobotState:
    """A snapshot of common robot state (world frame; quaternions are [x,y,z,w]).

    Returned by :meth:`MujocoRobot.get_robot_states`.
    """

    base_position: Vector3D
    base_com_position: Vector3D
    base_quaternion: QuatType
    base_velocity_linear: Vector3D
    base_velocity_angular: Vector3D
    actuated_joint_positions: np.ndarray
    actuated_joint_velocities: np.ndarray
    actuated_joint_torques: np.ndarray
    joint_order: List[str]
    """Actuated-joint names, in the order the joint arrays are given."""
    ee_order: List[str]
    """End-effector names reported."""
    ee_contact_states: np.ndarray
    """Binary contact state (1 = in contact, 0 = not), aligned with ee_order."""


@dataclass
class ContactInfo:
    """Detailed information about a single active contact (see ``get_contact_info``)."""

    pos: Vector3D
    """Contact point in the world frame."""
    frame: np.ndarray
    """3x3 contact frame; rows are the contact axes in world coordinates (row 0 = normal / x)."""
    wrench: np.ndarray
    """6D contact wrench [force(3), torque(3)] in the contact frame (from ``mj_contactForce``)."""


@dataclass(frozen=True)
class Frame:
    """A resolved link-or-site frame (see :meth:`MujocoObject.frame`).

    Methods that accept any frame (e.g. ``get_jacobian``) take a name or one of these, never a bare
    id, since a bare id cannot say whether it is a body id or a site id.
    """

    kind: Literal["link", "site"]
    id: int
    name: str
