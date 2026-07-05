"""MuJoCo robot interface. This is a stand-alone file.

Implemented purely on ``mujoco.MjModel`` + ``MjData`` (no client-server).

Quaternion convention: everything exposed by this module uses ``[x, y, z, w]`` (scipy convention).
MuJoCo internally uses ``[w, x, y, z]``; conversion happens ONLY at the boundary via
:func:`wxyz_to_xyzw` / :func:`xyzw_to_wxyz`.
"""

from typing import TypeAlias
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


class MujocoObject:
    """Static information about a MuJoCo object (name<->id maps, joint/link/dof bookkeeping).

    Coming soon.
    """


class MujocoRobot(MujocoObject):
    """Robot interface utility for a generic robot in MuJoCo.

    Coming soon.
    """
