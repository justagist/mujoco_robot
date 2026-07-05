"""Smoke tests for mujoco_robot (public API + quaternion boundary conversion)."""

import numpy as np

import mujoco_robot
from mujoco_robot import MujocoRobot
from mujoco_robot.mujoco_robot import wxyz_to_xyzw, xyzw_to_wxyz, wrap_angle


def test_public_api():
    assert mujoco_robot.__all__ == ["MujocoRobot", "__version__"]
    assert isinstance(mujoco_robot.__version__, str)
    assert MujocoRobot is mujoco_robot.MujocoRobot


def test_quat_convention_roundtrip():
    # External convention is [x, y, z, w]; MuJoCo is [w, x, y, z].
    q_xyzw = np.array([0.1, 0.2, 0.3, 0.9])
    assert np.allclose(wxyz_to_xyzw(xyzw_to_wxyz(q_xyzw)), q_xyzw)

    q_wxyz = np.array([0.9, 0.1, 0.2, 0.3])
    assert np.allclose(xyzw_to_wxyz(wxyz_to_xyzw(q_wxyz)), q_wxyz)

    # explicit ordering: [x,y,z,w]=[1,2,3,4] -> [w,x,y,z]=[4,1,2,3]
    assert np.allclose(xyzw_to_wxyz(np.array([1.0, 2.0, 3.0, 4.0])), [4.0, 1.0, 2.0, 3.0])


def test_wrap_angle():
    assert np.isclose(wrap_angle(3 * np.pi), np.pi) or np.isclose(wrap_angle(3 * np.pi), -np.pi)
    assert np.allclose(wrap_angle(np.array([0.0, 2 * np.pi])), [0.0, 0.0])
