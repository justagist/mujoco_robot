"""State-getter tests for MujocoRobot: joint/base/link state, Jacobian, gravity compensation.

Includes empirical checks for the two error-prone conventions: the [x,y,z,w] quaternion boundary
and the (linear, angular) ordering of velocities.
"""

import mujoco
import numpy as np

from mujoco_robot import MujocoRobot
from mujoco_robot.utils.robot_loader_utils import get_mjcf_from_awesome_robot_descriptions

PANDA = get_mjcf_from_awesome_robot_descriptions("panda_mj_description")


def make(**kw) -> MujocoRobot:
    kw.setdefault("run_async", False)
    kw.setdefault("verbose", False)
    kw.setdefault("place_on_ground", False)
    kw.setdefault("load_ground_plane", False)
    kw.setdefault("ee_names", ["hand"])
    return MujocoRobot(mjcf_path=PANDA, **kw)


def test_joint_accelerations_shape():
    r = make()
    try:
        r.step()
        assert r.get_actuated_joint_accelerations().shape == (9,)
    finally:
        r.shutdown()


def test_gravity_compensation_modes():
    # A within-limits config, so no joint-limit constraint force is active (which mj_inverse would
    # otherwise legitimately include, making the two modes disagree even at rest).
    r = make(default_joint_positions=[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.02, 0.02])
    try:
        pure = r.get_gravity_compensation_torques()
        assert pure.shape == (9,) and np.all(np.isfinite(pure))
        # at rest the Coriolis terms vanish, so both modes agree
        assert np.allclose(
            pure, r.get_gravity_compensation_torques(include_coriolis=True), atol=1e-6
        )
        # with motion they differ (Coriolis / centrifugal present)
        r.data.qvel[:] = 0.8
        mujoco.mj_forward(r.model, r.data)
        moving = r.get_gravity_compensation_torques(include_coriolis=True)
        pure_gravity = r.get_gravity_compensation_torques()  # scratch at qvel=0
        assert not np.allclose(moving, pure_gravity)
    finally:
        r.shutdown()


def test_joint_states_shapes_and_values():
    q0 = np.linspace(-0.4, 0.4, 9)
    r = make(default_joint_positions=q0)
    try:
        q, v, tau = r.get_joint_states()
        assert q.shape == v.shape == tau.shape == (9,)
        assert np.allclose(q, q0)
        assert np.allclose(v, 0.0)  # no motion yet
        assert np.allclose(r.get_actuated_joint_positions(), q0)
    finally:
        r.shutdown()


def test_base_pose_fixed_base():
    r = make()
    try:
        pos, quat = r.get_base_pose()
        assert np.allclose(pos, [0.0, 0.0, 0.0], atol=1e-6)
        assert np.allclose(quat, [0.0, 0.0, 0.0, 1.0], atol=1e-6)  # identity, xyzw
    finally:
        r.shutdown()


def test_base_pose_quaternion_roundtrip():
    # 90 degrees about +z -> xyzw = [0, 0, sin45, cos45]
    ref = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    r = make(use_fixed_base=False)
    try:
        r.reset_base_pose(position=np.array([0.0, 0.0, 1.0]), orientation=ref)
        _, quat = r.get_base_pose()
        # equal up to overall sign (q and -q are the same rotation)
        assert np.allclose(quat, ref, atol=1e-6) or np.allclose(-quat, ref, atol=1e-6)
    finally:
        r.shutdown()


def test_base_velocity_ordering():
    # Isolate linear and angular so there is no omega x r coupling at the body origin.
    r = make(use_fixed_base=False)
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    at_1m = np.array([0.0, 0.0, 1.0])
    try:
        # Pure linear: with no rotation every point moves at the base linear velocity.
        r.reset_base_pose(at_1m, identity, base_linear_velocity=np.array([0.3, 0.0, 0.0]))
        lin, ang = r.get_base_velocity()
        assert np.allclose(lin, [0.3, 0.0, 0.0], atol=1e-6)  # linear returned first
        assert np.allclose(ang, [0.0, 0.0, 0.0], atol=1e-6)  # angular returned second

        # Pure angular about +z reads back into the angular slot.
        r.reset_base_pose(at_1m, identity, base_angular_velocity=np.array([0.0, 0.0, 0.5]))
        _, ang2 = r.get_base_velocity()
        assert np.allclose(ang2, [0.0, 0.0, 0.5], atol=1e-6)
    finally:
        r.shutdown()


def test_link_pose_and_com():
    r = make()
    try:
        hand = r.get_link_id("hand")
        pos, quat = r.get_link_pose(hand)
        assert pos[2] > 0.0  # the hand sits above the base
        assert quat.shape == (4,)
        com_pos, com_quat = r.get_link_com_pose(hand)
        assert com_pos.shape == (3,) and com_quat.shape == (4,)
    finally:
        r.shutdown()


def test_link_velocity_shape():
    r = make()
    try:
        lin, ang = r.get_link_velocity(r.get_link_id("hand"))
        assert lin.shape == (3,) and ang.shape == (3,)
    finally:
        r.shutdown()


def test_jacobian_shape_and_no_side_effects():
    r = make()
    try:
        jac = r.get_jacobian("hand")
        assert jac.shape == (6, 9)
        assert np.all(np.isfinite(jac))

        # evaluating at a different configuration must not disturb the live state
        qpos_before = r.data.qpos.copy()
        jac2 = r.get_jacobian("hand", joint_angles=np.full(9, 0.2))
        assert jac2.shape == (6, 9)
        assert np.allclose(r.data.qpos, qpos_before)
    finally:
        r.shutdown()


def test_gravity_compensation():
    r = make()
    try:
        tau = r.get_gravity_compensation_torques()
        assert tau.shape == (9,)
        assert np.all(np.isfinite(tau))
        assert np.any(np.abs(tau) > 1e-6)  # the arm feels gravity at some joints
    finally:
        r.shutdown()


def test_robot_states_bundle():
    from mujoco_robot.mujoco_robot import MujocoRobotState

    r = make()
    try:
        state = r.get_robot_states()
        assert isinstance(state, MujocoRobotState)
        assert state.base_position.shape == (3,)
        assert state.base_com_position.shape == (3,)
        assert state.base_quaternion.shape == (4,)
        assert state.base_velocity_linear.shape == (3,)
        assert state.base_velocity_angular.shape == (3,)
        assert state.actuated_joint_positions.shape == (9,)
        assert state.actuated_joint_velocities.shape == (9,)
        assert state.actuated_joint_torques.shape == (9,)
        assert state.joint_order == r.actuated_joint_names
        assert state.ee_order == ["hand"]
    finally:
        r.shutdown()
