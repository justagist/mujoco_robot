"""Tests for the differential_ik helper. Physics only, no rendering."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import pytest

from mujoco_robot import MujocoRobot
from mujoco_robot.ik import differential_ik
from mujoco_robot.utils import model_builder as mb
from mujoco_robot.utils.robot_loader_utils import get_mjcf_from_awesome_robot_descriptions

PANDA = get_mjcf_from_awesome_robot_descriptions("panda_mj_description")
NEUTRAL = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.02, 0.02]


# A two-link arm on a mocap mount, with a site at the tip.
MOCAP_MOUNT_XML = """<mujoco><worldbody><body name="mount" mocap="true" pos="0 0 1">
  <body name="l1"><joint name="j1" type="hinge" axis="0 1 0"/>
    <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02" mass="1"/>
    <body name="l2" pos=".3 0 0"><joint name="j2" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02" mass="1"/>
      <site name="tip" pos=".3 0 0"/></body></body>
  </body></worldbody></mujoco>"""


def _robot() -> MujocoRobot:
    # Panda with an IK target site added at the hand.
    spec = mb.load_spec(mjcf_path=PANDA)
    mb.set_base_fixed(spec, fixed=True)
    mb.add_site(spec, "hand", "ik_target")
    return MujocoRobot(
        model=spec.compile(),
        run_async=False,
        verbose=False,
        place_on_ground=False,
        default_joint_positions=NEUTRAL,
    )


def test_position_only_converges():
    r = _robot()
    try:
        sid = r.get_site_id("ik_target")
        target = r.get_site_pose(sid)[0] + np.array([0.08, 0.05, -0.08])
        q = differential_ik(r.model, r.data, "ik_target", target)
        assert q.shape == (9,)  # actuated joints, in robot order
        r.reset_actuated_joint_positions(q)
        assert np.linalg.norm(r.get_site_pose(sid)[0] - target) < 1e-3
    finally:
        r.shutdown()


def test_pose_converges_and_keeps_orientation():
    r = _robot()
    try:
        sid = r.get_site_id("ik_target")
        start_pos, start_quat = r.get_site_pose(sid)
        target = start_pos + np.array([0.05, 0.0, -0.05])
        q = differential_ik(r.model, r.data, "ik_target", target, target_quat=start_quat)
        r.reset_actuated_joint_positions(q)
        reached_pos, reached_quat = r.get_site_pose(sid)
        assert np.linalg.norm(reached_pos - target) < 1e-3
        ang = (Rotation.from_quat(reached_quat) * Rotation.from_quat(start_quat).inv()).magnitude()
        assert ang < 1e-2
    finally:
        r.shutdown()


def test_does_not_mutate_live_data():
    r = _robot()
    try:
        qpos_before = r.data.qpos.copy()
        target = r.get_site_pose(r.get_site_id("ik_target"))[0] + np.array([0.1, 0.0, 0.0])
        differential_ik(r.model, r.data, "ik_target", target)
        assert np.allclose(r.data.qpos, qpos_before)  # solved on a copy
    finally:
        r.shutdown()


def test_uses_live_mocap_mount_pose():
    """A target at the current site pose needs no motion, also when a mocap mount has moved."""
    model = mujoco.MjModel.from_xml_string(MOCAP_MOUNT_XML)
    data = mujoco.MjData(model)
    data.qpos[:] = [0.3, -0.6]
    data.mocap_pos[0] = [0.2, -0.1, 1.3]
    data.mocap_quat[0] = [np.cos(0.4), np.sin(0.4), 0.0, 0.0]  # 0.8 rad about x
    mujoco.mj_forward(model, data)
    q = differential_ik(model, data, "tip", data.site_xpos[0].copy())
    np.testing.assert_allclose(q, [0.3, -0.6])


def test_runs_no_callbacks():
    """The solve must not run global MuJoCo callbacks on its copy."""
    calls = []
    r = _robot()
    try:
        target = r.get_site_pose(r.get_site_id("ik_target"))[0] + np.array([0.05, 0.0, -0.05])
        mujoco.set_mjcb_control(lambda m, d: calls.append("control"))
        mujoco.set_mjcb_passive(lambda m, d: calls.append("passive"))
        differential_ik(r.model, r.data, "ik_target", target)
    finally:
        mujoco.set_mjcb_control(None)
        mujoco.set_mjcb_passive(None)
        r.shutdown()
    assert calls == []


def test_missing_site_raises():
    r = _robot()
    try:
        with pytest.raises(ValueError):
            differential_ik(r.model, r.data, "no_such_site", np.zeros(3))
    finally:
        r.shutdown()


def test_helper_not_in_top_level_api():
    import mujoco_robot

    assert "differential_ik" not in mujoco_robot.__all__
