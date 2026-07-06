"""Contact and force-torque sensor tests for MujocoRobot. Physics only, no rendering."""

import mujoco
import numpy as np
import pytest

from mujoco_robot import MujocoRobot
from mujoco_robot.mujoco_robot import ContactInfo
from mujoco_robot.utils.robot_loader_utils import get_mjcf_from_awesome_robot_descriptions

PANDA = get_mjcf_from_awesome_robot_descriptions("panda_mj_description")
NEUTRAL = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.02, 0.02]

# A 1 kg box (freejoint) above a ground plane: deterministic contact.
BOX_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 .05"/>
    <body name="box" pos="0 0 .3">
      <freejoint/>
      <geom name="boxg" type="box" size=".1 .1 .1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def panda(**kw) -> MujocoRobot:
    kw.setdefault("run_async", False)
    kw.setdefault("verbose", False)
    kw.setdefault("place_on_ground", False)
    kw.setdefault("default_joint_positions", NEUTRAL)
    return MujocoRobot(mjcf_path=PANDA, **kw)


def test_ft_reading_sign_and_frames():
    r = panda(ft_sensor_links=["hand"])
    try:
        for _ in range(50):
            r.step()
        raw = r.get_link_ft_measurement("hand")  # sensor frame, no sign flip
        force, torque = r.get_ft_reading(in_global_frame=False)
        assert np.allclose(force, -raw[:3])  # negated to be the force on the parent body
        assert np.allclose(torque, -raw[3:])
        fg, tg = r.get_ft_reading(in_global_frame=True)
        assert fg.shape == (3,) and tg.shape == (3,)
    finally:
        r.shutdown()


def test_ft_reading_requires_a_sensor():
    r = panda()  # no ft_sensor_links
    try:
        with pytest.raises(ValueError):
            r.get_ft_reading()
    finally:
        r.shutdown()


def test_ft_smoothing_fills_buffer():
    r = panda(ft_sensor_links=["hand"], smooth_ft=True, ft_smoothing_window=5)
    try:
        for _ in range(10):
            r.step()
        assert len(r._ft_buffers["hand"]) == 5
        force, torque = r.get_ft_reading()
        assert np.all(np.isfinite(force)) and np.all(np.isfinite(torque))
    finally:
        r.shutdown()


def test_contact_info_on_resting_box():
    box = MujocoRobot(
        model=mujoco.MjModel.from_xml_string(BOX_XML),
        run_async=False,
        verbose=False,
        place_on_ground=False,
        default_base_position=[0.0, 0.0, 0.3],
    )
    try:
        for _ in range(700):
            box.step()
        contacts = box.get_contact_info()
        assert len(contacts) > 0
        contact = contacts[0]
        assert isinstance(contact, ContactInfo)
        assert contact.pos.shape == (3,)
        assert contact.frame.shape == (3, 3)
        assert contact.wrench.shape == (6,)
    finally:
        box.shutdown()


def test_contact_states_and_force_on_resting_box():
    robot = MujocoRobot(
        model=mujoco.MjModel.from_xml_string(BOX_XML),
        run_async=False,
        verbose=False,
        place_on_ground=False,
        default_base_position=[0.0, 0.0, 0.3],
    )
    try:
        box = robot.get_link_id("box")
        assert robot.get_contact_states_of_links([box])[0] == 0  # airborne at start
        for _ in range(700):
            robot.step()  # let it drop and settle
        assert robot.get_contact_states_of_links([box])[0] == 1  # now resting
        force = robot.get_link_contact_force("box")
        assert force[2] > 0.0  # net reaction points up
        assert np.isclose(force[2], 9.81, atol=1.0)  # supports its own weight (m*g)
    finally:
        robot.shutdown()


def test_no_ft_sensor_by_default():
    robot = panda()
    try:
        assert not robot.has_ft_sensor("hand")
        assert np.allclose(robot.get_link_ft_measurement("hand"), 0.0)
        assert robot.get_joint_ft_measurements().shape == (9, 6)
    finally:
        robot.shutdown()


def test_ft_sensor_injected_and_readable():
    robot = panda(ft_sensor_links=["hand"])
    try:
        assert robot.has_ft_sensor("hand")
        assert "hand" in robot.ft_sensor_links
        for _ in range(200):
            robot.step()  # settle
        wrench = robot.get_link_ft_measurement("hand")
        assert wrench.shape == (6,)
        assert np.all(np.isfinite(wrench))
        # the hand supports the fingers/tool against gravity, so it reads a non-zero load
        assert np.linalg.norm(wrench) > 1e-3
    finally:
        robot.shutdown()


def test_ft_sensor_toggle_gates_reading():
    robot = panda(ft_sensor_links=["hand"])
    try:
        for _ in range(200):
            robot.step()
        robot.toggle_ft_sensor_for_links("hand", enable=False)
        assert np.allclose(robot.get_link_ft_measurement("hand"), 0.0)
        robot.toggle_ft_sensor_for_links("hand", enable=True)
        assert not np.allclose(robot.get_link_ft_measurement("hand"), 0.0)
    finally:
        robot.shutdown()


def test_full_joint_states_shape():
    robot = panda(ft_sensor_links=["hand"])
    try:
        q, v, ft, tau = robot.get_full_joint_states()
        assert q.shape == (9,)
        assert v.shape == (9,)
        assert ft.shape == (9, 6)
        assert tau.shape == (9,)
    finally:
        robot.shutdown()


def test_ee_contact_states_in_robot_state():
    robot = panda(ee_names=["hand"])
    try:
        state = robot.get_robot_states()
        assert state.ee_contact_states.shape == (1,)
        assert state.ee_contact_states[0] == 0  # hand not touching anything
    finally:
        robot.shutdown()
