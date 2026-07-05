"""Addressing / structure tests for MujocoObject.

Uses a small inline MJCF (floating base + hinge + slide + continuous hinge) for exact qpos/dof
bookkeeping assertions, plus the reference Panda for a real-robot check.
"""

import mujoco
import numpy as np
import pytest

from mujoco_robot.mujoco_robot import MujocoObject
from mujoco_robot.utils.robot_loader_utils import get_mjcf_from_awesome_robot_descriptions

# base (freejoint) -> link1 (hinge, limited) -> link2 (slide, limited) -> link3 (hinge, unlimited)
TESTBOT_XML = """
<mujoco model="testbot">
  <compiler angle="radian"/>
  <worldbody>
    <light pos="0 0 2"/>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <geom name="base_geom" type="box" size="0.1 0.1 0.1" mass="2.0"/>
      <body name="link1" pos="0 0 0.1">
        <joint name="j1" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom name="g1" type="capsule" fromto="0 0 0 0 0 0.2" size="0.02" mass="1.0"/>
        <body name="link2" pos="0 0 0.2">
          <joint name="j2" type="slide" axis="1 0 0" range="0 0.5"/>
          <geom name="g2" type="sphere" size="0.03" mass="0.5"/>
          <body name="link3" pos="0.1 0 0">
            <joint name="j3" type="hinge" axis="0 1 0"/>
            <geom name="g3" type="sphere" size="0.03" mass="0.5"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def testbot() -> MujocoObject:
    return MujocoObject(mujoco.MjModel.from_xml_string(TESTBOT_XML))


def test_counts_and_dims(testbot):
    m = testbot.model
    # free(7/6) + hinge(1/1) + slide(1/1) + hinge(1/1)
    assert m.nq == 10
    assert m.nv == 9
    assert testbot.num_actuated_joints == 3


def test_floating_base_and_free_joint(testbot):
    assert testbot.has_floating_base
    root = testbot.joint_name_to_info["root"]
    assert root.joint_type == int(mujoco.mjtJoint.mjJNT_FREE)
    assert (root.qpos_adr, root.qpos_dim) == (0, 7)
    assert (root.dof_adr, root.dof_dim) == (0, 6)
    assert testbot.free_joint_id == testbot.get_joint_id("root")
    # the free (base) joint must NOT be counted as an actuated joint
    assert "root" not in testbot.actuated_joint_names


def test_actuated_joint_order_and_addressing(testbot):
    assert testbot.actuated_joint_names == ["j1", "j2", "j3"]
    assert testbot.actuated_joint_qpos_adrs == [7, 8, 9]
    assert testbot.actuated_joint_dof_adrs == [6, 7, 8]
    j1 = testbot.joint_name_to_info["j1"]
    assert (j1.qpos_adr, j1.dof_adr, j1.limited) == (7, 6, True)
    assert testbot.actuated_joint_lower_limits == [-1.0, 0.0, 0.0]


def test_continuous_joint_detection(testbot):
    # j3 is an unlimited hinge -> continuous; j2 (unlimited slide) is NOT continuous
    assert testbot.has_continuous_joints
    assert testbot.continuous_joint_names == ["j3"]
    assert not testbot.joint_name_to_info["j3"].limited


def test_links_exclude_world_and_base(testbot):
    assert testbot.link_names == ["base", "link1", "link2", "link3"]
    assert "world" not in testbot.link_names
    assert testbot.base_name == "base"
    assert testbot.name == "base"


def test_name_id_roundtrip(testbot):
    jid = testbot.get_joint_id("j2")
    assert testbot.get_joint_name(jid) == "j2"
    lid = testbot.get_link_id("link2")
    assert testbot.get_link_name(lid) == "link2"
    assert testbot.get_joint_ids(["j1", "j3"]) == [
        testbot.get_joint_id("j1"),
        testbot.get_joint_id("j3"),
    ]


def test_mass(testbot):
    assert np.isclose(testbot.mass, 4.0)  # 2.0 + 1.0 + 0.5 + 0.5
    assert testbot.get_link_mass("base") > 0
    # total mass equals the sum of per-link masses
    assert np.isclose(sum(testbot.get_link_mass(n) for n in testbot.link_names), testbot.mass)


def test_no_actuators_in_testbot(testbot):
    assert not testbot.has_actuators
    assert testbot.joint_name_to_info["j1"].actuator_id is None


# --- reference robot -------------------------------------------------------


@pytest.fixture(scope="module")
def panda() -> MujocoObject:
    return MujocoObject.from_mjcf_path(
        get_mjcf_from_awesome_robot_descriptions("panda_mj_description")
    )


def test_panda_structure(panda):
    # 7 arm joints + 2 finger joints, fixed base
    assert panda.num_actuated_joints == 9
    for name in ("joint1", "joint7", "finger_joint1"):
        assert name in panda.actuated_joint_names
    assert not panda.has_floating_base
    assert panda.free_joint_id is None
    assert panda.base_name == "link0"
    assert "hand" in panda.link_names
    assert panda.mass > 0
    assert panda.continuous_joint_names == []  # all panda joints are limited


def test_panda_actuator_mapping(panda):
    assert panda.has_actuators
    # arm joints are driven by JOINT-transmission actuators...
    assert panda.joint_name_to_info["joint1"].actuator_id is not None
    # ...the gripper fingers are driven by a tendon, so no direct joint actuator
    assert panda.joint_name_to_info["finger_joint1"].actuator_id is None
