"""Lifecycle tests for MujocoRobot: loading, base handling, stepping, resets, threading.

Physics only, no rendering (so no OpenGL is required).
"""

import time

import mujoco
import numpy as np

from mujoco_robot import MujocoRobot
from mujoco_robot.utils.robot_loader_utils import get_mjcf_from_awesome_robot_descriptions

PANDA = get_mjcf_from_awesome_robot_descriptions("panda_mj_description")


def make(**kw) -> MujocoRobot:
    """Construct a Panda with quiet, synchronous, world-free defaults for testing."""
    kw.setdefault("run_async", False)
    kw.setdefault("verbose", False)
    kw.setdefault("place_on_ground", False)
    kw.setdefault("load_ground_plane", False)
    return MujocoRobot(mjcf_path=PANDA, **kw)


def test_load_fixed_base():
    r = make()
    try:
        assert r.num_actuated_joints == 9
        assert not r.has_floating_base
        assert isinstance(r.data, mujoco.MjData)
        assert r.get_timestep() > 0
    finally:
        r.shutdown()


def test_floating_base_is_structural():
    r = make(use_fixed_base=False)
    try:
        assert r.has_floating_base
        assert r.model.nq == 9 + 7  # free joint adds 7 qpos
        assert r.model.nv == 9 + 6  # free joint adds 6 dof
    finally:
        r.shutdown()


def test_default_joint_positions_applied():
    q = np.linspace(-0.3, 0.3, 9)
    r = make(default_joint_positions=q)
    try:
        got = np.array(
            [r.data.qpos[r.joint_name_to_info[n].qpos_adr] for n in r.actuated_joint_names]
        )
        assert np.allclose(got, q)
    finally:
        r.shutdown()


def test_step_advances_time():
    r = make()
    try:
        t0 = r.data.time
        for _ in range(5):
            r.step()
        assert np.isclose(r.data.time, t0 + 5 * r.get_timestep())
    finally:
        r.shutdown()


def test_gravity_get_set():
    r = make()
    try:
        assert np.allclose(r.get_gravity(), [0.0, 0.0, -9.81])
        r.set_gravity([0.0, 0.0, -1.0])
        assert np.allclose(r.get_gravity(), [0.0, 0.0, -1.0])
    finally:
        r.shutdown()


def test_ground_plane_added():
    r = make(load_ground_plane=True)
    try:
        plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
        assert any(int(r.model.geom_type[i]) == plane for i in range(r.model.ngeom))
    finally:
        r.shutdown()


def test_reset_base_pose_floating():
    r = make(use_fixed_base=False)
    try:
        pos = np.array([0.1, 0.2, 0.5])
        r.reset_base_pose(pos, np.array([0.0, 0.0, 0.0, 1.0]))
        adr = r.joint_name_to_info[r.joint_id_to_name[r.free_joint_id]].qpos_adr
        assert np.allclose(r.data.qpos[adr : adr + 3], pos)
        assert np.allclose(r.data.qpos[adr + 3 : adr + 7], [1.0, 0.0, 0.0, 0.0])  # wxyz identity
    finally:
        r.shutdown()


def test_place_on_ground_clears_contact():
    r = make(load_ground_plane=True, place_on_ground=True, use_fixed_base=False)
    try:
        assert r.default_start_pose[0][2] >= 0.0
        assert not r._touching_ground(r._ground_geom_ids())
    finally:
        r.shutdown()


def test_async_thread_runs_then_shuts_down():
    r = make(run_async=True)
    try:
        assert r._running
        time.sleep(0.1)
        assert r.data.time > 0.0  # background thread advanced the sim
    finally:
        r.shutdown()
    assert not r._running
    assert r._async_thread is None


def test_wrap_existing_model():
    model = mujoco.MjModel.from_xml_path(PANDA)
    data = mujoco.MjData(model)
    r = MujocoRobot(model=model, data=data, run_async=False, verbose=False, place_on_ground=False)
    try:
        assert r.model is model
        assert r.data is data
        assert r.num_actuated_joints == 9
    finally:
        r.shutdown()
