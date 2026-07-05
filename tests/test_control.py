"""Control tests for MujocoRobot: position control (ctrl path), torque control (qfrc path),
mode switching, and the PVT-PD command. Physics only, no rendering.
"""

import mujoco
import numpy as np

from mujoco_robot import MujocoRobot
from mujoco_robot.utils.robot_loader_utils import get_mjcf_from_awesome_robot_descriptions

PANDA = get_mjcf_from_awesome_robot_descriptions("panda_mj_description")
# A valid within-limits Panda configuration (7 arm joints + 2 fingers).
NEUTRAL = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.02, 0.02]
ARM = [f"joint{i}" for i in range(1, 8)]
_ACTUATION = int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)


def make(**kw) -> MujocoRobot:
    kw.setdefault("run_async", False)
    kw.setdefault("verbose", False)
    kw.setdefault("place_on_ground", False)
    kw.setdefault("load_ground_plane", False)
    kw.setdefault("default_joint_positions", NEUTRAL)
    return MujocoRobot(mjcf_path=PANDA, **kw)


def test_position_control_converges():
    """The milestone: position commands drive the arm to the target and hold it."""
    r = make()
    try:
        r.set_position_control_mode()
        target = np.array(NEUTRAL[:7]) + 0.15
        for _ in range(2000):
            r.set_joint_positions(target, actuated_joint_names=ARM)
            r.step()
        reached = r.get_actuated_joint_positions(ARM)
        assert np.allclose(reached, target, atol=0.03)
    finally:
        r.shutdown()


def test_mode_switch_flags():
    r = make()
    try:
        r.set_torque_control_mode()
        assert r.in_torque_mode
        assert r.model.opt.disableflags & _ACTUATION  # actuation disabled
        r.set_position_control_mode()
        assert not r.in_torque_mode
        assert not (r.model.opt.disableflags & _ACTUATION)  # actuation re-enabled
    finally:
        r.shutdown()


def test_torque_mode_falls_without_command():
    r = make(enable_torque_mode=True)
    try:
        assert r.in_torque_mode
        for _ in range(300):
            r.step()  # no torques commanded -> gravity pulls the arm down
        drift = np.abs(r.get_actuated_joint_positions() - NEUTRAL)
        assert np.max(drift) > 0.1
    finally:
        r.shutdown()


def test_gravity_compensation_holds():
    r = make(enable_torque_mode=True)
    try:
        for _ in range(500):
            r.set_joint_torques(r.get_gravity_compensation_torques())
            r.step()
        drift = np.abs(r.get_actuated_joint_positions() - NEUTRAL)
        assert np.max(drift) < 0.1  # feedforward gravity comp keeps it roughly static
    finally:
        r.shutdown()


def test_set_joint_torques_writes_qfrc():
    r = make(enable_torque_mode=True)
    try:
        tau = np.linspace(1.0, 9.0, 9)
        r.set_joint_torques(tau)
        applied = np.array(
            [r.data.qfrc_applied[r.joint_name_to_info[n].dof_adr] for n in r.actuated_joint_names]
        )
        assert np.allclose(applied, tau)
    finally:
        r.shutdown()


def test_set_actuated_joint_commands_feedforward_torque():
    r = make(enable_torque_mode=True)
    try:
        tau = np.full(9, 2.5)
        r.set_actuated_joint_commands(tau=tau)  # Kp=Kd=0 -> pure feedforward
        applied = np.array(
            [r.data.qfrc_applied[r.joint_name_to_info[n].dof_adr] for n in r.actuated_joint_names]
        )
        assert np.allclose(applied, tau)
    finally:
        r.shutdown()


def test_positions_delta_updates_ctrl():
    r = make()
    try:
        r.set_position_control_mode()
        before = r.get_actuated_joint_positions(ARM)
        delta = np.full(7, 0.1)
        r.set_joint_positions_delta(delta, actuated_joint_names=ARM)
        ctrl = np.array(
            [r.data.ctrl[r.joint_name_to_info[n].actuator_id] for n in ARM]
        )
        assert np.allclose(ctrl, before + delta, atol=1e-6)
    finally:
        r.shutdown()
