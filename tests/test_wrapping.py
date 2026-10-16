"""Wrapping tests: one scoped robot inside a larger, host-stepped world. Physics only, no rendering.

The world is built in-test with MjSpec: a two-link hinge arm attached under prefix ``arm/`` (with
position actuators from ``add_position_actuators``), a slide joint ``pusher`` with its own
actuator, a hinge ``gate`` with no actuator, and three free boxes resting on the ground.
"""

import importlib
import sys

import mujoco
import numpy as np
import pytest

from mujoco_robot import MujocoRobot
from mujoco_robot.ik import differential_ik
from mujoco_robot.mujoco_robot import MujocoObject
from mujoco_robot.utils import model_builder as mb

ARM_JOINTS = ["arm/j1", "arm/j2"]
BOXES = ["box0", "box1", "box2"]
ARM_KP, ARM_KV = [500.0, 300.0], [30.0, 20.0]
_HINGE = mujoco.mjtJoint.mjJNT_HINGE
_SLIDE = mujoco.mjtJoint.mjJNT_SLIDE
_BOX = mujoco.mjtGeom.mjGEOM_BOX
_CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE


def arm_spec(floating: bool = False) -> mujoco.MjSpec:
    """A two-link planar arm (hinges about y) with a tool site and position actuators."""
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    base = spec.worldbody.add_body(name="base")
    if floating:
        base.add_freejoint(name="root")
    base.add_geom(type=_BOX, size=[0.05, 0.05, 0.05], mass=2.0)
    link1 = base.add_body(name="link1", pos=[0, 0, 0.05])
    link1.add_joint(name="j1", type=_HINGE, axis=[0, 1, 0], range=[-2.5, 2.5])
    link1.add_geom(type=_CAPSULE, fromto=[0, 0, 0, 0, 0, 0.3], size=[0.02, 0, 0], mass=1.0)
    link2 = link1.add_body(name="link2", pos=[0, 0, 0.3])
    link2.add_joint(name="j2", type=_HINGE, axis=[0, 1, 0], range=[-2.5, 2.5])
    link2.add_geom(type=_CAPSULE, fromto=[0, 0, 0, 0, 0, 0.3], size=[0.02, 0, 0], mass=1.0)
    link2.add_site(name="tcp", pos=[0, 0, 0.3])
    spec.add_exclude(bodyname1="base", bodyname2="link1")  # link1's capsule starts in the base
    mb.add_position_actuators(spec, ["j1", "j2"], kp=ARM_KP, kv=ARM_KV)
    return spec


def build_world(base_pos=(0.0, 0.0, 0.5), base_quat_xyzw=None, floating=False):
    """Compile the cell world and let the free boxes settle (as a host would before handing over)."""
    world = mujoco.MjSpec()
    world.compiler.degree = False
    world.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # stiff servos
    mb.add_ground_plane(world)
    mb.add_light(world)
    mb.attach_robot(world, arm_spec(floating), "arm/", pos=base_pos, quat_xyzw=base_quat_xyzw)

    pusher = world.worldbody.add_body(name="pusher_body", pos=[1.0, 0.0, 0.5])
    pusher.add_joint(name="pusher", type=_SLIDE, axis=[1, 0, 0], range=[0.0, 0.3])
    pusher.add_geom(type=_BOX, size=[0.03, 0.1, 0.03], mass=0.5)
    mb.add_position_actuators(world, ["pusher"], kp=200.0, kv=20.0)

    gate = world.worldbody.add_body(name="gate_body", pos=[0.0, 1.0, 0.5])
    gate.add_joint(name="gate", type=_HINGE, axis=[0, 0, 1], range=[-1.5, 1.5], armature=0.5)
    gate.add_geom(type=_BOX, size=[0.1, 0.01, 0.05], pos=[0.1, 0, 0], mass=0.3)

    for i, name in enumerate(BOXES):
        box = world.worldbody.add_body(name=name, pos=[-1.0, 0.4 * (i - 1), 0.05])
        box.add_freejoint(name=f"{name}_free")
        box.add_geom(type=_BOX, size=[0.05, 0.05, 0.05], mass=0.2)

    model = world.compile()
    data = mujoco.MjData(model)
    for _ in range(300):
        mujoco.mj_step(model, data)
    return model, data


def wrap(model, data, **kw) -> MujocoRobot:
    kw.setdefault("joint_names", ARM_JOINTS)
    kw.setdefault("verbose", False)
    return MujocoRobot(model=model, data=data, **kw)


def qadr(model, joint: str) -> int:
    return int(model.jnt_qposadr[model.joint(joint).id])


def box_qpos(model, data) -> np.ndarray:
    return np.concatenate(
        [data.qpos[qadr(model, f"{b}_free") : qadr(model, f"{b}_free") + 7] for b in BOXES]
    )


def assert_same_rotation(q, expected):
    assert np.allclose(q, expected) or np.allclose(q, -expected)


def test_scoped_construction_changes_nothing():
    model, data = build_world()
    # Give the state and every control input distinctive host-set values so any write would show.
    data.qvel[:2] = [0.05, -0.05]  # the arm is moving
    data.ctrl[:] = [0.1, -0.2, 0.05]
    data.qfrc_applied[:] = np.linspace(0.1, 1.0, model.nv)
    before = {k: np.array(getattr(data, k)) for k in ("qpos", "qvel", "ctrl", "qfrc_applied")}
    flags = int(model.opt.disableflags)

    r = wrap(model, data)  # run_async / place_on_ground defaults apply
    try:
        assert r.num_actuated_joints == 2
        assert r.actuated_joint_names == ARM_JOINTS
        assert r.has_floating_base is False
        assert r.base_name == "arm/base"  # inferred: common ancestor of the joints
        assert not r._running  # wrapping default: no background stepping thread
        for key, value in before.items():
            np.testing.assert_array_equal(getattr(data, key), value, err_msg=key)
        assert int(model.opt.disableflags) == flags
    finally:
        r.shutdown()


def test_position_commands_write_only_own_ctrl_and_track():
    model, data = build_world()
    r = wrap(model, data)
    try:
        # Actuators are matched by transmission; their names differ from the joint names.
        own = [r.joint_name_to_info[j].actuator_id for j in ARM_JOINTS]
        assert [r.actuator_id_to_name[a] for a in own] == ["arm/j1_pos", "arm/j2_pos"]

        ctrl0, qfrc0 = np.array(data.ctrl), np.array(data.qfrc_applied)
        pusher0, gate0 = data.qpos[qadr(model, "pusher")], data.qpos[qadr(model, "gate")]
        boxes0 = box_qpos(model, data)

        target = np.array([0.4, -0.3])
        r.set_joint_positions(target)
        assert np.flatnonzero(data.ctrl != ctrl0).tolist() == own
        np.testing.assert_array_equal(data.ctrl[own], target)
        np.testing.assert_array_equal(data.qfrc_applied, qfrc0)

        for _ in range(500):
            mujoco.mj_step(model, data)  # the host steps

        assert np.allclose(r.get_actuated_joint_positions(), target, atol=0.03)
        assert np.isclose(data.qpos[qadr(model, "pusher")], pusher0, atol=1e-6)
        assert np.isclose(data.qpos[qadr(model, "gate")], gate0, atol=1e-6)
        assert np.allclose(box_qpos(model, data), boxes0, atol=1e-3)
    finally:
        r.shutdown()


def test_foreign_joints_are_never_written():
    model, data = build_world()
    r = wrap(model, data)
    try:
        with pytest.raises(ValueError):
            r.set_joint_positions([0.1], actuated_joint_names=["pusher"])
        with pytest.raises(ValueError):
            r.set_joint_torques([1.0], actuated_joint_names=["gate"])
        with pytest.raises(ValueError):
            r.reset_joints([model.joint("gate").id], [0.2])
    finally:
        r.shutdown()


def test_unactuated_joint_needs_managed_pd():
    model, data = build_world()
    r = wrap(model, data, joint_names=["gate"])
    try:
        with pytest.raises(ValueError, match="managed_pd"):
            r.set_joint_positions([0.5])
    finally:
        r.shutdown()

    r = wrap(model, data, joint_names=["gate"], managed_pd=True)
    try:
        qfrc0 = np.array(data.qfrc_applied)
        r.set_joint_positions([0.5])
        for _ in range(1000):
            r.apply_managed_control()  # the host runs the robot's PD in its pre-step logic
            mujoco.mj_step(model, data)
        assert np.isclose(data.qpos[qadr(model, "gate")], 0.5, atol=0.05)
        changed = np.flatnonzero(data.qfrc_applied != qfrc0).tolist()
        assert changed == r.actuated_joint_dof_adrs
    finally:
        r.shutdown()


def test_scoped_torque_mode_disables_only_own_actuators():
    model, data = build_world()
    pusher_act = model.actuator("pusher_pos").id
    data.ctrl[pusher_act] = 0.1  # host keeps driving its pusher
    gate_dof = int(model.jnt_dofadr[model.joint("gate").id])
    data.qfrc_applied[gate_dof] = 0.3  # and applies its own force to the gate
    limits0 = (np.array(model.actuator_forcelimited), np.array(model.actuator_forcerange))
    flags = int(model.opt.disableflags)
    r = wrap(model, data)
    try:
        r.set_torque_control_mode()
        assert int(model.opt.disableflags) == flags  # no model-wide actuation switch
        assert data.ctrl[pusher_act] == 0.1
        r.set_joint_torques([1.0, -2.0])
        mujoco.mj_forward(model, data)
        np.testing.assert_array_equal(data.actuator_force[r.robot_actuator_ids], 0.0)
        assert abs(data.actuator_force[pusher_act]) > 0.0
        expected = np.zeros(model.nv)
        expected[gate_dof] = 0.3
        expected[r.actuated_joint_dof_adrs] = [1.0, -2.0]
        np.testing.assert_array_equal(data.qfrc_applied, expected)

        r.set_position_control_mode()
        np.testing.assert_array_equal(model.actuator_forcelimited, limits0[0])
        np.testing.assert_array_equal(model.actuator_forcerange, limits0[1])
    finally:
        r.shutdown()


def test_root_body_overrides_inferred_root():
    quat = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])  # 90 deg about z
    model, data = build_world(base_pos=(0.2, -0.1, 0.4), base_quat_xyzw=quat)
    # Scoped to j2 alone, the inferred root is link1 (the body j2 moves relative to).
    assert MujocoObject(model, joint_names=["arm/j2"]).base_name == "arm/link1"
    r = wrap(model, data, joint_names=["arm/j2"], root_body="arm/base")
    try:
        assert r.base_name == "arm/base"
        assert not r.has_floating_base
        pos, q = r.get_base_pose()
        assert np.allclose(pos, [0.2, -0.1, 0.4])
        assert_same_rotation(q, quat)
    finally:
        r.shutdown()


def test_reset_base_pose_under_a_nested_parent():
    """A fixed root inside a host body: the requested world pose is converted to the parent frame."""
    world = mujoco.MjSpec()
    world.compiler.degree = False
    cell = world.worldbody.add_body(name="cell", pos=[1.0, 0.5, 0.2], quat=[0.0, 0.0, 0.0, 1.0])
    arm = arm_spec()  # keep the child spec alive while its body is being attached
    cell.add_frame().attach_body(mb.get_root_body(arm), "arm/", "")
    model = world.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    r = wrap(model, data)
    try:
        target = np.array([0.3, 0.1, 0.6])
        quat = np.array([0.0, 0.0, np.sin(np.pi / 8), np.cos(np.pi / 8)])
        r.reset_base_pose(target, quat)
        pos, q = r.get_base_pose()
        assert np.allclose(pos, target)
        assert_same_rotation(q, quat)
    finally:
        r.shutdown()


def test_floating_root_ignores_free_boxes():
    model, data = build_world(floating=True)
    r = wrap(model, data)
    try:
        assert r.has_floating_base is True
        assert r.free_joint_id == model.joint("arm/root").id
        assert set(r.link_names) == {"arm/base", "arm/link1", "arm/link2"}
        adr = qadr(model, "arm/root")
        mujoco.mj_forward(model, data)  # after mj_step, xpos lags qpos by one integration
        pos, _ = r.get_base_pose()
        assert np.allclose(pos, data.qpos[adr : adr + 3])
    finally:
        r.shutdown()


def test_unscoped_base_is_root_not_last_free_joint():
    """Without scoping, an unrelated free body later in the model is not the floating base."""
    model, _ = build_world()
    obj = MujocoObject(model)
    assert obj.base_name == "arm/base"
    assert obj.has_floating_base is False


def test_jacobian_link_pose_and_contacts_on_scoped_links():
    model, data = build_world(base_pos=(0.0, 0.0, 0.3))
    r = wrap(model, data)
    try:
        adrs = r.actuated_joint_qpos_adrs
        # Poses from known geometry: upright arm, link2 sits 0.05 + 0.3 above the base and the
        # tcp a further 0.3. Names and ids must give the same frames.
        data.qpos[adrs] = [0.0, 0.0]
        mujoco.mj_forward(model, data)
        link2 = r.get_link_id("arm/link2")
        for link in (link2, "arm/link2"):
            np.testing.assert_allclose(r.get_link_pose(link)[0], [0.0, 0.0, 0.65], atol=1e-12)
        for site in (model.site("arm/tcp").id, "arm/tcp"):
            np.testing.assert_allclose(r.get_site_pose(site)[0], [0.0, 0.0, 0.95], atol=1e-12)
        # link1's capsule runs 0.3 up from its origin (0.05 above the base), so its COM is 0.2 up.
        for link in (r.get_link_id("arm/link1"), "arm/link1"):
            np.testing.assert_allclose(r.get_link_com_pose(link)[0], [0.0, 0.0, 0.5], atol=1e-12)

        # Jacobian columns are the two arm joints: check against finite differences of the tcp.
        q0 = np.array([0.3, -0.5])
        data.qpos[adrs] = q0
        mujoco.mj_forward(model, data)
        jac = r.get_jacobian("arm/tcp")
        assert jac.shape == (6, 2)
        scratch, tcp, eps = mujoco.MjData(model), model.site("arm/tcp").id, 1e-6
        for i in range(2):
            ends = []
            for sign in (1.0, -1.0):
                scratch.qpos[:] = data.qpos
                scratch.qpos[adrs[i]] = q0[i] + sign * eps
                mujoco.mj_kinematics(model, scratch)
                ends.append(np.array(scratch.site_xpos[tcp]))
            np.testing.assert_allclose(jac[:3, i], (ends[0] - ends[1]) / (2 * eps), atol=1e-6)
        np.testing.assert_allclose(jac[3:], [[0, 0], [1, 1], [0, 0]], atol=1e-12)  # both about y

        # The host poses the arm so link2 reaches into the ground (link1 stays clear).
        data.qpos[adrs] = [2.0, 1.0]
        mujoco.mj_forward(model, data)
        link1 = r.get_link_id("arm/link1")
        assert r.get_contact_states_of_links([link1, link2]).tolist() == [0, 1]
        assert r.get_contact_states_of_links(["arm/link1", "arm/link2"]).tolist() == [0, 1]
        assert r.get_link_contact_force("arm/link2")[2] > 0.0
    finally:
        r.shutdown()


def test_link_ids_resolve_only_to_robot_links():
    """Link ids follow the same rule as link names: only this robot's own links resolve."""
    model, data = build_world()
    r = wrap(model, data)
    try:
        for bad in (model.body("box0").id, 0, -1, model.nbody):  # foreign, world, negative, range
            with pytest.raises(KeyError):
                r.get_link_pose(bad)
        with pytest.raises(KeyError):
            r.get_link_pose("box0")
        with pytest.raises(KeyError):
            r.get_site_pose(-1)  # negative ids no longer wrap around
    finally:
        r.shutdown()


def test_differential_ik_over_scoped_joints():
    model, data = build_world()
    r = wrap(model, data)
    try:
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = data.qpos
        scratch.qpos[r.actuated_joint_qpos_adrs] = [0.5, 0.6]
        mujoco.mj_forward(model, scratch)
        target = np.array(scratch.site_xpos[model.site("arm/tcp").id])

        q = differential_ik(model, data, "arm/tcp", target, joint_names=r.actuated_joint_names)
        assert q.shape == (2,)
        scratch.qpos[:] = data.qpos
        scratch.qpos[r.actuated_joint_qpos_adrs] = q
        mujoco.mj_forward(model, scratch)
        assert np.allclose(scratch.site_xpos[model.site("arm/tcp").id], target, atol=1e-3)
    finally:
        r.shutdown()


def test_scoped_set_total_mass_leaves_world_alone():
    model, data = build_world()
    r = wrap(model, data)
    try:
        foreign = [b for b in range(model.nbody) if b not in r.link_ids]
        foreign_mass = np.array(model.body_mass[foreign])
        r.set_total_mass(8.0)
        assert np.isclose(sum(model.body_mass[b] for b in r.link_ids), 8.0)
        np.testing.assert_array_equal(model.body_mass[foreign], foreign_mass)
    finally:
        r.shutdown()


def test_add_position_actuators_is_an_affine_servo():
    """Regression for the silent pure-gain trap: biasprm alone, with biastype left at NONE."""
    spec = mujoco.MjSpec()  # compiler.degree defaults to True: the range below is in degrees
    spec.option.gravity = [0.0, 0.0, 0.0]
    link = spec.worldbody.add_body(name="link")
    link.add_joint(name="hinge", type=_HINGE, axis=[0, 0, 1], range=[-90.0, 90.0])
    link.add_geom(type=_CAPSULE, fromto=[0, 0, 0, 0.3, 0, 0], size=[0.02, 0, 0], mass=1.0)
    assert mb.add_position_actuators(spec, ["hinge"], kp=20.0, kv=2.0) == ["hinge_pos"]
    model = spec.compile()
    data = mujoco.MjData(model)

    assert int(model.actuator_biastype[0]) == int(mujoco.mjtBias.mjBIAS_AFFINE)
    np.testing.assert_allclose(model.actuator_ctrlrange[0], [-np.pi / 2, np.pi / 2])
    data.ctrl[0] = 0.5
    for _ in range(3000):
        mujoco.mj_step(model, data)
    assert np.isclose(data.qpos[0], 0.5, atol=1e-2)
    assert abs(data.qvel[0]) < 1e-2

    # Per-joint gains land on the matching actuators.
    arm = arm_spec().compile()
    np.testing.assert_array_equal(arm.actuator_gainprm[:, 0], ARM_KP)
    np.testing.assert_array_equal(arm.actuator_biasprm[:, 1], -np.array(ARM_KP))
    np.testing.assert_array_equal(arm.actuator_biasprm[:, 2], -np.array(ARM_KV))


def test_attach_robot_places_base():
    world = mujoco.MjSpec()
    quat = np.array([0.0, 0.0, np.sin(np.pi / 6), np.cos(np.pi / 6)])  # 60 deg about z
    body = mb.attach_robot(world, arm_spec(), "left/", pos=[0.5, -0.2, 0.8], quat_xyzw=quat)
    assert body.name == "left/base"
    model = world.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bid = model.body("left/base").id
    np.testing.assert_allclose(data.xpos[bid], [0.5, -0.2, 0.8], atol=1e-12)
    assert_same_rotation(np.roll(np.array(data.xquat[bid]), -1), quat)  # wxyz -> xyzw
    assert {model.actuator(i).name for i in range(model.nu)} == {"left/j1_pos", "left/j2_pos"}


def test_core_works_without_robot_descriptions(monkeypatch):
    model, data = build_world()
    for name in list(sys.modules):
        if name == "robot_descriptions" or name.startswith("robot_descriptions."):
            monkeypatch.setitem(sys.modules, name, None)  # None makes `import` raise
        elif name == "mujoco_robot" or name.startswith("mujoco_robot."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "robot_descriptions", None)

    fresh = importlib.import_module("mujoco_robot")
    r = fresh.MujocoRobot(model=model, data=data, joint_names=ARM_JOINTS, verbose=False)
    try:
        assert r.num_actuated_joints == 2
    finally:
        r.shutdown()

    loader = importlib.import_module("mujoco_robot.utils.robot_loader_utils")
    with pytest.raises(ImportError, match=r"mujoco_robot\[descriptions\]"):
        loader.get_mjcf_from_awesome_robot_descriptions("panda_mj_description")
