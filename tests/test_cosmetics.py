"""Cosmetics / dynamics tests for MujocoRobot: transparency, geom collision, dynamics editing.

Physics only, no rendering.
"""

import mujoco
import numpy as np
import pytest

from mujoco_robot import MujocoRobot, viewer_utils
from mujoco_robot.utils.robot_loader_utils import get_mjcf_from_awesome_robot_descriptions

PANDA = get_mjcf_from_awesome_robot_descriptions("panda_mj_description")
NEUTRAL = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.02, 0.02]


def make(**kw) -> MujocoRobot:
    kw.setdefault("run_async", False)
    kw.setdefault("verbose", False)
    kw.setdefault("place_on_ground", False)
    kw.setdefault("default_joint_positions", NEUTRAL)
    return MujocoRobot(mjcf_path=PANDA, **kw)


def test_transparency_sets_robot_geom_alpha():
    r = make()
    try:
        r.set_robot_transparency(0.3)
        alphas = {round(float(r.model.geom_rgba[g, 3]), 3) for g in r.robot_geom_ids}
        assert alphas == {0.3}
        assert len(r.robot_geom_ids) > 0
    finally:
        r.shutdown()


def test_geom_collision_toggle_saves_and_restores():
    r = make()
    try:
        gid = r.robot_geom_ids[0]
        orig = (int(r.model.geom_contype[gid]), int(r.model.geom_conaffinity[gid]))
        r.set_geom_collision(gid, enabled=False)
        assert int(r.model.geom_contype[gid]) == 0
        assert int(r.model.geom_conaffinity[gid]) == 0
        r.set_geom_collision(gid, enabled=True)
        assert (int(r.model.geom_contype[gid]), int(r.model.geom_conaffinity[gid])) == orig
    finally:
        r.shutdown()


def test_set_dof_fields():
    r = make()
    try:
        r.set_dof_damping("joint1", 0.5)
        r.set_dof_frictionloss("joint1", 0.2)
        r.set_dof_armature("joint1", 0.1)
        adr = r.joint_name_to_info["joint1"].dof_adr
        assert np.isclose(r.model.dof_damping[adr], 0.5)
        assert np.isclose(r.model.dof_frictionloss[adr], 0.2)
        assert np.isclose(r.model.dof_armature[adr], 0.1)
    finally:
        r.shutdown()


def test_set_geom_friction_scalar_and_vector():
    r = make()
    try:
        named = [r.geom_id_to_name[g] for g in r.robot_geom_ids if r.geom_id_to_name[g]]
        if not named:
            pytest.skip("no named robot geoms on this model")
        name = named[0]
        gid = r.geom_name_to_id[name]
        r.set_geom_friction(name, 1.5)  # scalar -> sliding only
        assert np.isclose(r.model.geom_friction[gid, 0], 1.5)
        r.set_geom_friction(name, [0.8, 0.02, 0.001])
        assert np.allclose(r.model.geom_friction[gid], [0.8, 0.02, 0.001])
    finally:
        r.shutdown()


def test_set_body_mass_scales_inertia_and_subtree():
    r = make()
    try:
        bid = r.body_name_to_id["link1"]
        m0 = float(r.model.body_mass[bid])
        inertia0 = r.model.body_inertia[bid].copy()
        subtree0 = float(r.model.body_subtreemass[bid])
        r.set_body_mass("link1", 2.0 * m0)
        assert np.isclose(r.model.body_mass[bid], 2.0 * m0)
        assert np.allclose(r.model.body_inertia[bid], 2.0 * inertia0)  # scaled to stay consistent
        assert np.isclose(r.model.body_subtreemass[bid], subtree0 + m0)  # delta propagated
    finally:
        r.shutdown()


def test_set_body_mass_no_inertia_scale():
    r = make()
    try:
        bid = r.body_name_to_id["link1"]
        m0 = float(r.model.body_mass[bid])
        inertia0 = r.model.body_inertia[bid].copy()
        r.set_body_mass("link1", 2.0 * m0, scale_inertia=False)
        assert np.allclose(r.model.body_inertia[bid], inertia0)  # left untouched
    finally:
        r.shutdown()


def test_set_total_mass():
    r = make()
    try:
        r.set_total_mass(10.0)
        total = sum(float(r.model.body_mass[i]) for i in r.link_ids)
        assert np.isclose(total, 10.0, atol=1e-6)
    finally:
        r.shutdown()


def test_randomize_dynamics_batch():
    r = make()
    try:
        bid = r.body_name_to_id["link1"]
        m0 = float(r.model.body_mass[bid])
        r.randomize_dynamics(
            dof_damping={"joint1": 0.3, "joint2": 0.4},
            body_mass={"link1": 2.0 * m0},
        )
        assert np.isclose(r.model.dof_damping[r.joint_name_to_info["joint1"].dof_adr], 0.3)
        assert np.isclose(r.model.dof_damping[r.joint_name_to_info["joint2"].dof_adr], 0.4)
        assert np.isclose(r.model.body_mass[bid], 2.0 * m0)
    finally:
        r.shutdown()


def test_dynamics_setters_reject_non_robot_elements():
    r = make()  # load_ground_plane=True by default, so a "ground" geom exists (not the robot's)
    try:
        with pytest.raises(ValueError):
            r.set_geom_friction("ground", 1.0)
        with pytest.raises(ValueError):
            r.set_dof_damping("not_a_joint", 1.0)
        with pytest.raises(ValueError):
            r.set_body_mass("not_a_body", 1.0)
    finally:
        r.shutdown()
