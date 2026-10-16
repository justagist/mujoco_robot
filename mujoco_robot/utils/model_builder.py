"""World/model construction helpers built on MuJoCo's ``MjSpec`` API.

A bare robot description (MJCF/URDF from robot_descriptions) usually has no world: no ground
plane and no light, and its base is fixed or floating depending on whether the root body has a
free joint. These helpers load a description, set the base to fixed/floating, optionally wrap it
in a minimal world (ground plane + light), and compile to a fresh ``MjModel``.

NOTE: A fresh ``mujoco.MjSpec`` compiler defaults to degrees, but a spec loaded from a file keeps
that file's setting, and the helpers here only emit unit-free quantities (positions, sizes), no
angle literals. If you add angle literals (euler / joint range), set ``spec.compiler.degree =
False`` first so they are treated as radians, matching the radian public API.
"""

from typing import List, Sequence

import mujoco
import numpy as np

_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
_PLANE = int(mujoco.mjtGeom.mjGEOM_PLANE)


def load_spec(mjcf_path: str = None, urdf_path: str = None) -> mujoco.MjSpec:
    """Load an MJCF or URDF file into an editable ``MjSpec``."""
    path = mjcf_path if mjcf_path is not None else urdf_path
    if path is None:
        raise ValueError("Provide mjcf_path or urdf_path.")
    return mujoco.MjSpec.from_file(path)


def get_root_body(spec: mujoco.MjSpec):
    """Return the robot's root body (the first child of the world body)."""
    children = spec.worldbody.bodies
    if not children:
        raise ValueError("Model has no body under the worldbody.")
    return children[0]


def set_base_fixed(spec: mujoco.MjSpec, fixed: bool):
    """Make the robot base fixed (no free joint) or floating (a free joint on the root body).

    Args:
        spec (mujoco.MjSpec): The editable model spec.
        fixed (bool): If True, remove any free joint on the root body. If False, add one if the
            root body does not already have a free joint.
    """
    root = get_root_body(spec)
    free_joints = [j for j in root.joints if int(j.type) == _FREE]
    if fixed:
        for j in free_joints:
            spec.delete(j)
    elif not free_joints:
        root.add_freejoint()


def add_ground_plane(spec: mujoco.MjSpec, name: str = "ground"):
    """Add an infinite ground plane at z=0 if the world does not already have one."""
    world = spec.worldbody
    if any(int(g.type) == _PLANE for g in world.geoms):
        return
    geom = world.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_PLANE
    geom.size = [0.0, 0.0, 0.05]  # infinite plane; third value is the visual grid spacing


def add_light(spec: mujoco.MjSpec):
    """Add a light above the scene if the world does not already have one."""
    world = spec.worldbody
    if world.lights:
        return
    light = world.add_light()
    light.pos = [0.0, 0.0, 3.0]
    light.dir = [0.0, 0.0, -1.0]


def add_box(spec: mujoco.MjSpec, name, pos, half_size, rgba=(0.6, 0.4, 0.25, 1.0)):
    """Add a static (world-fixed) box geom to the world, e.g. a table or an obstacle.

    Args:
        spec (mujoco.MjSpec): The editable model spec.
        name (str): Geom name.
        pos (Sequence[float]): World-frame centre position [x, y, z].
        half_size (Sequence[float]): Box half-extents [x, y, z].
        rgba (Sequence[float], optional): Colour. Defaults to a brown.
    """
    geom = spec.worldbody.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.pos = list(pos)
    geom.size = list(half_size)
    geom.rgba = list(rgba)


def add_site(spec: mujoco.MjSpec, body_name: str, site_name: str, pos=(0.0, 0.0, 0.0)):
    """Attach a named site (a frame) to a body, e.g. an IK / tool target frame.

    Args:
        spec (mujoco.MjSpec): The editable model spec.
        body_name (str): Body to attach the site to.
        site_name (str): Name for the new site.
        pos (Sequence[float], optional): Site position in the body frame. Defaults to the origin.

    Returns:
        str: The site name.
    """
    body = next((b for b in spec.bodies if b.name == body_name), None)
    if body is None:
        raise ValueError(f"No body named '{body_name}' to attach a site to.")
    site = body.add_site()
    site.name = site_name
    site.pos = list(pos)
    return site_name


def add_ft_sensors(spec: mujoco.MjSpec, link_names: Sequence[str]):
    """Attach a force + torque sensor (on a new site) to each named body.

    MuJoCo sensors are compile-time, so force-torque sensing must be declared before compilation.
    Each link ``L`` gets a site ``L_ft_site`` and sensors ``L_force`` / ``L_torque`` that measure
    the interaction wrench transmitted through that site (in the site frame).

    Args:
        spec (mujoco.MjSpec): The editable model spec.
        link_names (Sequence[str]): Names of bodies to instrument.
    """
    bodies = {b.name: b for b in spec.bodies}
    for link in link_names:
        if link not in bodies:
            raise ValueError(f"No body named '{link}' to attach a force-torque sensor to.")
        site_name = f"{link}_ft_site"
        site = bodies[link].add_site()
        site.name = site_name
        for stype, suffix in (
            (mujoco.mjtSensor.mjSENS_FORCE, "force"),
            (mujoco.mjtSensor.mjSENS_TORQUE, "torque"),
        ):
            sensor = spec.add_sensor()
            sensor.name = f"{link}_{suffix}"
            sensor.type = stype
            sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
            sensor.objname = site_name


def add_position_actuators(
    spec: mujoco.MjSpec,
    joint_names: Sequence[str],
    kp,
    kv,
    ctrl_from_joint_range: bool = True,
    prefix: str = "",
) -> List[str]:
    """Add a position servo (``general`` actuator) on each named hinge/slide joint.

    Each actuator applies ``kp * (ctrl - q) - kv * qdot``: ``gaintype=FIXED`` with
    ``gainprm[0] = kp``, and ``biastype=AFFINE`` with ``biasprm[1] = -kp`` and
    ``biasprm[2] = -kv``. Use this for descriptions that ship no actuators (most URDFs), so the
    robot is position-controlled through ``data.ctrl`` instead of a PD loop on
    ``qfrc_applied``.

    NOTE: ``MjSpec`` actuators default to ``biastype=NONE``. Setting ``biasprm`` without also
    setting ``biastype=AFFINE`` compiles without complaint but silently yields a pure gain
    (force ``kp * ctrl``), not a position servo.

    Args:
        spec (mujoco.MjSpec): The editable model spec.
        joint_names (Sequence[str]): Joints to actuate, as named in the robot's description.
        kp (float | Sequence[float]): Position gain(s), scalar or one per joint.
        kv (float | Sequence[float]): Velocity (damping) gain(s), scalar or one per joint.
        ctrl_from_joint_range (bool, optional): Limit each actuator's ``ctrlrange`` to its
            joint's range (unlimited joints stay unlimited). The compiler converts the range, so
            this is correct whatever the spec's angle unit. Defaults to True.
        prefix (str, optional): Prepended to each joint name to find it in ``spec``, e.g. the
            prefix a robot was attached under with :func:`attach_robot`. Defaults to "".

    Returns:
        List[str]: The new actuator names, ``f"{prefix}{joint}_pos"``, in ``joint_names`` order.
    """
    joint_names = list(joint_names)
    kps = np.broadcast_to(np.asarray(kp, dtype=float), (len(joint_names),))
    kvs = np.broadcast_to(np.asarray(kv, dtype=float), (len(joint_names),))
    joints = {j.name: j for j in spec.joints}
    one_dof = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    names = []
    for name, kp_i, kv_i in zip(joint_names, kps, kvs):
        full_name = f"{prefix}{name}"
        joint = joints.get(full_name)
        if joint is None:
            raise ValueError(f"No joint named '{full_name}' to add an actuator to.")
        if int(joint.type) not in one_dof:
            raise ValueError(f"Joint '{full_name}' is not a hinge or slide joint.")
        act = spec.add_actuator()
        act.name = f"{full_name}_pos"
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.target = full_name
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = kp_i
        act.biasprm[1] = -kp_i
        act.biasprm[2] = -kv_i
        if ctrl_from_joint_range:
            # ctrlrange = joint range, converted by the compiler; ignored for unlimited joints.
            act.inheritrange = 1.0
        names.append(act.name)
    return names


def attach_robot(
    world_spec: mujoco.MjSpec,
    robot_spec: mujoco.MjSpec,
    prefix: str,
    pos: Sequence[float] = None,
    quat_xyzw: Sequence[float] = None,
):
    """Attach a loaded robot spec into a world spec, with its base at the given pose.

    The robot's root body (and its actuators, sensors and other elements) is copied under a new
    frame of the world body, with every name prefixed by ``prefix``. Wrap the result with
    ``MujocoRobot(model=..., data=..., joint_names=[f"{prefix}{j}" for j in ...])``.

    Args:
        world_spec (mujoco.MjSpec): The world to attach into (modified in place).
        robot_spec (mujoco.MjSpec): The robot, e.g. from :func:`load_spec`.
        prefix (str): Name prefix for every attached element, e.g. ``"arm/"``.
        pos (Sequence[float], optional): Base position in the world. Defaults to the origin.
        quat_xyzw (Sequence[float], optional): Base orientation quaternion [x, y, z, w].
            Defaults to identity.

    Returns:
        mujoco.MjsBody: The attached root body (in ``world_spec``).
    """
    frame = world_spec.worldbody.add_frame()
    if pos is not None:
        frame.pos = list(pos)
    if quat_xyzw is not None:
        x, y, z, w = (float(v) for v in quat_xyzw)
        frame.quat = [w, x, y, z]  # MuJoCo order
    return frame.attach_body(get_root_body(robot_spec), prefix, "")


def build_model(
    mjcf_path: str = None,
    urdf_path: str = None,
    use_fixed_base: bool = True,
    load_ground_plane: bool = True,
    ft_sensor_links: Sequence[str] = None,
) -> mujoco.MjModel:
    """Load a description, set the base, optionally add a ground plane + FT sensors, and compile.

    Args:
        mjcf_path (str): Path to an MJCF file (preferred).
        urdf_path (str): Path to a URDF file (best-effort; MuJoCo URDF has no actuators/sensors).
        use_fixed_base (bool): Whether the robot base is fixed (True) or floating (False).
        load_ground_plane (bool): Whether to add a ground plane + light.
        ft_sensor_links (Sequence[str], optional): Bodies to instrument with force-torque sensors.

    Returns:
        mujoco.MjModel: The compiled model.
    """
    spec = load_spec(mjcf_path=mjcf_path, urdf_path=urdf_path)
    set_base_fixed(spec, use_fixed_base)
    if ft_sensor_links:
        add_ft_sensors(spec, ft_sensor_links)
    if load_ground_plane:
        add_ground_plane(spec)
        add_light(spec)
    return spec.compile()
