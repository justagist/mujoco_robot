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

from typing import Sequence

import mujoco

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
