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


def build_model(
    mjcf_path: str = None,
    urdf_path: str = None,
    use_fixed_base: bool = True,
    load_ground_plane: bool = True,
) -> mujoco.MjModel:
    """Load a description, set the base, optionally add a ground plane + light, and compile.

    Args:
        mjcf_path (str): Path to an MJCF file (preferred).
        urdf_path (str): Path to a URDF file (best-effort; MuJoCo URDF has no actuators/sensors).
        use_fixed_base (bool): Whether the robot base is fixed (True) or floating (False).
        load_ground_plane (bool): Whether to add a ground plane + light.

    Returns:
        mujoco.MjModel: The compiled model.
    """
    spec = load_spec(mjcf_path=mjcf_path, urdf_path=urdf_path)
    set_base_fixed(spec, use_fixed_base)
    if load_ground_plane:
        add_ground_plane(spec)
        add_light(spec)
    return spec.compile()
