"""Debug-draw helpers for the passive MuJoCo viewer.

These draw markers into ``viewer.user_scn`` (the user scene of a ``mujoco.viewer`` passive
viewer). Typical use per frame: call :func:`clear_user_geoms`, add markers, then let the viewer
sync (``MujocoRobot.step`` syncs for you). No OpenGL calls are made here.
"""

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


def clear_user_geoms(viewer):
    """Reset the user scene so markers can be redrawn from scratch for this frame."""
    viewer.user_scn.ngeom = 0


def render_point(viewer, pos, size: float = 0.02, rgba=(1.0, 1.0, 0.0, 1.0)):
    """Draw a sphere marker at a world-frame position.

    Args:
        viewer: A ``mujoco.viewer`` passive viewer handle.
        pos: World-frame position ``[x, y, z]``.
        size (float): Sphere radius.
        rgba: Colour.
    """
    scene = viewer.user_scn
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, size, dtype=float),
        np.asarray(pos, dtype=float),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def render_frame(viewer, pos, quat, scale: float = 0.1, alpha: float = 1.0):
    """Draw a coordinate frame (x=red, y=green, z=blue arrows) at a world-frame pose.

    Args:
        viewer: A ``mujoco.viewer`` passive viewer handle.
        pos: World-frame origin ``[x, y, z]``.
        quat: Orientation quaternion ``[x, y, z, w]``.
        scale (float): Axis-arrow length (metres).
        alpha (float): Arrow opacity.
    """
    scene = viewer.user_scn
    rot = Rotation.from_quat(quat).as_matrix()
    origin = np.asarray(pos, dtype=float)
    colours = (
        (1.0, 0.0, 0.0, alpha),
        (0.0, 1.0, 0.0, alpha),
        (0.0, 0.0, 1.0, alpha),
    )
    for axis in range(3):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            np.asarray(colours[axis], dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.005, origin, origin + scale * rot[:, axis]
        )
        scene.ngeom += 1
