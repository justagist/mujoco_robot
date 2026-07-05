"""MuJoCo robot interface. This is a stand-alone file.

Implemented purely on ``mujoco.MjModel`` + ``MjData`` (no client-server).

Quaternion convention: everything exposed by this module uses ``[x, y, z, w]`` (scipy convention).
MuJoCo internally uses ``[w, x, y, z]``; conversion happens ONLY at the boundary via
:func:`wxyz_to_xyzw` / :func:`xyzw_to_wxyz`.

Angle convention: all joint quantities exposed here (positions, velocities, limits) are in
radians. A compiled ``MjModel`` always stores angles in radians regardless of the MJCF
``compiler angle`` setting (which only affects how the XML *source* is parsed), so no runtime
conversion is needed or performed.
"""

from typing import Dict, List, Mapping, Optional, TypeAlias
from dataclasses import dataclass

import numpy as np
import mujoco

QuatType: TypeAlias = np.ndarray
"""Numpy array representing a quaternion in format [x,y,z,w] (scipy convention)."""
Vector3D: TypeAlias = np.ndarray
"""Numpy array representing a 3D cartesian vector in format [x,y,z]."""

# qpos / dof widths per joint type. free=7/6, ball=4/3, slide=1/1, hinge=1/1.
_QPOS_DIM: Dict[int, int] = {
    int(mujoco.mjtJoint.mjJNT_FREE): 7,
    int(mujoco.mjtJoint.mjJNT_BALL): 4,
    int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
    int(mujoco.mjtJoint.mjJNT_HINGE): 1,
}
_DOF_DIM: Dict[int, int] = {
    int(mujoco.mjtJoint.mjJNT_FREE): 6,
    int(mujoco.mjtJoint.mjJNT_BALL): 3,
    int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
    int(mujoco.mjtJoint.mjJNT_HINGE): 1,
}


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap the provided angle(s) (radians) to be within -pi and pi.

    Args:
        angle (float | np.ndarray): Input angle (or array of angles) in radians.

    Returns:
        float | np.ndarray: Output after wrapping.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def wxyz_to_xyzw(quat_wxyz: np.ndarray) -> QuatType:
    """Convert a MuJoCo quaternion ``[w,x,y,z]`` to the external ``[x,y,z,w]`` convention.

    This is the single conversion point when reading orientations out of MuJoCo.

    Args:
        quat_wxyz (np.ndarray): Quaternion in MuJoCo order ``[w, x, y, z]``.

    Returns:
        QuatType: Quaternion in ``[x, y, z, w]`` order.
    """
    q = np.asarray(quat_wxyz, dtype=float)
    return np.array([q[1], q[2], q[3], q[0]])


def xyzw_to_wxyz(quat_xyzw: QuatType) -> np.ndarray:
    """Convert an external ``[x,y,z,w]`` quaternion to MuJoCo's ``[w,x,y,z]`` convention.

    This is the single conversion point when writing orientations into MuJoCo.

    Args:
        quat_xyzw (QuatType): Quaternion in ``[x, y, z, w]`` order.

    Returns:
        np.ndarray: Quaternion in MuJoCo order ``[w, x, y, z]``.
    """
    q = np.asarray(quat_xyzw, dtype=float)
    return np.array([q[3], q[0], q[1], q[2]])


def _id_name_maps(model: mujoco.MjModel, obj_type: int, n: int):
    """Build (names, name->id, id->name) for a MuJoCo element kind. Unnamed elements map to ''."""
    names: List[str] = []
    name_to_id: Dict[str, int] = {}
    id_to_name: Dict[int, str] = {}
    for i in range(n):
        name = mujoco.mj_id2name(model, obj_type, i)
        name = name if name is not None else ""
        names.append(name)
        id_to_name[i] = name
        if name:
            name_to_id[name] = i
    return names, name_to_id, id_to_name


@dataclass
class MujocoJointInfo:
    """Static information about a single joint (from ``MjModel``)."""

    joint_id: int
    joint_name: str
    joint_type: int
    """MuJoCo joint type (``mujoco.mjtJoint``): free / ball / slide / hinge."""
    qpos_adr: int
    """Start index of this joint in ``data.qpos``."""
    qpos_dim: int
    """Number of ``qpos`` entries (free=7, ball=4, slide/hinge=1)."""
    dof_adr: int
    """Start index of this joint in ``data.qvel`` / dof-indexed arrays."""
    dof_dim: int
    """Number of dof entries (free=6, ball=3, slide/hinge=1)."""
    body_id: int
    """Id of the body this joint moves."""
    body_name: str
    lower_limit: float
    upper_limit: float
    limited: bool
    """Whether the joint has enforced range limits."""
    axis: Vector3D
    """Joint axis in the local frame."""
    damping: float
    """dof damping (nan for multi-dof joints)."""
    actuator_id: Optional[int]
    """Id of the JOINT-transmission actuator driving this joint, or None."""


class MujocoObject:
    """Static information about a MuJoCo object: name<->id maps and joint/link/dof bookkeeping.

    Built from an existing ``MjModel``. No physics is performed here; this only reads the model's
    structure (equivalent role to reading a loaded body's joint/link tables).
    """

    def __init__(self, model: mujoco.MjModel):
        """Read all addressing/structure information from a compiled model.

        Args:
            model (mujoco.MjModel): A compiled MuJoCo model.
        """
        self.model = model

        obj = mujoco.mjtObj
        self.body_names, self.body_name_to_id, self.body_id_to_name = _id_name_maps(
            model, obj.mjOBJ_BODY, model.nbody
        )
        self.joint_names, self.joint_name_to_id, self.joint_id_to_name = _id_name_maps(
            model, obj.mjOBJ_JOINT, model.njnt
        )
        self.actuator_names, self.actuator_name_to_id, self.actuator_id_to_name = _id_name_maps(
            model, obj.mjOBJ_ACTUATOR, model.nu
        )
        self.site_names, self.site_name_to_id, self.site_id_to_name = _id_name_maps(
            model, obj.mjOBJ_SITE, model.nsite
        )
        self.geom_names, self.geom_name_to_id, self.geom_id_to_name = _id_name_maps(
            model, obj.mjOBJ_GEOM, model.ngeom
        )

        # Links are bodies, excluding the world body (id 0).
        self.link_ids: List[int] = list(range(1, model.nbody))
        self.link_names: List[str] = [self.body_id_to_name[i] for i in self.link_ids]
        self.link_name_to_id: Dict[str, int] = {
            n: i for n, i in zip(self.link_names, self.link_ids) if n
        }
        self.link_id_to_name: Dict[int, str] = dict(zip(self.link_ids, self.link_names))

        # Base = root body (first body whose parent is the world).
        self.base_id: int = next(
            (i for i in range(1, model.nbody) if int(model.body_parentid[i]) == 0), 1
        )
        self.base_name: str = self.body_id_to_name.get(self.base_id, "")
        self.name: str = self.base_name
        """Name of the robot (its base/root link name)."""

        # Map each JOINT-transmission actuator to the joint it drives.
        self.joint_id_to_actuator_id: Dict[int, int] = {}
        for a in range(model.nu):
            if int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT):
                self.joint_id_to_actuator_id[int(model.actuator_trnid[a, 0])] = a
        self.has_actuators: bool = model.nu > 0

        # Total mass. Per-body dynamics (mass/inertia/com) are cheap direct lookups on the
        # model's ``body_*`` arrays, so there is no per-link info struct; see get_link_mass.
        self.mass: float = float(sum(model.body_mass[i] for i in self.link_ids))
        """Total mass of this object."""

        self._build_joint_info()

    def _build_joint_info(self):
        model = self.model
        free_type = int(mujoco.mjtJoint.mjJNT_FREE)
        hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)

        self.joint_name_to_info: Mapping[str, MujocoJointInfo] = {}
        self.actuated_joint_names: List[str] = []
        self.actuated_joint_ids: List[int] = []
        self.actuated_joint_lower_limits: List[float] = []
        self.actuated_joint_upper_limits: List[float] = []
        self.actuated_joint_qpos_adrs: List[int] = []
        self.actuated_joint_dof_adrs: List[int] = []
        self.continuous_joint_names: List[str] = []
        self.continuous_joint_ids: List[int] = []
        self.free_joint_id: Optional[int] = None

        for j in range(model.njnt):
            jtype = int(model.jnt_type[j])
            name = self.joint_id_to_name[j]
            qadr = int(model.jnt_qposadr[j])
            dadr = int(model.jnt_dofadr[j])
            limited = bool(model.jnt_limited[j])
            lower, upper = (float(x) for x in model.jnt_range[j])
            dof_dim = _DOF_DIM[jtype]

            self.joint_name_to_info[name] = MujocoJointInfo(
                joint_id=j,
                joint_name=name,
                joint_type=jtype,
                qpos_adr=qadr,
                qpos_dim=_QPOS_DIM[jtype],
                dof_adr=dadr,
                dof_dim=dof_dim,
                body_id=int(model.jnt_bodyid[j]),
                body_name=self.body_id_to_name[int(model.jnt_bodyid[j])],
                lower_limit=lower,
                upper_limit=upper,
                limited=limited,
                axis=np.array(model.jnt_axis[j]),
                damping=float(model.dof_damping[dadr]) if dof_dim == 1 else float("nan"),
                actuator_id=self.joint_id_to_actuator_id.get(j),
            )

            # The free joint is the floating base, not an actuated joint.
            if jtype == free_type:
                self.free_joint_id = j
                continue

            self.actuated_joint_ids.append(j)
            self.actuated_joint_names.append(name)
            self.actuated_joint_lower_limits.append(lower)
            self.actuated_joint_upper_limits.append(upper)
            self.actuated_joint_qpos_adrs.append(qadr)
            self.actuated_joint_dof_adrs.append(dadr)

            # A revolute joint with no limits is a continuous joint.
            if jtype == hinge_type and not limited:
                self.continuous_joint_names.append(name)
                self.continuous_joint_ids.append(j)

        self.num_actuated_joints: int = len(self.actuated_joint_ids)
        self.has_floating_base: bool = self.free_joint_id is not None
        self.has_continuous_joints: bool = len(self.continuous_joint_ids) > 0

    # --- convenience loaders ------------------------------------------------

    @classmethod
    def from_mjcf_path(cls, mjcf_path: str) -> "MujocoObject":
        """Compile a model from an MJCF (or URDF) file and read its structure.

        Args:
            mjcf_path (str): Path to an MJCF/URDF file.

        Returns:
            MujocoObject: Structure information for the compiled model.
        """
        return cls(mujoco.MjModel.from_xml_path(mjcf_path))

    # --- name <-> id accessors ---------------------------------------------

    def get_joint_id(self, joint_name: str) -> int:
        """Get the id of the specified joint."""
        return self.joint_name_to_id[joint_name]

    def get_joint_name(self, joint_id: int) -> str:
        """Get the name of the joint with the given id."""
        return self.joint_id_to_name[joint_id]

    def get_joint_ids(self, joint_names: List[str]) -> List[int]:
        """Get the ids of the specified joints, in the given order."""
        return [self.joint_name_to_id[name] for name in joint_names]

    def get_link_id(self, link_name: str) -> int:
        """Get the body id of the specified link."""
        return self.link_name_to_id[link_name]

    def get_link_name(self, link_id: int) -> str:
        """Get the name of the link (body) with the given id."""
        return self.link_id_to_name[link_id]

    def get_link_mass(self, link_name: str) -> float:
        """Get the mass of the specified link (body)."""
        return float(self.model.body_mass[self.get_link_id(link_name)])

    def get_actuator_id(self, actuator_name: str) -> int:
        """Get the id of the specified actuator."""
        return self.actuator_name_to_id[actuator_name]

    def get_site_id(self, site_name: str) -> int:
        """Get the id of the specified site."""
        return self.site_name_to_id[site_name]


class MujocoRobot(MujocoObject):
    """Robot interface utility for a generic robot in MuJoCo.

    Coming soon.
    """
