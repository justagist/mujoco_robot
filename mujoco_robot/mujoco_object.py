"""Model structure for mujoco_robot: name<->id maps and joint/link/dof bookkeeping."""

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import mujoco

from .types import Frame, MujocoJointInfo

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


class MujocoObject:
    """Static information about a MuJoCo object: name<->id maps and joint/link/dof bookkeeping.

    Built from an existing ``MjModel``. No physics is performed here; this only reads the model's
    structure (equivalent role to reading a loaded body's joint/link tables).

    By default the whole model is the object: every non-free joint is an actuated joint, every
    body is a link, and the root is the first child of the world. Passing ``joint_names`` scopes
    the object to one robot inside a larger world: only those joints are actuated, and the links
    are the subtree under the root body. Everything else in the model is foreign.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        joint_names: Optional[Sequence[str]] = None,
        root_body: Optional[str] = None,
    ):
        """Read all addressing/structure information from a compiled model.

        Args:
            model (mujoco.MjModel): A compiled MuJoCo model.
            joint_names (Sequence[str], optional): Scope the object to exactly these 1-dof (hinge
                or slide) joints, in this order. Defaults to None (the whole model).
            root_body (str, optional): Name of the robot's root body. Base pose and velocity are
                read from it, and the base is floating iff it carries a free joint. Defaults to
                the first child of the world, or, when ``joint_names`` is given, the deepest body
                that is an ancestor of every scoped joint.
        """
        self.model = model
        self.is_scoped: bool = joint_names is not None
        """Whether this object covers only the given ``joint_names`` (not the whole model)."""

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
        self.sensor_names, self.sensor_name_to_id, self.sensor_id_to_name = _id_name_maps(
            model, obj.mjOBJ_SENSOR, model.nsensor
        )

        scoped_joint_ids = self._resolve_scoped_joints(joint_names) if self.is_scoped else None

        # Base = root body: the given one, else the common ancestor of the scoped joints, else
        # the first body whose parent is the world.
        if root_body is not None:
            if root_body not in self.body_name_to_id or self.body_name_to_id[root_body] == 0:
                raise ValueError(f"'{root_body}' is not a body that can be a robot root.")
            self.base_id: int = self.body_name_to_id[root_body]
        elif self.is_scoped:
            self.base_id = self._infer_root_body(scoped_joint_ids)
        else:
            self.base_id = next(
                (i for i in range(1, model.nbody) if int(model.body_parentid[i]) == 0), 1
            )
        self.base_name: str = self.body_id_to_name.get(self.base_id, "")
        self.name: str = self.base_name
        """Name of the robot (its base/root link name)."""

        # Links are bodies, excluding the world body (id 0). Scoped: the subtree under the root.
        if self.is_scoped:
            self.link_ids: List[int] = self._subtree_body_ids(self.base_id)
        else:
            self.link_ids = list(range(1, model.nbody))
        self.link_names: List[str] = [self.body_id_to_name[i] for i in self.link_ids]
        self.link_name_to_id: Dict[str, int] = {
            n: i for n, i in zip(self.link_names, self.link_ids) if n
        }
        self.link_id_to_name: Dict[int, str] = dict(zip(self.link_ids, self.link_names))

        # Geoms belonging to the robot's links (for transparency / collision toggling).
        link_id_set = set(self.link_ids)
        self.robot_geom_ids: List[int] = [
            g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in link_id_set
        ]

        # Map each joint-transmission actuator to the joint it drives (matched by transmission
        # target, so actuator names need not match joint names).
        joint_trn = (int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT))
        self.joint_id_to_actuator_id: Dict[int, int] = {}
        joint_actuators: Dict[int, List[int]] = {}
        for a in range(model.nu):
            if int(model.actuator_trntype[a]) in joint_trn:
                jid = int(model.actuator_trnid[a, 0])
                self.joint_id_to_actuator_id[jid] = a
                joint_actuators.setdefault(jid, []).append(a)
        if self.is_scoped:
            self.robot_actuator_ids: List[int] = sorted(
                a for j in scoped_joint_ids for a in joint_actuators.get(j, [])
            )
            """Ids of the actuators driving this robot's actuated joints."""
            self.has_actuators: bool = len(self.robot_actuator_ids) > 0
        else:
            self.robot_actuator_ids = list(range(model.nu))
            self.has_actuators = model.nu > 0

        # Total mass. Per-body dynamics (mass/inertia/com) are cheap direct lookups on the
        # model's ``body_*`` arrays, so there is no per-link info struct; see get_link_mass.
        self.mass: float = float(sum(model.body_mass[i] for i in self.link_ids))
        """Total mass of this object."""

        self._build_joint_info(scoped_joint_ids)

    def _resolve_scoped_joints(self, joint_names: Sequence[str]) -> List[int]:
        """Validate the scoped joint names and return their ids, in the given order."""
        one_dof = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        joint_names = list(joint_names)
        if not joint_names:
            raise ValueError("joint_names must name at least one joint.")
        if len(set(joint_names)) != len(joint_names):
            raise ValueError(f"joint_names contains duplicates: {joint_names}")
        ids = []
        for name in joint_names:
            if name not in self.joint_name_to_id:
                raise ValueError(f"No joint named '{name}' in the model.")
            jid = self.joint_name_to_id[name]
            if int(self.model.jnt_type[jid]) not in one_dof:
                raise ValueError(f"Joint '{name}' is not a hinge or slide joint.")
            ids.append(jid)
        return ids

    def _ancestors(self, body_id: int) -> List[int]:
        """The body and its ancestors, nearest first, ending at the world body (id 0)."""
        chain = [body_id]
        while chain[-1] != 0:
            chain.append(int(self.model.body_parentid[chain[-1]]))
        return chain

    def _common_ancestor(self, body_ids: List[int]) -> int:
        """Deepest body that is an ancestor of (or equal to) every body in ``body_ids``."""
        common = self._ancestors(body_ids[0])
        for b in body_ids[1:]:
            ancestors = set(self._ancestors(b))
            common = [c for c in common if c in ancestors]
        return common[0]

    def _infer_root_body(self, joint_ids: List[int]) -> int:
        """Root of a scoped robot: the deepest body that the scoped joints all move relative to.

        That is the common ancestor of the joints' parent bodies (the arm base for a serial
        chain, the hand for a pair of fingers). If that is the world, fall back to the common
        ancestor of the joint bodies themselves.
        """
        joint_bodies = [int(self.model.jnt_bodyid[j]) for j in joint_ids]
        root = self._common_ancestor([int(self.model.body_parentid[b]) for b in joint_bodies])
        if root == 0:
            root = self._common_ancestor(joint_bodies)
        if root == 0:
            raise ValueError(
                "The given joints share no common body other than the world; pass root_body."
            )
        return root

    def _subtree_body_ids(self, root_id: int) -> List[int]:
        """Ids of ``root_id`` and all of its descendants (MuJoCo orders parents before children)."""
        subtree = {root_id}
        for b in range(root_id + 1, self.model.nbody):
            if int(self.model.body_parentid[b]) in subtree:
                subtree.add(b)
        return sorted(subtree)

    def _build_joint_info(self, scoped_joint_ids: Optional[List[int]] = None):
        model = self.model
        free_type = int(mujoco.mjtJoint.mjJNT_FREE)
        hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)

        self.joint_name_to_info: Mapping[str, MujocoJointInfo] = {}
        # Keyed by id as well, since unnamed joints all share the name "".
        self._joint_info_by_id: Dict[int, MujocoJointInfo] = {}
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

            info = MujocoJointInfo(
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
            self.joint_name_to_info[name] = info
            self._joint_info_by_id[j] = info

            # A free joint on the root body is the floating base. Free joints elsewhere belong to
            # other bodies in the world. Neither is an actuated joint.
            if jtype == free_type:
                if int(model.jnt_bodyid[j]) == self.base_id:
                    self.free_joint_id = j
                continue

        if scoped_joint_ids is None:
            scoped_joint_ids = [
                j for j in range(model.njnt) if int(model.jnt_type[j]) != free_type
            ]
        else:
            subtree = set(self.link_ids)
            outside = [
                self.joint_id_to_name[j]
                for j in scoped_joint_ids
                if int(model.jnt_bodyid[j]) not in subtree
            ]
            if outside:
                raise ValueError(f"Joints {outside} are not under root body '{self.base_name}'.")

        for j in scoped_joint_ids:
            info = self._joint_info_by_id[j]
            self.actuated_joint_ids.append(j)
            self.actuated_joint_names.append(info.joint_name)
            self.actuated_joint_lower_limits.append(info.lower_limit)
            self.actuated_joint_upper_limits.append(info.upper_limit)
            self.actuated_joint_qpos_adrs.append(info.qpos_adr)
            self.actuated_joint_dof_adrs.append(info.dof_adr)

            # A revolute joint with no limits is a continuous joint.
            if info.joint_type == hinge_type and not info.limited:
                self.continuous_joint_names.append(info.joint_name)
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

    def get_link_mass(self, link_name: int | str) -> float:
        """Get the mass of the specified link (body), given by name or id."""
        return float(self.model.body_mass[self._resolve_link_id(link_name)])

    def get_actuator_id(self, actuator_name: str) -> int:
        """Get the id of the specified actuator."""
        return self.actuator_name_to_id[actuator_name]

    def get_site_id(self, site_name: str) -> int:
        """Get the id of the specified site."""
        return self.site_name_to_id[site_name]

    # --- name-or-id resolution ----------------------------------------------
    #
    # Methods bound to one kind of element take a name or that kind's id. Numpy integers count as
    # ids (ids read from model arrays are numpy types). Unknown names raise KeyError.

    @staticmethod
    def _is_id(x) -> bool:
        return isinstance(x, (int, np.integer)) and not isinstance(x, bool)

    def _resolve_link_id(self, link: int | str) -> int:
        return int(link) if self._is_id(link) else self.link_name_to_id[link]

    def _resolve_site_id(self, site: int | str) -> int:
        return int(site) if self._is_id(site) else self.site_name_to_id[site]

    def _resolve_geom_id(self, geom: int | str) -> int:
        return int(geom) if self._is_id(geom) else self.geom_name_to_id[geom]

    def _joint_info(self, joint: int | str) -> MujocoJointInfo:
        if self._is_id(joint):
            return self._joint_info_by_id[int(joint)]
        return self.joint_name_to_info[joint]

    def _resolve_joint_ids(self, joints) -> List[int]:
        return [self._joint_info(j).joint_id for j in joints]

    def frame(self, name: str) -> Frame:
        """Resolve a link or site name to a typed :class:`Frame` handle.

        Use it to cache a frame for methods that accept any frame (e.g. ``get_jacobian``), such as
        a tool frame that is a body in one description and a site in another. A link name takes
        precedence over a site with the same name.

        Args:
            name (str): Link (body) or site name.

        Returns:
            Frame: The resolved frame.
        """
        if name in self.link_name_to_id:
            return Frame("link", self.link_name_to_id[name], name)
        if name in self.site_name_to_id:
            return Frame("site", self.site_name_to_id[name], name)
        raise KeyError(f"No link or site named '{name}'.")
