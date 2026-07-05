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

from typing import Dict, List, Mapping, Optional, Tuple, TypeAlias
from dataclasses import dataclass
import atexit
import threading
import time

import numpy as np
import mujoco

from .utils.model_builder import build_model

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


@dataclass
class MujocoRobotState:
    """A snapshot of common robot state (world frame; quaternions are [x,y,z,w]).

    Returned by :meth:`MujocoRobot.get_robot_states`.
    """

    base_position: Vector3D
    base_com_position: Vector3D
    base_quaternion: QuatType
    base_velocity_linear: Vector3D
    base_velocity_angular: Vector3D
    actuated_joint_positions: np.ndarray
    actuated_joint_velocities: np.ndarray
    actuated_joint_torques: np.ndarray
    joint_order: List[str]
    """Actuated-joint names, in the order the joint arrays are given."""
    ee_order: List[str]
    """End-effector names reported."""


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

    Loads a robot from an MJCF/URDF file (optionally wrapping it in a minimal world with a ground
    plane and light) or wraps an existing ``(model, data)`` pair, and manages the simulation
    lifecycle: stepping, gravity/timestep, an optional background stepping thread, an optional
    passive viewer, and shutdown.
    """

    def __init__(
        self,
        mjcf_path: str = None,
        urdf_path: str = None,
        model: mujoco.MjModel = None,
        data: mujoco.MjData = None,
        ee_names: List[str] = None,
        ee_sites: List[str] = None,
        run_async: bool = True,
        use_fixed_base: bool = True,
        default_joint_positions: List[float] = None,
        place_on_ground: bool = True,
        default_base_position: Vector3D = None,
        default_base_orientation: QuatType = None,
        enable_torque_mode: bool = False,
        verbose: bool = True,
        load_ground_plane: bool = True,
        ghost_mode: bool = False,
        render: bool = False,
    ):
        """Create a robot interface. Does not start the visualiser by default.

        Args:
            mjcf_path (str, optional): Path to an MJCF file to load (preferred).
            urdf_path (str, optional): Path to a URDF file to load (best-effort).
            model (mujoco.MjModel, optional): An already-compiled model to wrap. Takes precedence
                over the path arguments; no world-building is applied to it.
            data (mujoco.MjData, optional): Existing state to wrap. A fresh ``MjData`` is created
                if not provided.
            ee_names (List[str], optional): End-effector body names.
            ee_sites (List[str], optional): Optional site names to use as end-effector frames
                (overrides the body frame for those end-effectors when reading poses/jacobians).
            run_async (bool, optional): Step physics in a background thread. If False, call
                ``step()`` manually. Defaults to True.
            use_fixed_base (bool, optional): Fixed (True) or floating (False) base. Only applied
                when loading from a path. Defaults to True.
            default_joint_positions (List[float], optional): Initial actuated-joint positions
                (in ``actuated_joint_names`` order). Defaults to zeros.
            place_on_ground (bool, optional): Raise the base until the robot is not penetrating the
                ground. Defaults to True.
            default_base_position (Vector3D, optional): Initial base position. Defaults to zeros.
            default_base_orientation (QuatType, optional): Initial base orientation quaternion
                [x,y,z,w]. Defaults to identity.
            enable_torque_mode (bool, optional): Start in effort/torque control mode. Defaults to
                False.
            verbose (bool, optional): Print a robot info summary on construction. Defaults to True.
            load_ground_plane (bool, optional): Add a ground plane + light when loading from a
                path. Defaults to True.
            ghost_mode (bool, optional): Make the robot collision-free and translucent. Defaults
                to False.
            render (bool, optional): Launch a passive viewer on construction (needs a display).
                Defaults to False.
        """
        if model is None:
            if mjcf_path is None and urdf_path is None:
                raise ValueError("Provide one of: model, mjcf_path, urdf_path.")
            model = build_model(
                mjcf_path=mjcf_path,
                urdf_path=urdf_path,
                use_fixed_base=use_fixed_base,
                load_ground_plane=load_ground_plane,
            )
        super().__init__(model)
        self.data: mujoco.MjData = data if data is not None else mujoco.MjData(self.model)

        self.sync_mode: bool = not run_async
        self._lock = threading.RLock()
        self._viewer = None
        self._async_thread: Optional[threading.Thread] = None
        self._running: bool = False
        # Control state (consumed by the control methods and the per-step control hook).
        self._enable_torque_mode: bool = enable_torque_mode
        self._ghost_mode: bool = ghost_mode
        self._in_torque_mode: bool = False
        self._managed_targets: Dict[int, Tuple[Optional[float], float]] = {}
        self.default_position_kp: float = 100.0
        self.default_position_kd: float = 10.0

        if default_base_position is None:
            default_base_position = np.zeros(3)
        if default_base_orientation is None:
            default_base_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.default_start_pose = [
            np.array(default_base_position, dtype=float),
            np.array(default_base_orientation, dtype=float),
        ]

        self.ee_names: List[str] = list(ee_names) if ee_names else []
        self.ee_sites: List[str] = list(ee_sites) if ee_sites else []
        self.ee_ids: List[int] = [self.get_link_id(n) for n in self.ee_names]
        self.ee_site_ids: List[int] = [self.get_site_id(n) for n in self.ee_sites]

        if default_joint_positions is not None:
            self.default_joint_positions = np.asarray(default_joint_positions, dtype=float)
        else:
            self.default_joint_positions = np.zeros(self.num_actuated_joints)

        if self.has_floating_base:
            self.reset_base_pose(self.default_start_pose[0], self.default_start_pose[1])
        self.reset_joints(self.actuated_joint_ids, self.default_joint_positions)

        # Enter the initial control mode (position mode holds the current configuration).
        if self._enable_torque_mode:
            self.set_torque_control_mode()
        else:
            self.set_position_control_mode()

        if verbose:
            self._print_robot_info()
        if place_on_ground:
            self._place_robot_on_ground()
        if render:
            self.launch_viewer()
        if run_async:
            self._start_async()

        atexit.register(self.shutdown)

    # --- simulation lifecycle ----------------------------------------------

    def step(self):
        """Advance the simulation by one timestep (only in synchronous mode)."""
        if self.sync_mode:
            with self._lock:
                self._apply_managed_control()
                mujoco.mj_step(self.model, self.data)
                if self._viewer is not None:
                    self._viewer.sync()

    def _start_async(self):
        if self._running:
            return
        self._running = True
        self._async_thread = threading.Thread(target=self._async_loop, daemon=True)
        self._async_thread.start()

    def _async_loop(self):
        while self._running:
            dt = float(self.model.opt.timestep)
            start = time.perf_counter()
            with self._lock:
                self._apply_managed_control()
                mujoco.mj_step(self.model, self.data)
                if self._viewer is not None:
                    self._viewer.sync()
            time.sleep(max(0.0, dt - (time.perf_counter() - start)))

    def _stop_async(self):
        self._running = False
        thread = self._async_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._async_thread = None

    def launch_viewer(self):
        """Launch a passive MuJoCo viewer (requires a display). Returns the viewer handle."""
        import mujoco.viewer  # imported lazily so headless use never touches OpenGL

        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self._viewer

    def is_viewer_running(self) -> bool:
        """Whether a passive viewer is currently open."""
        return self._viewer is not None and self._viewer.is_running()

    def shutdown(self):
        """Stop the background thread (if any) and close the viewer (if any)."""
        self._stop_async()
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:  # noqa: BLE001 - viewer may already be gone
                pass
            self._viewer = None

    # --- world / timestep / gravity ----------------------------------------

    def get_timestep(self) -> float:
        """Get the physics timestep (seconds)."""
        return float(self.model.opt.timestep)

    def set_timestep(self, dt: float):
        """Set the physics timestep (seconds)."""
        self.model.opt.timestep = dt

    def get_gravity(self) -> Vector3D:
        """Get the gravity vector."""
        return np.array(self.model.opt.gravity)

    def set_gravity(self, gravity: Vector3D):
        """Set the gravity vector."""
        self.model.opt.gravity[:] = np.asarray(gravity, dtype=float)

    # --- resets -------------------------------------------------------------

    def reset_joints(
        self,
        joint_ids: List[int],
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray = None,
    ):
        """Set the positions (and optionally velocities) of the given (1-dof) joints.

        Writes ``qpos``/``qvel`` directly and recomputes derived quantities with ``mj_forward``
        (no dynamics stepping). Assumes 1-dof (hinge/slide) joints, as is the case for actuated
        joints.

        Args:
            joint_ids (List[int]): Joint ids to set.
            joint_positions (np.ndarray): Target positions, aligned with ``joint_ids``.
            joint_velocities (np.ndarray, optional): Target velocities. Defaults to zeros.
        """
        joint_positions = np.asarray(joint_positions, dtype=float)
        with self._lock:
            for n, jid in enumerate(joint_ids):
                info = self.joint_name_to_info[self.joint_id_to_name[jid]]
                self.data.qpos[info.qpos_adr] = joint_positions[n]
                if joint_velocities is not None:
                    self.data.qvel[info.dof_adr] = joint_velocities[n]
                elif info.dof_dim == 1:
                    self.data.qvel[info.dof_adr] = 0.0
            mujoco.mj_forward(self.model, self.data)

    def reset_base_pose(
        self,
        position: Vector3D = None,
        orientation: QuatType = None,
        base_linear_velocity: Vector3D = None,
        base_angular_velocity: Vector3D = None,
    ):
        """Set the base pose (and optionally base velocity for a floating base).

        For a floating base this writes the free-joint ``qpos``/``qvel``. For a fixed base it
        offsets the root body's placement in the model (``body_pos``/``body_quat``). Either way,
        ``mj_forward`` is called afterwards.

        Args:
            position (Vector3D, optional): Base position. Defaults to the stored start position.
            orientation (QuatType, optional): Base orientation quaternion [x,y,z,w]. Defaults to
                the stored start orientation.
            base_linear_velocity (Vector3D, optional): Floating-base linear velocity. Defaults to
                zeros. Ignored for a fixed base.
            base_angular_velocity (Vector3D, optional): Floating-base angular velocity. Defaults
                to zeros. Ignored for a fixed base.
        """
        if position is None:
            position = self.default_start_pose[0]
        if orientation is None:
            orientation = self.default_start_pose[1]
        position = np.asarray(position, dtype=float)
        with self._lock:
            if self.has_floating_base:
                info = self.joint_name_to_info[self.joint_id_to_name[self.free_joint_id]]
                a, d = info.qpos_adr, info.dof_adr
                self.data.qpos[a : a + 3] = position
                self.data.qpos[a + 3 : a + 7] = xyzw_to_wxyz(orientation)
                self.data.qvel[d : d + 3] = (
                    np.zeros(3) if base_linear_velocity is None else base_linear_velocity
                )
                self.data.qvel[d + 3 : d + 6] = (
                    np.zeros(3) if base_angular_velocity is None else base_angular_velocity
                )
            else:
                self.model.body_pos[self.base_id] = position
                self.model.body_quat[self.base_id] = xyzw_to_wxyz(orientation)
            mujoco.mj_forward(self.model, self.data)

    # --- placement ----------------------------------------------------------

    def _ground_geom_ids(self) -> set:
        plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
        return {i for i in range(self.model.ngeom) if int(self.model.geom_type[i]) == plane}

    def _touching_ground(self, ground_ids: set) -> bool:
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            if con.geom1 in ground_ids or con.geom2 in ground_ids:
                return True
        return False

    def _place_robot_on_ground(self, move_resolution: float = 0.01, max_iters: int = 1000):
        ground_ids = self._ground_geom_ids()
        if not ground_ids:
            return
        pos = np.array(self.default_start_pose[0], dtype=float)
        pos[2] = 0.0
        for _ in range(max_iters):
            self.reset_base_pose(position=pos, orientation=self.default_start_pose[1])
            if not self._touching_ground(ground_ids):
                break
            pos[2] += move_resolution
        self.default_start_pose[0] = pos

    # --- control ------------------------------------------------------------

    def set_position_control_mode(self):
        """Switch to position control: model actuators are active and hold the current pose.

        Position commands go to the model's actuators (``data.ctrl``); for any actuated joint
        without an actuator a PD controller drives ``data.qfrc_applied`` each step.
        """
        with self._lock:
            self.model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
            self.data.qfrc_applied[:] = 0.0
            self._managed_targets.clear()
            self._in_torque_mode = False
            for name in self.actuated_joint_names:
                info = self.joint_name_to_info[name]
                if info.actuator_id is not None:
                    self.data.ctrl[info.actuator_id] = self.data.qpos[info.qpos_adr]

    def set_torque_control_mode(self):
        """Switch to effort/torque control: model actuators are disabled.

        The robot will not hold itself up unless torque commands are sent each control cycle.
        Commands are applied via ``data.qfrc_applied``.
        """
        with self._lock:
            self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
            self.data.ctrl[:] = 0.0
            self.data.qfrc_applied[:] = 0.0
            self._managed_targets.clear()
            self._in_torque_mode = True

    @property
    def in_torque_mode(self) -> bool:
        """Whether the robot is currently in effort/torque control mode."""
        return self._in_torque_mode

    def _apply_managed_control(self):
        """Apply PD control (via qfrc_applied) for joints tracked without a model actuator.

        Called under the lock immediately before each ``mj_step``. Does nothing when there are no
        managed targets (e.g. when every actuated joint has a position actuator).
        """
        if not self._managed_targets:
            return
        kp, kd = self.default_position_kp, self.default_position_kd
        for jid, (q_des, dq_des) in self._managed_targets.items():
            info = self.joint_name_to_info[self.joint_id_to_name[jid]]
            tau = kd * (dq_des - self.data.qvel[info.dof_adr])
            if q_des is not None:
                err = q_des - self.data.qpos[info.qpos_adr]
                if jid in self.continuous_joint_ids:
                    err = wrap_angle(err)
                tau += kp * err
            self.data.qfrc_applied[info.dof_adr] = tau

    def _normalise_cmd(self, cmd, n: int) -> np.ndarray:
        cmd = np.asarray(cmd, dtype=float)
        return np.full(n, float(cmd)) if cmd.ndim == 0 else cmd

    def set_joint_positions(
        self, cmd: np.ndarray, actuated_joint_names: List[str] = None, vels: np.ndarray = None
    ):
        """Command target positions for the given actuated joints (position control).

        Joints with a position actuator receive the target via ``data.ctrl``; joints without one
        are driven toward the target by a PD controller on ``data.qfrc_applied`` each step.

        Args:
            cmd (np.ndarray): Target joint positions.
            actuated_joint_names (List[str], optional): Joints to command, in the order of ``cmd``.
                Defaults to all actuated joints.
            vels (np.ndarray, optional): Optional target velocities (used by the PD fallback).
        """
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        cmd = self._normalise_cmd(cmd, len(actuated_joint_names))
        with self._lock:
            for i, name in enumerate(actuated_joint_names):
                info = self.joint_name_to_info[name]
                dq_des = float(vels[i]) if vels is not None else 0.0
                if not self._in_torque_mode and info.actuator_id is not None:
                    self.data.ctrl[info.actuator_id] = cmd[i]
                    self._managed_targets.pop(info.joint_id, None)
                else:
                    self._managed_targets[info.joint_id] = (float(cmd[i]), dq_des)

    def set_joint_velocities(self, cmd: np.ndarray, actuated_joint_names: List[str] = None):
        """Command target velocities for the given actuated joints (velocity control).

        Implemented as a damping PD toward the target velocity via ``data.qfrc_applied`` (gain
        ``default_position_kd``), since MuJoCo models rarely ship velocity actuators.

        Args:
            cmd (np.ndarray): Target joint velocities.
            actuated_joint_names (List[str], optional): Joints to command, in the order of ``cmd``.
        """
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        cmd = self._normalise_cmd(cmd, len(actuated_joint_names))
        with self._lock:
            for i, name in enumerate(actuated_joint_names):
                info = self.joint_name_to_info[name]
                self._managed_targets[info.joint_id] = (None, float(cmd[i]))

    def set_joint_torques(self, cmd: np.ndarray, actuated_joint_names: List[str] = None):
        """Command target torques/efforts for the given actuated joints (effort control).

        Writes ``data.qfrc_applied`` directly; intended for use in torque control mode.

        Args:
            cmd (np.ndarray): Target joint torques.
            actuated_joint_names (List[str], optional): Joints to command, in the order of ``cmd``.
        """
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        cmd = self._normalise_cmd(cmd, len(actuated_joint_names))
        with self._lock:
            for i, name in enumerate(actuated_joint_names):
                info = self.joint_name_to_info[name]
                self.data.qfrc_applied[info.dof_adr] = cmd[i]
                self._managed_targets.pop(info.joint_id, None)

    def set_joint_positions_delta(self, cmd: np.ndarray, actuated_joint_names: List[str] = None):
        """Command a position increment relative to the current positions (experimental).

        Args:
            cmd (np.ndarray): Position deltas.
            actuated_joint_names (List[str], optional): Joints to command, in the order of ``cmd``.
        """
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        current = self.get_actuated_joint_positions(actuated_joint_names)
        self.set_joint_positions(
            current + self._normalise_cmd(cmd, len(actuated_joint_names)), actuated_joint_names
        )

    def compute_joint_pd_error(
        self,
        joint_ids: List[int],
        q_des: np.ndarray,
        dq_des: np.ndarray,
        Kp: float | np.ndarray,
        Kd: float | np.ndarray,
    ) -> np.ndarray:
        """Compute a joint-space PD term: ``Kp * (q_des - q) + Kd * (dq_des - dq)``.

        Continuous joints use the wrapped position error. A None target zeroes that term.

        Args:
            joint_ids (List[int]): Joints to compute the error for.
            q_des (np.ndarray): Desired positions (or None to skip the position term).
            dq_des (np.ndarray): Desired velocities (or None to skip the velocity term).
            Kp (float | np.ndarray): Position gain(s).
            Kd (float | np.ndarray): Velocity gain(s).

        Returns:
            np.ndarray: The PD term per joint.
        """
        q, dq, _ = self.get_joint_states(joint_ids)
        p_term = np.zeros_like(q)
        if q_des is not None:
            p_term = Kp * (q_des - q)
            if self.has_continuous_joints:
                for n, jid in enumerate(joint_ids):
                    if jid in self.continuous_joint_ids:
                        kp_n = Kp[n] if np.ndim(Kp) else Kp
                        p_term[n] = kp_n * wrap_angle(q_des[n] - q[n])
        d_term = np.zeros_like(dq)
        if dq_des is not None:
            d_term = Kd * (dq_des - dq)
        return p_term + d_term

    def set_actuated_joint_commands(
        self,
        actuated_joint_names: List[str] = None,
        q: float | np.ndarray = 0,
        dq: float | np.ndarray = 0,
        Kp: float | np.ndarray = 0,
        Kd: float | np.ndarray = 0,
        tau: float | np.ndarray = 0,
    ):
        """Send a position-velocity-torque + PD (PVT-PD) command.

        In torque control mode the applied effort is ``Kp*(q - q_now) + Kd*(dq - dq_now) + tau``,
        written to ``data.qfrc_applied``. In position control mode ``q`` is used as the position
        target (and ``dq`` as an optional target velocity); the PD/feedforward terms are ignored
        because the model's actuators close the loop.

        Args:
            actuated_joint_names (List[str], optional): Joints to command. Defaults to all.
            q (float | np.ndarray, optional): Position command(s). Defaults to 0.
            dq (float | np.ndarray, optional): Velocity command(s). Defaults to 0.
            Kp (float | np.ndarray, optional): Stiffness gain(s). Defaults to 0.
            Kd (float | np.ndarray, optional): Damping gain(s). Defaults to 0.
            tau (float | np.ndarray, optional): Feedforward torque(s). Defaults to 0.
        """
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        joint_ids = self.get_joint_ids(actuated_joint_names)
        if self._in_torque_mode:
            tau_cmd = self.compute_joint_pd_error(joint_ids, q_des=q, dq_des=dq, Kp=Kp, Kd=Kd) + tau
            with self._lock:
                for i, jid in enumerate(joint_ids):
                    info = self.joint_name_to_info[self.joint_id_to_name[jid]]
                    self.data.qfrc_applied[info.dof_adr] = tau_cmd[i]
                    self._managed_targets.pop(jid, None)
        else:
            vels = dq if np.ndim(dq) else None
            self.set_joint_positions(q, actuated_joint_names=actuated_joint_names, vels=vels)

    # --- scratch data (for off-simulation queries) -------------------------

    def _get_scratch_data(self) -> mujoco.MjData:
        """A reusable ``MjData`` for queries that must not disturb the live state."""
        if getattr(self, "_scratch_data", None) is None:
            self._scratch_data = mujoco.MjData(self.model)
        return self._scratch_data

    # --- joint state --------------------------------------------------------

    def get_joint_states(
        self, joint_ids: List[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get (positions, velocities, applied efforts) for the given joints.

        Continuous-joint positions are wrapped to [-pi, pi]. Effort is the applied actuation
        generalised force on the joint (``qfrc_actuator`` + ``qfrc_applied``).

        Args:
            joint_ids (List[int], optional): Joints to read. Defaults to all actuated joints.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: positions, velocities, efforts.
        """
        if joint_ids is None:
            joint_ids = self.actuated_joint_ids
        n = len(joint_ids)
        q, v, tau = np.zeros(n), np.zeros(n), np.zeros(n)
        with self._lock:
            for i, jid in enumerate(joint_ids):
                info = self.joint_name_to_info[self.joint_id_to_name[jid]]
                pos = self.data.qpos[info.qpos_adr]
                q[i] = wrap_angle(pos) if jid in self.continuous_joint_ids else pos
                v[i] = self.data.qvel[info.dof_adr]
                tau[i] = (
                    self.data.qfrc_actuator[info.dof_adr]
                    + self.data.qfrc_applied[info.dof_adr]
                )
        return q, v, tau

    def get_actuated_joint_positions(self, actuated_joint_names: List[str] = None) -> np.ndarray:
        """Get current positions of the actuated joints (defaults to all, in default order)."""
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        return self.get_joint_states(self.get_joint_ids(actuated_joint_names))[0]

    def get_actuated_joint_velocities(self, actuated_joint_names: List[str] = None) -> np.ndarray:
        """Get current velocities of the actuated joints (defaults to all, in default order)."""
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        return self.get_joint_states(self.get_joint_ids(actuated_joint_names))[1]

    def get_actuated_joint_torques(self, actuated_joint_names: List[str] = None) -> np.ndarray:
        """Get current efforts of the actuated joints (defaults to all, in default order)."""
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        return self.get_joint_states(self.get_joint_ids(actuated_joint_names))[2]

    # --- base state ---------------------------------------------------------

    def _object_velocity(self, body_id: int) -> Tuple[Vector3D, Vector3D]:
        """World-frame spatial velocity of a body as (linear, angular)."""
        res = np.zeros(6)
        with self._lock:
            mujoco.mj_objectVelocity(
                self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, int(body_id), res, 0
            )
        # mj_objectVelocity returns [angular (3), linear (3)].
        return np.array(res[3:]), np.array(res[:3])

    def get_base_pose(self) -> Tuple[Vector3D, QuatType]:
        """Get the world-frame base pose as (position, orientation quaternion [x,y,z,w])."""
        with self._lock:
            pos = np.array(self.data.xpos[self.base_id])
            quat = wxyz_to_xyzw(self.data.xquat[self.base_id])
        return pos, quat

    def get_base_velocity(self) -> Tuple[Vector3D, Vector3D]:
        """Get the world-frame base velocity as (linear velocity, angular velocity)."""
        return self._object_velocity(self.base_id)

    def get_base_com_position(self) -> Vector3D:
        """Get the world-frame position of the base link's centre of mass."""
        with self._lock:
            return np.array(self.data.xipos[self.base_id])

    # --- link / site state --------------------------------------------------

    def get_link_pose(self, link_id: int) -> Tuple[Vector3D, QuatType]:
        """Get the world-frame link (body) pose as (position, orientation quaternion [x,y,z,w])."""
        with self._lock:
            return np.array(self.data.xpos[link_id]), wxyz_to_xyzw(self.data.xquat[link_id])

    def get_link_velocity(self, link_id: int) -> Tuple[Vector3D, Vector3D]:
        """Get the world-frame link velocity as (linear velocity, angular velocity)."""
        return self._object_velocity(link_id)

    def get_link_com_pose(self, link_id: int) -> Tuple[Vector3D, QuatType]:
        """Get the world-frame pose of the link's centre of mass as (position, orientation)."""
        quat = np.zeros(4)
        with self._lock:
            pos = np.array(self.data.xipos[link_id])
            mujoco.mju_mat2Quat(quat, self.data.ximat[link_id])
        return pos, wxyz_to_xyzw(quat)

    def get_site_pose(self, site_id: int) -> Tuple[Vector3D, QuatType]:
        """Get the world-frame pose of a site as (position, orientation quaternion [x,y,z,w])."""
        quat = np.zeros(4)
        with self._lock:
            pos = np.array(self.data.site_xpos[site_id])
            mujoco.mju_mat2Quat(quat, self.data.site_xmat[site_id])
        return pos, wxyz_to_xyzw(quat)

    # --- jacobian -----------------------------------------------------------

    def _jacobian_from_data(self, data: mujoco.MjData, frame_name: str) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        if frame_name in self.link_name_to_id:
            mujoco.mj_jacBody(self.model, data, jacp, jacr, self.link_name_to_id[frame_name])
        elif frame_name in self.site_name_to_id:
            mujoco.mj_jacSite(self.model, data, jacp, jacr, self.site_name_to_id[frame_name])
        else:
            raise KeyError(f"No link or site named '{frame_name}'.")
        cols = self.actuated_joint_dof_adrs
        return np.vstack([jacp[:, cols], jacr[:, cols]])

    def get_jacobian(self, ee_link_name: str, joint_angles: np.ndarray = None) -> np.ndarray:
        """Compute the 6xN geometric Jacobian (linear stacked over angular) for a frame.

        The frame may be a link (body) name or a site name. Columns correspond to the actuated
        joints, in ``actuated_joint_names`` order.

        Args:
            ee_link_name (str): Name of the link (body) or site to compute the Jacobian for.
            joint_angles (np.ndarray, optional): Actuated-joint configuration to evaluate at.
                Defaults to None (use the current configuration). Evaluated on a scratch copy so
                the live state is not disturbed.

        Returns:
            np.ndarray: 6xN Jacobian matrix.
        """
        if joint_angles is None:
            with self._lock:
                return self._jacobian_from_data(self.data, ee_link_name)
        with self._lock:
            scratch = self._get_scratch_data()
            scratch.qpos[:] = self.data.qpos
            for i, name in enumerate(self.actuated_joint_names):
                scratch.qpos[self.joint_name_to_info[name].qpos_adr] = joint_angles[i]
            mujoco.mj_forward(self.model, scratch)
            return self._jacobian_from_data(scratch, ee_link_name)

    # --- dynamics -----------------------------------------------------------

    def get_gravity_compensation_torques(self) -> np.ndarray:
        """Joint torques that hold the robot static against gravity at the current configuration.

        Computed as the bias force (``qfrc_bias``) evaluated at zero velocity and acceleration on
        a scratch copy, i.e. the pure gravity term, per actuated joint (``actuated_joint_names``
        order). Useful as a feedforward term for torque / impedance control.

        Returns:
            np.ndarray: Gravity-compensation torque per actuated joint.
        """
        with self._lock:
            scratch = self._get_scratch_data()
            scratch.qpos[:] = self.data.qpos
            scratch.qvel[:] = 0.0
            scratch.qacc[:] = 0.0
            mujoco.mj_forward(self.model, scratch)
            bias = np.array(scratch.qfrc_bias)
        return np.array(
            [bias[self.joint_name_to_info[n].dof_adr] for n in self.actuated_joint_names]
        )

    # --- state bundle -------------------------------------------------------

    def get_robot_states(
        self, actuated_joint_names: List[str] = None, ee_names: List[str] = None
    ) -> MujocoRobotState:
        """Get a consistent snapshot of common robot state.

        Args:
            actuated_joint_names (List[str], optional): Joints to include. Defaults to all.
            ee_names (List[str], optional): End-effectors to report the order of. Defaults to all.

        Returns:
            MujocoRobotState: The state snapshot.
        """
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        if ee_names is None:
            ee_names = self.ee_names
        with self._lock:
            joint_ids = self.get_joint_ids(actuated_joint_names)
            base_pos, base_quat = self.get_base_pose()
            base_lin, base_ang = self.get_base_velocity()
            q, v, tau = self.get_joint_states(joint_ids)
            return MujocoRobotState(
                base_position=base_pos,
                base_com_position=self.get_base_com_position(),
                base_quaternion=base_quat,
                base_velocity_linear=base_lin,
                base_velocity_angular=base_ang,
                actuated_joint_positions=q,
                actuated_joint_velocities=v,
                actuated_joint_torques=tau,
                joint_order=list(actuated_joint_names),
                ee_order=list(ee_names),
            )

    # --- info ---------------------------------------------------------------

    def _print_robot_info(self):
        bar = "*" * 80
        print(f"\n{bar}\nMujocoRobot Info\n{bar}")
        print(f"  name:               {self.name}")
        print(f"  total mass:         {self.mass:.4f}")
        print(f"  base link:          {self.base_name}")
        print(f"  floating base:      {self.has_floating_base}")
        print(f"  has actuators:      {self.has_actuators}")
        print(f"  links ({len(self.link_names)}):           {self.link_names}")
        print(f"  actuated joints ({self.num_actuated_joints}): {self.actuated_joint_names}")
        print(f"  lower limits:       {np.round(self.actuated_joint_lower_limits, 3).tolist()}")
        print(f"  upper limits:       {np.round(self.actuated_joint_upper_limits, 3).tolist()}")
        print(f"  continuous joints:  {self.continuous_joint_names}")
        print(f"  end-effectors:      {self.ee_names}")
        print(f"  timestep:           {self.get_timestep()}")
        print(f"{bar}\n")
