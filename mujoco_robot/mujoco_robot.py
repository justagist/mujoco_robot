"""MuJoCo robot interface, implemented purely on ``mujoco.MjModel`` + ``MjData`` (no client-server).

Conventions (``[x, y, z, w]`` quaternions, radians) are documented in :mod:`mujoco_robot.types`.
The types and :class:`MujocoObject` are re-exported here, so existing imports from
``mujoco_robot.mujoco_robot`` keep working.
"""

from typing import Dict, List, Optional, Tuple
from collections import deque
import atexit
import threading
import time
import warnings

import numpy as np
import mujoco

from .mujoco_object import MujocoObject
from .types import (
    ContactInfo,
    Frame,
    MujocoJointInfo,
    MujocoRobotState,
    QuatType,
    Vector3D,
    wrap_angle,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from .utils.model_builder import build_model

__all__ = [
    "ContactInfo",
    "Frame",
    "MujocoJointInfo",
    "MujocoObject",
    "MujocoRobot",
    "MujocoRobotState",
    "QuatType",
    "Vector3D",
    "wrap_angle",
    "wxyz_to_xyzw",
    "xyzw_to_wxyz",
]


class MujocoRobot(MujocoObject):
    """Robot interface utility for a generic robot in MuJoCo.

    Loads a robot from an MJCF/URDF file (optionally wrapping it in a minimal world with a ground
    plane and light) or wraps an existing ``(model, data)`` pair, and manages the simulation
    lifecycle: stepping, gravity/timestep, an optional background stepping thread, an optional
    passive viewer, and shutdown.

    Wrapping mode (``model`` passed in) is for a robot that lives inside a world owned by someone
    else. The host steps ``data``; this object only reads state and writes ``ctrl`` /
    ``qfrc_applied`` for its own joints. Construction changes nothing unless initial positions
    are given explicitly. Pass ``joint_names`` (and optionally ``root_body``) to scope the robot
    to one part of the model, leaving every other joint, actuator and body alone.
    """

    def __init__(
        self,
        mjcf_path: str = None,
        urdf_path: str = None,
        model: mujoco.MjModel = None,
        data: mujoco.MjData = None,
        ee_names: List[str] = None,
        ee_sites: List[str] = None,
        run_async: Optional[bool] = None,
        use_fixed_base: bool = True,
        default_joint_positions: List[float] = None,
        place_on_ground: Optional[bool] = None,
        default_base_position: Vector3D = None,
        default_base_orientation: QuatType = None,
        enable_torque_mode: bool = False,
        verbose: bool = True,
        load_ground_plane: bool = True,
        ft_sensor_links: List[str] = None,
        smooth_ft: bool = False,
        ft_smoothing_window: int = 10,
        render: bool = False,
        joint_names: List[str] = None,
        root_body: str = None,
        managed_pd: Optional[bool] = None,
    ):
        """Create a robot interface. Does not start the visualiser by default.

        Args:
            mjcf_path (str, optional): Path to an MJCF file to load (preferred).
            urdf_path (str, optional): Path to a URDF file to load (best-effort).
            model (mujoco.MjModel, optional): An already-compiled model to wrap. Takes precedence
                over the path arguments; no world-building is applied to it. Selects wrapping
                mode, which changes several defaults below.
            data (mujoco.MjData, optional): Existing state to wrap. A fresh ``MjData`` is created
                if not provided.
            ee_names (List[str], optional): End-effector body names.
            ee_sites (List[str], optional): Optional site names to use as end-effector frames
                (overrides the body frame for those end-effectors when reading poses/jacobians).
            run_async (bool, optional): Step physics in a background thread. If False, call
                ``step()`` manually. Defaults to True when loading from a path and False in
                wrapping mode.
            use_fixed_base (bool, optional): Fixed (True) or floating (False) base. Only applied
                when loading from a path. Defaults to True.
            default_joint_positions (List[float], optional): Initial actuated-joint positions
                (in ``actuated_joint_names`` order). Applied on construction if given. Defaults
                to zeros (applied) when loading from a path, and to the current positions (not
                applied) in wrapping mode.
            place_on_ground (bool, optional): Raise the base until the robot is not penetrating the
                ground. Defaults to True when loading from a path and False in wrapping mode.
            default_base_position (Vector3D, optional): Initial base position. Applied on
                construction if given. Defaults to zeros (applied) when loading from a path, and
                to the current base position (not applied) in wrapping mode.
            default_base_orientation (QuatType, optional): Initial base orientation quaternion
                [x,y,z,w]. Same rules as ``default_base_position``; defaults to identity when
                loading from a path.
            enable_torque_mode (bool, optional): Start in effort/torque control mode. Defaults to
                False.
            verbose (bool, optional): Print a robot info summary on construction. Defaults to True.
            load_ground_plane (bool, optional): Add a ground plane + light when loading from a
                path. Defaults to True.
            ft_sensor_links (List[str], optional): Bodies to instrument with force-torque sensors
                (compile-time; only applied when loading from a path). Read with
                ``get_link_ft_measurement`` / ``get_ft_reading``.
            smooth_ft (bool, optional): Average each force-torque sensor over a moving window
                (filled by a post-step callable). Defaults to False.
            ft_smoothing_window (int, optional): Window length used when ``smooth_ft`` is set.
            render (bool, optional): Launch a passive viewer on construction (needs a display).
                Defaults to False.
            joint_names (List[str], optional): Scope the robot to exactly these hinge/slide
                joints, in this order. They become ``actuated_joint_names``; every other joint
                in the model is foreign and is never written. Defaults to None (every non-free
                joint in the model).
            root_body (str, optional): The robot's root body. Base pose and velocity come from
                it, and the base is floating iff it carries a free joint. Defaults to the first
                child of the world, or, with ``joint_names``, the common ancestor of the joints.
            managed_pd (bool, optional): Allow position/velocity commands to joints that have no
                actuator, by running a PD loop on ``qfrc_applied`` before each step. If False,
                such commands raise instead. Defaults to True when loading from a path and False
                in wrapping mode. In wrapping mode the host must call
                :meth:`apply_managed_control` before each ``mj_step`` for the PD to act.
        """
        wrapping = model is not None
        if model is None:
            if mjcf_path is None and urdf_path is None:
                raise ValueError("Provide one of: model, mjcf_path, urdf_path.")
            model = build_model(
                mjcf_path=mjcf_path,
                urdf_path=urdf_path,
                use_fixed_base=use_fixed_base,
                load_ground_plane=load_ground_plane,
                ft_sensor_links=ft_sensor_links,
            )
        if run_async is None:
            run_async = not wrapping
        if place_on_ground is None:
            place_on_ground = not wrapping
        if managed_pd is None:
            managed_pd = not wrapping

        super().__init__(model, joint_names=joint_names, root_body=root_body)
        self.data: mujoco.MjData = data if data is not None else mujoco.MjData(self.model)
        self.managed_pd: bool = managed_pd
        """Whether commands to joints without an actuator are served by a PD on qfrc_applied."""
        self._actuated_joint_id_set = frozenset(self.actuated_joint_ids)

        self.sync_mode: bool = not run_async
        self._lock = threading.RLock()
        self._viewer = None
        self._async_thread: Optional[threading.Thread] = None
        self._running: bool = False
        # Extensibility hooks run around each step: {name: (callable, args)}.
        self._pre_step_callables: Dict[str, Tuple] = {}
        self._post_step_callables: Dict[str, Tuple] = {}
        # Control state (consumed by the control methods and the per-step control hook).
        self._enable_torque_mode: bool = enable_torque_mode
        self._in_torque_mode: bool = False
        self._managed_targets: Dict[int, Tuple[Optional[float], float]] = {}
        # Cosmetics: geoms whose collision masks were saved when disabled (for restore).
        self._saved_geom_masks: Dict[int, Tuple[int, int]] = {}
        # Scoped torque mode: own actuators' (forcelimited, forcerange) saved while clamped to 0.
        self._saved_actuator_limits: Dict[int, Tuple[int, np.ndarray]] = {}
        self.default_position_kp: float = 100.0
        self.default_position_kd: float = 10.0

        base_given = default_base_position is not None or default_base_orientation is not None
        if wrapping:
            current_pos, current_wxyz = self._kinematic_body_pose(self.base_id)
            current_quat = wxyz_to_xyzw(current_wxyz)
        else:
            current_pos, current_quat = np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])
        if default_base_position is None:
            default_base_position = current_pos
        if default_base_orientation is None:
            default_base_orientation = current_quat
        self.default_start_pose = [
            np.array(default_base_position, dtype=float),
            np.array(default_base_orientation, dtype=float),
        ]

        self.ee_names: List[str] = list(ee_names) if ee_names else []
        self.ee_sites: List[str] = list(ee_sites) if ee_sites else []
        self.ee_ids: List[int] = [self.get_link_id(n) for n in self.ee_names]
        self.ee_site_ids: List[int] = [self.get_site_id(n) for n in self.ee_sites]

        # Force-torque sensors that were actually compiled into the model (see ft_sensor_links).
        self.ft_sensor_links: List[str] = [
            name for name in (ft_sensor_links or []) if f"{name}_force" in self.sensor_name_to_id
        ]
        self._ft_enabled_links: set = set(self.ft_sensor_links)

        # Optional force-torque smoothing: a moving-window buffer per instrumented link, filled by
        # a post-step callable. get_ft_reading returns the window mean when enabled.
        self._ft_smoothing_window: int = ft_smoothing_window
        self._ft_buffers: Dict[str, deque] = {}
        if smooth_ft and self.ft_sensor_links:
            self._ft_buffers = {
                link: deque(maxlen=ft_smoothing_window) for link in self.ft_sensor_links
            }
            self.add_post_step_callable({"_ft_smoothing": [self._record_ft, []]})

        joints_given = default_joint_positions is not None
        if joints_given:
            self.default_joint_positions = np.asarray(default_joint_positions, dtype=float)
        elif wrapping:
            self.default_joint_positions = np.array(
                [self.data.qpos[adr] for adr in self.actuated_joint_qpos_adrs]
            )
        else:
            self.default_joint_positions = np.zeros(self.num_actuated_joints)

        # In wrapping mode the world's current state is left exactly as it is unless an initial
        # base pose or joint configuration was asked for.
        if (wrapping and base_given) or (not wrapping and self.has_floating_base):
            self.reset_base_pose(self.default_start_pose[0], self.default_start_pose[1])
        if joints_given or not wrapping:
            self.reset_joints(self.actuated_joint_ids, self.default_joint_positions)

        # Enter the initial control mode (position mode holds the current configuration). A
        # wrapped robot keeps the host's ctrl unless its joints were just reset.
        if self._enable_torque_mode:
            self.set_torque_control_mode()
        elif joints_given or not wrapping:
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
        """Advance the simulation by one timestep (only in synchronous mode).

        Runs the registered pre-step callables, applies control, steps, then runs the post-step
        callables and syncs the viewer.

        Use this only if this robot owns stepping. A wrapped robot inside a host's world should
        leave stepping to the host (which steps the whole world once per tick); calling this as
        well would advance the world twice.
        """
        if self.sync_mode:
            self._locked_step()

    def _locked_step(self):
        """One step with hooks + control, holding the lock so it is safe against the async thread."""
        with self._lock:
            self._run_callables(self._pre_step_callables)
            self.apply_managed_control()
            mujoco.mj_step(self.model, self.data)
            self._run_callables(self._post_step_callables)
            if self._viewer is not None:
                self._viewer.sync()

    def add_pre_step_callable(self, f_dict: dict):
        """Register callables to run before each ``mj_step``.

        Args:
            f_dict (dict): ``{name: [callable, [args]]}``. A name already registered is skipped
                (existing entry kept), so callables are de-duplicated by name.
        """
        self._register_callables(self._pre_step_callables, f_dict)

    def add_post_step_callable(self, f_dict: dict):
        """Register callables to run after each ``mj_step``.

        Args:
            f_dict (dict): ``{name: [callable, [args]]}``. A name already registered is skipped
                (existing entry kept), so callables are de-duplicated by name.
        """
        self._register_callables(self._post_step_callables, f_dict)

    @staticmethod
    def _register_callables(store: dict, f_dict: dict):
        for name, spec in f_dict.items():
            if name in store:
                continue  # dedup by name: keep the existing callable
            store[name] = (spec[0], spec[1] if len(spec) > 1 else [])

    @staticmethod
    def _run_callables(store: dict):
        for fn, args in store.values():
            fn(*args)

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
            self._locked_step()
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

    @property
    def viewer(self):
        """The passive viewer handle (or None if not launched)."""
        return self._viewer

    def shutdown(self):
        """Stop the background thread (if any) and close the viewer (if any).

        Only a viewer opened by :meth:`launch_viewer` is held here, so a host's own viewer is
        never closed. Registered with ``atexit``.
        """
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
        joint_ids: List[int | str],
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray = None,
    ):
        """Set the positions (and optionally velocities) of the given (1-dof) joints.

        Writes ``qpos``/``qvel`` directly and recomputes derived quantities with ``mj_forward``
        (no dynamics stepping). Only this robot's actuated joints may be given.

        Args:
            joint_ids (List[int | str]): Joint ids or names to set.
            joint_positions (np.ndarray): Target positions, aligned with ``joint_ids``.
            joint_velocities (np.ndarray, optional): Target velocities. Defaults to zeros.
        """
        joint_positions = np.asarray(joint_positions, dtype=float)
        infos = [self._own_joint_info(j) for j in joint_ids]
        with self._lock:
            for n, info in enumerate(infos):
                self.data.qpos[info.qpos_adr] = joint_positions[n]
                if joint_velocities is not None:
                    self.data.qvel[info.dof_adr] = joint_velocities[n]
                elif info.dof_dim == 1:
                    self.data.qvel[info.dof_adr] = 0.0
            mujoco.mj_forward(self.model, self.data)

    def reset_actuated_joint_positions(
        self,
        joint_positions: np.ndarray = None,
        joint_names: List[str] = None,
        joint_velocities: np.ndarray = None,
    ):
        """Reset the actuated joints to the given positions (no controller; ``qpos`` + forward).

        Args:
            joint_positions (np.ndarray, optional): Target positions, in the order of
                ``joint_names``. Defaults to ``default_joint_positions``.
            joint_names (List[str | int], optional): Joint names or ids to set, in the order of
                ``joint_positions``. Defaults to all actuated joints.
            joint_velocities (np.ndarray, optional): Target velocities. Defaults to zeros.
        """
        if joint_positions is None:
            joint_positions = self.default_joint_positions
        if joint_names is None:
            joint_names = self.actuated_joint_names
        self.reset_joints(joint_names, joint_positions, joint_velocities)

    def reset_base_pose(
        self,
        position: Vector3D = None,
        orientation: QuatType = None,
        base_linear_velocity: Vector3D = None,
        base_angular_velocity: Vector3D = None,
    ):
        """Set the base pose (and optionally base velocity for a floating base).

        For a floating base this writes the free-joint ``qpos``/``qvel``. For a fixed base it
        offsets the root body's placement in the model (``body_pos``/``body_quat``, converted
        into the parent body's frame when the root is not a child of the world). Either way,
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
                info = self._joint_info_by_id[self.free_joint_id]
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
                local_pos, local_quat = position, xyzw_to_wxyz(orientation)
                parent = int(self.model.body_parentid[self.base_id])
                if parent != 0:
                    parent_pos, parent_quat = self._kinematic_body_pose(parent)
                    inv_pos, inv_quat = np.zeros(3), np.zeros(4)
                    mujoco.mju_negPose(inv_pos, inv_quat, parent_pos, parent_quat)
                    local_pos, quat = np.zeros(3), np.zeros(4)
                    mujoco.mju_mulPose(local_pos, quat, inv_pos, inv_quat, position, local_quat)
                    local_quat = quat
                self.model.body_pos[self.base_id] = local_pos
                self.model.body_quat[self.base_id] = local_quat
            mujoco.mj_forward(self.model, self.data)

    def _kinematic_body_pose(self, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """World pose ``(pos, quat_wxyz)`` of a body at the current ``qpos``, without touching data.

        Runs forward kinematics on the scratch copy, so it is valid even before the host has
        called ``mj_forward`` on the live data.
        """
        scratch = self._get_scratch_data()
        scratch.qpos[:] = self.data.qpos
        mujoco.mj_kinematics(self.model, scratch)
        return np.array(scratch.xpos[body_id]), np.array(scratch.xquat[body_id])

    # --- placement ----------------------------------------------------------

    def _ground_geom_ids(self) -> set:
        plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
        return {i for i in range(self.model.ngeom) if int(self.model.geom_type[i]) == plane}

    def _touching_ground(self, ground_ids: set) -> bool:
        """Whether any of the robot's own geoms touches one of ``ground_ids``."""
        own = set(self.robot_geom_ids)
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if (g1 in ground_ids and g2 in own) or (g2 in ground_ids and g1 in own):
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
        without an actuator a PD controller drives ``data.qfrc_applied`` each step (if
        ``managed_pd`` is enabled).
        """
        with self._lock:
            self._set_own_actuation(True)
            self._zero_own_qfrc_applied()
            self._managed_targets.clear()
            self._in_torque_mode = False
            for jid in self.actuated_joint_ids:
                info = self._joint_info_by_id[jid]
                if info.actuator_id is not None:
                    self.data.ctrl[info.actuator_id] = self.data.qpos[info.qpos_adr]

    def set_torque_control_mode(self):
        """Switch to effort/torque control: model actuators are disabled.

        The robot will not hold itself up unless torque commands are sent each control cycle.
        Commands are applied via ``data.qfrc_applied``.

        A scoped robot (``joint_names`` given) disables only its own actuators, by clamping
        their force range to zero, so the rest of the world keeps its actuation. Otherwise the
        model-wide actuation flag is used.
        """
        with self._lock:
            self._set_own_actuation(False)
            self._zero_own_qfrc_applied()
            self._managed_targets.clear()
            self._in_torque_mode = True

    def _set_own_actuation(self, enabled: bool):
        """Enable or disable the robot's actuators (and zero their ctrl when disabling)."""
        if not self.is_scoped:
            flag = int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
            if enabled:
                self.model.opt.disableflags &= ~flag
            else:
                self.model.opt.disableflags |= flag
                self.data.ctrl[:] = 0.0
            return
        if enabled:
            for a, (limited, frc_range) in self._saved_actuator_limits.items():
                self.model.actuator_forcelimited[a] = limited
                self.model.actuator_forcerange[a] = frc_range
            self._saved_actuator_limits.clear()
            return
        for a in self.robot_actuator_ids:
            if a not in self._saved_actuator_limits:
                self._saved_actuator_limits[a] = (
                    int(self.model.actuator_forcelimited[a]),
                    np.array(self.model.actuator_forcerange[a]),
                )
            self.model.actuator_forcelimited[a] = 1
            self.model.actuator_forcerange[a] = 0.0
            self.data.ctrl[a] = 0.0

    def _zero_own_qfrc_applied(self):
        """Zero ``qfrc_applied`` on the robot's dofs (all dofs if the robot is the whole model)."""
        if self.is_scoped:
            self.data.qfrc_applied[self.actuated_joint_dof_adrs] = 0.0
        else:
            self.data.qfrc_applied[:] = 0.0

    @property
    def in_torque_mode(self) -> bool:
        """Whether the robot is currently in effort/torque control mode."""
        return self._in_torque_mode

    def _own_joint_info(self, joint: int | str) -> MujocoJointInfo:
        """Info for one of this robot's actuated joints, by name or id; others raise ValueError."""
        try:
            info = self._joint_info(joint)
        except KeyError:
            info = None
        if info is None or info.joint_id not in self._actuated_joint_id_set:
            raise ValueError(f"{joint!r} is not an actuated joint of this robot.")
        return info

    def _set_managed_target(self, info: MujocoJointInfo, q_des: Optional[float], dq_des: float):
        if not self.managed_pd:
            raise ValueError(
                f"Joint '{info.joint_name}' has no usable actuator for this command and managed_pd "
                "is off. Add an actuator (e.g. model_builder.add_position_actuators) or construct "
                "with managed_pd=True."
            )
        self._managed_targets[info.joint_id] = (q_des, dq_des)

    def apply_managed_control(self):
        """Apply PD control (via ``qfrc_applied``) for joints tracked without a model actuator.

        Called automatically before each ``mj_step`` by :meth:`step` and the background thread.
        A wrapped robot with ``managed_pd=True`` needs the host to call it before each of its own
        ``mj_step`` calls. Does nothing when there are no managed targets (e.g. when every
        actuated joint has a position actuator). Only the robot's own dofs are written.
        """
        with self._lock:
            if not self._managed_targets:
                return
            kp, kd = self.default_position_kp, self.default_position_kd
            for jid, (q_des, dq_des) in self._managed_targets.items():
                info = self._joint_info_by_id[jid]
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
        are driven toward the target by a PD controller on ``data.qfrc_applied`` each step (only
        if ``managed_pd`` is enabled; otherwise this raises).

        Args:
            cmd (np.ndarray): Target joint positions.
            actuated_joint_names (List[str | int], optional): Joint names or ids to command, in
                the order of ``cmd``.
                Defaults to all actuated joints.
            vels (np.ndarray, optional): Optional target velocities (used by the PD fallback).
        """
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        cmd = self._normalise_cmd(cmd, len(actuated_joint_names))
        infos = [self._own_joint_info(name) for name in actuated_joint_names]
        with self._lock:
            for i, info in enumerate(infos):
                dq_des = float(vels[i]) if vels is not None else 0.0
                if not self._in_torque_mode and info.actuator_id is not None:
                    self.data.ctrl[info.actuator_id] = cmd[i]
                    self._managed_targets.pop(info.joint_id, None)
                else:
                    self._set_managed_target(info, float(cmd[i]), dq_des)

    def set_joint_velocities(self, cmd: np.ndarray, actuated_joint_names: List[str] = None):
        """Command target velocities for the given actuated joints (velocity control).

        Implemented as a damping PD toward the target velocity via ``data.qfrc_applied`` (gain
        ``default_position_kd``), since MuJoCo models rarely ship velocity actuators. Needs
        ``managed_pd`` to be enabled.

        Args:
            cmd (np.ndarray): Target joint velocities.
            actuated_joint_names (List[str | int], optional): Joint names or ids to command, in
                the order of ``cmd``.
        """
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        cmd = self._normalise_cmd(cmd, len(actuated_joint_names))
        infos = [self._own_joint_info(name) for name in actuated_joint_names]
        with self._lock:
            for i, info in enumerate(infos):
                self._set_managed_target(info, None, float(cmd[i]))

    def set_joint_torques(self, cmd: np.ndarray, actuated_joint_names: List[str] = None):
        """Command target torques/efforts for the given actuated joints (effort control).

        Writes ``data.qfrc_applied`` directly; intended for use in torque control mode.

        Args:
            cmd (np.ndarray): Target joint torques.
            actuated_joint_names (List[str | int], optional): Joint names or ids to command, in
                the order of ``cmd``.
        """
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        cmd = self._normalise_cmd(cmd, len(actuated_joint_names))
        infos = [self._own_joint_info(name) for name in actuated_joint_names]
        with self._lock:
            for i, info in enumerate(infos):
                self.data.qfrc_applied[info.dof_adr] = cmd[i]
                self._managed_targets.pop(info.joint_id, None)

    def set_joint_positions_delta(self, cmd: np.ndarray, actuated_joint_names: List[str] = None):
        """Command a position increment relative to the current positions (experimental).

        Args:
            cmd (np.ndarray): Position deltas.
            actuated_joint_names (List[str | int], optional): Joint names or ids to command, in
                the order of ``cmd``.
        """
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        current = self.get_actuated_joint_positions(actuated_joint_names)
        self.set_joint_positions(
            current + self._normalise_cmd(cmd, len(actuated_joint_names)), actuated_joint_names
        )

    def compute_joint_pd_error(
        self,
        joint_ids: List[int | str],
        q_des: np.ndarray,
        dq_des: np.ndarray,
        Kp: float | np.ndarray,
        Kd: float | np.ndarray,
    ) -> np.ndarray:
        """Compute a joint-space PD term: ``Kp * (q_des - q) + Kd * (dq_des - dq)``.

        Continuous joints use the wrapped position error. A None target zeroes that term.

        Args:
            joint_ids (List[int | str]): Joint ids or names to compute the error for.
            q_des (np.ndarray): Desired positions (or None to skip the position term).
            dq_des (np.ndarray): Desired velocities (or None to skip the velocity term).
            Kp (float | np.ndarray): Position gain(s).
            Kd (float | np.ndarray): Velocity gain(s).

        Returns:
            np.ndarray: The PD term per joint.
        """
        joint_ids = self._resolve_joint_ids(joint_ids)
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
            actuated_joint_names (List[str | int], optional): Joint names or ids to command.
                Defaults to all.
            q (float | np.ndarray, optional): Position command(s). Defaults to 0.
            dq (float | np.ndarray, optional): Velocity command(s). Defaults to 0.
            Kp (float | np.ndarray, optional): Stiffness gain(s). Defaults to 0.
            Kd (float | np.ndarray, optional): Damping gain(s). Defaults to 0.
            tau (float | np.ndarray, optional): Feedforward torque(s). Defaults to 0.
        """
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        joint_ids = self._resolve_joint_ids(actuated_joint_names)  # KeyError if unknown
        infos = [self._own_joint_info(jid) for jid in joint_ids]  # ValueError if not the robot's
        if self._in_torque_mode:
            tau_cmd = self.compute_joint_pd_error(joint_ids, q_des=q, dq_des=dq, Kp=Kp, Kd=Kd) + tau
            with self._lock:
                for i, info in enumerate(infos):
                    self.data.qfrc_applied[info.dof_adr] = tau_cmd[i]
                    self._managed_targets.pop(info.joint_id, None)
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
        self, joint_ids: List[int | str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get (positions, velocities, applied efforts) for the given joints.

        Continuous-joint positions are wrapped to [-pi, pi]. Effort is the applied actuation
        generalised force on the joint (``qfrc_actuator`` + ``qfrc_applied``).

        Args:
            joint_ids (List[int | str], optional): Joint ids or names to read. Defaults to all
                actuated joints.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: positions, velocities, efforts.
        """
        if joint_ids is None:
            joint_ids = self.actuated_joint_ids
        infos = [self._joint_info(j) for j in joint_ids]
        n = len(infos)
        q, v, tau = np.zeros(n), np.zeros(n), np.zeros(n)
        with self._lock:
            for i, info in enumerate(infos):
                pos = self.data.qpos[info.qpos_adr]
                q[i] = wrap_angle(pos) if info.joint_id in self.continuous_joint_ids else pos
                v[i] = self.data.qvel[info.dof_adr]
                tau[i] = (
                    self.data.qfrc_actuator[info.dof_adr]
                    + self.data.qfrc_applied[info.dof_adr]
                )
        return q, v, tau

    def get_actuated_joint_positions(
        self, actuated_joint_names: List[str | int] = None
    ) -> np.ndarray:
        """Get current positions of the given joints (names or ids; defaults to all actuated)."""
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        return self.get_joint_states(actuated_joint_names)[0]

    def get_actuated_joint_velocities(
        self, actuated_joint_names: List[str | int] = None
    ) -> np.ndarray:
        """Get current velocities of the given joints (names or ids; defaults to all actuated)."""
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        return self.get_joint_states(actuated_joint_names)[1]

    def get_actuated_joint_torques(
        self, actuated_joint_names: List[str | int] = None
    ) -> np.ndarray:
        """Get current efforts of the given joints (names or ids; defaults to all actuated)."""
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        return self.get_joint_states(actuated_joint_names)[2]

    def get_actuated_joint_accelerations(
        self, actuated_joint_names: List[str | int] = None
    ) -> np.ndarray:
        """Get current accelerations (``data.qacc``) of the given joints (names or ids)."""
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        with self._lock:
            return np.array(
                [self.data.qacc[self._joint_info(j).dof_adr] for j in actuated_joint_names]
            )

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

    def get_link_pose(self, link_id: int | str) -> Tuple[Vector3D, QuatType]:
        """Get the world-frame link (body) pose as (position, orientation quaternion [x,y,z,w]).

        Args:
            link_id (int | str): Link (body) id or name.
        """
        link_id = self._resolve_link_id(link_id)
        with self._lock:
            return np.array(self.data.xpos[link_id]), wxyz_to_xyzw(self.data.xquat[link_id])

    def get_link_velocity(self, link_id: int | str) -> Tuple[Vector3D, Vector3D]:
        """Get the world-frame link velocity as (linear velocity, angular velocity).

        Args:
            link_id (int | str): Link (body) id or name.
        """
        return self._object_velocity(self._resolve_link_id(link_id))

    def get_link_com_pose(self, link_id: int | str) -> Tuple[Vector3D, QuatType]:
        """Get the world-frame pose of the link's centre of mass as (position, orientation).

        Args:
            link_id (int | str): Link (body) id or name.
        """
        link_id = self._resolve_link_id(link_id)
        quat = np.zeros(4)
        with self._lock:
            pos = np.array(self.data.xipos[link_id])
            mujoco.mju_mat2Quat(quat, self.data.ximat[link_id])
        return pos, wxyz_to_xyzw(quat)

    def get_site_pose(self, site_id: int | str) -> Tuple[Vector3D, QuatType]:
        """Get the world-frame pose of a site as (position, orientation quaternion [x,y,z,w]).

        Args:
            site_id (int | str): Site id or name.
        """
        site_id = self._resolve_site_id(site_id)
        quat = np.zeros(4)
        with self._lock:
            pos = np.array(self.data.site_xpos[site_id])
            mujoco.mju_mat2Quat(quat, self.data.site_xmat[site_id])
        return pos, wxyz_to_xyzw(quat)

    # --- jacobian -----------------------------------------------------------

    def _as_frame(self, frame: str | Frame) -> Frame:
        """A link-or-site argument as a Frame. Bare ids are refused: they cannot say which kind."""
        if isinstance(frame, Frame):
            return frame
        if isinstance(frame, str):
            return self.frame(frame)
        raise TypeError(
            f"Expected a link/site name or a Frame from robot.frame(name), got {frame!r}. A bare "
            "id is ambiguous between a body id and a site id."
        )

    def _jacobian_from_data(self, data: mujoco.MjData, frame: Frame) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        if frame.kind == "link":
            mujoco.mj_jacBody(self.model, data, jacp, jacr, frame.id)
        else:
            mujoco.mj_jacSite(self.model, data, jacp, jacr, frame.id)
        cols = self.actuated_joint_dof_adrs
        return np.vstack([jacp[:, cols], jacr[:, cols]])

    def get_jacobian(
        self, ee_link_name: str | Frame, joint_angles: np.ndarray = None
    ) -> np.ndarray:
        """Compute the 6xN geometric Jacobian (linear stacked over angular) for a frame.

        The frame may be a link (body) or a site, given by name or as a :class:`Frame` from
        :meth:`frame`. Columns correspond to the actuated joints, in ``actuated_joint_names``
        order.

        Args:
            ee_link_name (str | Frame): Link (body) or site to compute the Jacobian for.
            joint_angles (np.ndarray, optional): Actuated-joint configuration to evaluate at.
                Defaults to None (use the current configuration). Evaluated on a scratch copy so
                the live state is not disturbed.

        Returns:
            np.ndarray: 6xN Jacobian matrix.
        """
        frame = self._as_frame(ee_link_name)
        if joint_angles is None:
            with self._lock:
                return self._jacobian_from_data(self.data, frame)
        with self._lock:
            scratch = self._get_scratch_data()
            scratch.qpos[:] = self.data.qpos
            scratch.qpos[self.actuated_joint_qpos_adrs] = joint_angles
            mujoco.mj_forward(self.model, scratch)
            return self._jacobian_from_data(scratch, frame)

    # --- dynamics -----------------------------------------------------------

    def get_gravity_compensation_torques(self, include_coriolis: bool = False) -> np.ndarray:
        """Joint torques that counteract passive dynamics, per actuated joint (default order).

        The bias force ``qfrc_bias`` includes gravity plus Coriolis / centrifugal terms at the
        current velocity. This method exposes the choice:

        - ``include_coriolis=True``: returns ``data.qfrc_bias`` at the current velocity directly.
        - ``include_coriolis=False`` (default, "hold against gravity"): the pure gravity term,
          computed via inverse dynamics on a scratch copy at zero velocity and acceleration
          (``mj_inverse`` -> ``qfrc_inverse``). The live simulation state is not mutated.

        Returns:
            np.ndarray: Torque per actuated joint (a useful feedforward term for torque/impedance
                control).
        """
        with self._lock:
            if include_coriolis:
                bias = np.array(self.data.qfrc_bias)
            else:
                scratch = self._get_scratch_data()
                scratch.qpos[:] = self.data.qpos
                scratch.qvel[:] = 0.0
                scratch.qacc[:] = 0.0
                mujoco.mj_inverse(self.model, scratch)
                bias = np.array(scratch.qfrc_inverse)
        return np.array(
            [bias[adr] for adr in self.actuated_joint_dof_adrs]
        )

    # --- state bundle -------------------------------------------------------

    def get_robot_states(
        self, actuated_joint_names: List[str | int] = None, ee_names: List[str | int] = None
    ) -> MujocoRobotState:
        """Get a consistent snapshot of common robot state.

        Args:
            actuated_joint_names (List[str | int], optional): Joint names or ids to include.
                Defaults to all. Reported by name in ``joint_order``.
            ee_names (List[str | int], optional): End-effector link names or ids to report.
                Defaults to all. Reported by name in ``ee_order``.

        Returns:
            MujocoRobotState: The state snapshot.
        """
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        if ee_names is None:
            ee_names = self.ee_names
        with self._lock:
            joint_ids = self._resolve_joint_ids(actuated_joint_names)
            ee_ids = [self._resolve_link_id(e) for e in ee_names]
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
                joint_order=[self.joint_id_to_name[j] for j in joint_ids],
                ee_order=[self.link_id_to_name[e] for e in ee_ids],
                ee_contact_states=self.get_contact_states_of_links(ee_ids),
            )

    # --- contacts -----------------------------------------------------------

    def get_contact_states_of_links(self, link_ids: List[int | str]) -> np.ndarray:
        """Binary contact state (1 = in contact, 0 = not) for each of the given links (bodies).

        Args:
            link_ids (List[int | str]): Link (body) ids or names to query.

        Returns:
            np.ndarray: One 0/1 value per link, in the given order.
        """
        link_ids = [self._resolve_link_id(link) for link in link_ids]
        with self._lock:
            touching = set()
            for c in range(self.data.ncon):
                con = self.data.contact[c]
                touching.add(int(self.model.geom_bodyid[con.geom1]))
                touching.add(int(self.model.geom_bodyid[con.geom2]))
        return np.array([int(lid in touching) for lid in link_ids])

    def get_ee_contact_states(self, ee_names: List[str | int] = None) -> np.ndarray:
        """Binary contact state for the given end-effectors (defaults to the configured ones)."""
        if ee_names is None:
            ee_names = self.ee_names
        return self.get_contact_states_of_links(ee_names)

    def get_link_contact_force(self, link) -> Vector3D:
        """Net world-frame contact force acting on the given link (body).

        Sums each contact's force (from ``mj_contactForce``, rotated into the world frame) over the
        contacts that involve the link, with the sign chosen so the result is the force ON the
        link. Zero if the link has no contacts.

        Args:
            link (int | str): Link (body) id or name.

        Returns:
            Vector3D: Net contact force in the world frame.
        """
        lid = self._resolve_link_id(link)
        net = np.zeros(3)
        f6 = np.zeros(6)
        with self._lock:
            for c in range(self.data.ncon):
                con = self.data.contact[c]
                b1 = int(self.model.geom_bodyid[con.geom1])
                b2 = int(self.model.geom_bodyid[con.geom2])
                if lid not in (b1, b2):
                    continue
                mujoco.mj_contactForce(self.model, self.data, c, f6)
                frame = con.frame
                fw = f6[0] * frame[0:3] + f6[1] * frame[3:6] + f6[2] * frame[6:9]
                net += fw if lid == b2 else -fw
        return net

    def get_contact_info(self) -> List[ContactInfo]:
        """Detailed info for every active contact: position, contact frame, and wrench.

        Calls ``mj_rnePostConstraint`` first so the contact forces are available. This is the
        detailed variant; :meth:`get_ee_contact_states` remains the lightweight boolean check.

        Returns:
            List[ContactInfo]: one entry per active contact (empty if none).
        """
        wrench = np.zeros(6)
        contacts: List[ContactInfo] = []
        with self._lock:
            mujoco.mj_rnePostConstraint(self.model, self.data)
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                mujoco.mj_contactForce(self.model, self.data, i, wrench)
                contacts.append(
                    ContactInfo(
                        pos=np.array(contact.pos),
                        frame=np.array(contact.frame).reshape(3, 3),
                        wrench=wrench.copy(),
                    )
                )
        return contacts

    # --- force-torque sensors -----------------------------------------------

    def _ft_link_name(self, link: int | str) -> str:
        """Force-torque sensors are keyed by link name; an id is converted to its body name."""
        return self.body_id_to_name[int(link)] if self._is_id(link) else link

    def has_ft_sensor(self, link_name: int | str) -> bool:
        """Whether a force-torque sensor was compiled in for the given link (name or id)."""
        return f"{self._ft_link_name(link_name)}_force" in self.sensor_name_to_id

    def _read_sensor(self, name: str) -> np.ndarray:
        sid = self.sensor_name_to_id[name]
        adr = int(self.model.sensor_adr[sid])
        dim = int(self.model.sensor_dim[sid])
        return np.array(self.data.sensordata[adr : adr + dim])

    def get_link_ft_measurement(self, link_name: int | str) -> np.ndarray:
        """Force-torque reading for a link as a 6D wrench ``[fx,fy,fz,tx,ty,tz]`` (site frame).

        The link is given by name or id. Returns zeros if the link has no force-torque sensor (see
        ``ft_sensor_links``) or its sensor has been disabled via ``toggle_ft_sensor_for_links``.
        """
        link_name = self._ft_link_name(link_name)
        if link_name not in self._ft_enabled_links or not self.has_ft_sensor(link_name):
            return np.zeros(6)
        with self._lock:
            force = self._read_sensor(f"{link_name}_force")
            torque = self._read_sensor(f"{link_name}_torque")
        return np.concatenate([force, torque])

    def _read_ft_raw(self, link_name: str) -> np.ndarray:
        """Raw 6D sensor reading (force then torque, sensor frame), no sign flip or gating."""
        return np.concatenate(
            [self._read_sensor(f"{link_name}_force"), self._read_sensor(f"{link_name}_torque")]
        )

    def _record_ft(self):
        """Post-step callable: append each instrumented link's raw reading to its smoothing buffer."""
        for link, buffer in self._ft_buffers.items():
            buffer.append(self._read_ft_raw(link))

    def get_ft_reading(
        self, in_global_frame: bool = True, sensor_name: int | str = None
    ) -> Tuple[Vector3D, Vector3D]:
        """Force-torque reading as ``(force[3], torque[3])``, expressed on the parent body.

        The sensor is resolved by name (``sensor_name`` is the instrumented link; defaults to the
        first in ``ft_sensor_links``) rather than by index. The raw sensor value is negated so the
        result is the wrench acting *on* the parent body. If smoothing is enabled the moving-window
        mean is used.

        Args:
            in_global_frame (bool, optional): Rotate the wrench into the world frame using the
                sensor site's orientation. If the site is unavailable, warns and returns the
                sensor-frame values. Defaults to True.
            sensor_name (int | str, optional): Instrumented link name or id. Defaults to the first
                instrumented
                link.

        Returns:
            Tuple[Vector3D, Vector3D]: (force, torque).
        """
        link = self._ft_link_name(sensor_name) if sensor_name is not None else (
            self.ft_sensor_links[0] if self.ft_sensor_links else None
        )
        if link is None or not self.has_ft_sensor(link):
            raise ValueError(
                "No force-torque sensor available; construct with ft_sensor_links=[...]."
            )
        with self._lock:
            buffer = self._ft_buffers.get(link)
            raw = np.mean(np.stack(buffer), axis=0) if buffer else self._read_ft_raw(link)
            # Negate to express the wrench on the parent body.
            force, torque = -raw[:3], -raw[3:]
            if in_global_frame:
                site_id = self.site_name_to_id.get(f"{link}_ft_site")
                if site_id is None:
                    warnings.warn(
                        f"No site '{link}_ft_site'; returning sensor-frame force-torque.",
                        stacklevel=2,
                    )
                else:
                    rot = np.array(self.data.site_xmat[site_id]).reshape(3, 3)
                    force, torque = rot @ force, rot @ torque
        return force, torque

    def get_joint_ft_measurements(self, joint_ids: List[int | str] = None) -> np.ndarray:
        """Force-torque wrench transmitted at each joint (measured at the joint's child link).

        Args:
            joint_ids (List[int | str], optional): Joint ids or names to query. Defaults to all
                actuated joints.

        Returns:
            np.ndarray: An (N, 6) array of ``[fx,fy,fz,tx,ty,tz]`` per joint (zeros where no
                sensor is present).
        """
        if joint_ids is None:
            joint_ids = self.actuated_joint_ids
        infos = [self._joint_info(j) for j in joint_ids]
        out = np.zeros((len(infos), 6))
        for i, info in enumerate(infos):
            out[i] = self.get_link_ft_measurement(info.body_name)
        return out

    def get_full_joint_states(
        self, joint_ids: List[int | str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get joint positions, velocities, force-torque measurements, and efforts.

        Args:
            joint_ids (List[int | str], optional): Joint ids or names to read. Defaults to all
                actuated joints.

        Returns:
            Tuple: (positions, velocities, (N,6) force-torque measurements, efforts).
        """
        q, v, tau = self.get_joint_states(joint_ids)
        ft = self.get_joint_ft_measurements(joint_ids)
        return q, v, ft, tau

    def toggle_ft_sensor_for_links(self, link_names, enable: bool = True):
        """Enable/disable reading of the (compile-time) force-torque sensors for the given links.

        MuJoCo sensors cannot be added or removed at runtime; this only gates whether the getters
        return the sensor value or zeros. Declare the sensors up front with ``ft_sensor_links``.

        Args:
            link_names (str | int | List[str | int]): Link name(s) or id(s).
            enable (bool, optional): True to read, False to return zeros. Defaults to True.
        """
        if isinstance(link_names, str) or self._is_id(link_names):
            link_names = [link_names]
        for name in (self._ft_link_name(link) for link in link_names):
            if enable and self.has_ft_sensor(name):
                self._ft_enabled_links.add(name)
            else:
                self._ft_enabled_links.discard(name)

    def toggle_ft_sensor_for_joints(self, joint_ids, enable: bool = True):
        """Enable/disable the force-torque sensors at the given joints' child links.

        See ``toggle_ft_sensor_for_links``; sensors are compile-time, so this only gates reading.

        Args:
            joint_ids (int | str | List[int | str]): Joint id(s) or name(s).
            enable (bool, optional): True to read, False to return zeros. Defaults to True.
        """
        if isinstance(joint_ids, str) or self._is_id(joint_ids):
            joint_ids = [joint_ids]
        links = [self._joint_info(j).body_name for j in joint_ids]
        self.toggle_ft_sensor_for_links(links, enable)

    # --- cosmetics / dynamics -----------------------------------------------

    def set_robot_transparency(self, alpha: float):
        """Set the visual transparency of the robot's geoms.

        Mutates ``model.geom_rgba[:, 3]`` for the robot's geoms; the change is picked up live by
        the viewer on the next sync. Note that geoms that draw from a material/texture may ignore
        ``geom_rgba``, so transparency is most reliable on primitive/rgba geoms.

        Args:
            alpha (float): Opacity from 0 (invisible) to 1 (opaque).
        """
        with self._lock:
            for g in self.robot_geom_ids:
                self.model.geom_rgba[g, 3] = alpha

    def set_geom_collision(self, geom, enabled: bool):
        """Enable or disable collisions for a single geom.

        Disabling zeroes the geom's ``contype``/``conaffinity`` (saving the originals); enabling
        restores the saved masks. Useful for making a reference/preview robot pass through the
        world without interacting.

        Args:
            geom (int | str): Geom id or name.
            enabled (bool): True to collide, False to disable collisions.
        """
        gid = self._resolve_geom_id(geom)
        with self._lock:
            if enabled:
                if gid in self._saved_geom_masks:
                    contype, conaffinity = self._saved_geom_masks.pop(gid)
                    self.model.geom_contype[gid] = contype
                    self.model.geom_conaffinity[gid] = conaffinity
            else:
                if gid not in self._saved_geom_masks:
                    self._saved_geom_masks[gid] = (
                        int(self.model.geom_contype[gid]),
                        int(self.model.geom_conaffinity[gid]),
                    )
                self.model.geom_contype[gid] = 0
                self.model.geom_conaffinity[gid] = 0

    # --- dynamics editing (domain randomisation) ----------------------------
    #
    # These edit the compiled model directly, addressed by MuJoCo object type and MuJoCo field
    # names. Elements are given by name or id (ids reach unnamed geoms). They are restricted to the
    # robot's own elements and are safe to call between steps.

    def _robot_geom_id(self, geom: int | str) -> int:
        gid = int(geom) if self._is_id(geom) else self.geom_name_to_id.get(geom)
        if gid is None or gid not in self.robot_geom_ids:
            raise ValueError(f"{geom!r} is not a geom belonging to this robot.")
        return gid

    def _robot_body_id(self, body: int | str) -> int:
        bid = int(body) if self._is_id(body) else self.body_name_to_id.get(body)
        if bid is None or bid not in self.link_ids:
            raise ValueError(f"{body!r} is not a body belonging to this robot.")
        return bid

    def _robot_joint_info(self, joint: int | str) -> MujocoJointInfo:
        try:
            info = self._joint_info(joint)
        except KeyError:
            info = None
        if info is None or info.body_id not in self.link_ids:
            raise ValueError(f"{joint!r} is not a joint belonging to this robot.")
        return info

    def _write_geom_friction(self, geom_name, friction):
        gid = self._robot_geom_id(geom_name)
        friction = np.asarray(friction, dtype=float)
        if friction.ndim == 0:
            self.model.geom_friction[gid, 0] = float(friction)  # scalar -> sliding friction
        else:
            self.model.geom_friction[gid] = friction  # [sliding, torsional, rolling]

    def _write_dof_field(self, field: str, joint_name: str, value):
        info = self._robot_joint_info(joint_name)
        if info.dof_dim > 1 and np.ndim(value) == 0:
            raise ValueError(
                f"Joint '{joint_name}' has {info.dof_dim} dofs; pass a per-dof array for {field}."
            )
        getattr(self.model, field)[info.dof_adr : info.dof_adr + info.dof_dim] = value

    def _write_body_mass(self, body_name, mass, scale_inertia: bool = True):
        self._write_body_mass_by_id(self._robot_body_id(body_name), mass, scale_inertia)

    def _write_body_mass_by_id(self, bid: int, mass, scale_inertia: bool = True):
        old = float(self.model.body_mass[bid])
        self.model.body_mass[bid] = mass
        if scale_inertia and old > 0.0:
            self.model.body_inertia[bid] = self.model.body_inertia[bid] * (float(mass) / old)
        # Keep subtree masses consistent: the delta propagates to this body and every ancestor.
        delta = float(mass) - old
        b = bid
        while True:
            self.model.body_subtreemass[b] += delta
            if b == 0:
                break
            b = int(self.model.body_parentid[b])

    def set_geom_friction(self, geom_name: int | str, friction):
        """Set a robot geom's friction. Scalar sets sliding friction; a 3-vector sets all three."""
        with self._lock:
            self._write_geom_friction(geom_name, friction)
            mujoco.mj_forward(self.model, self.data)

    def set_geom_solref(self, geom_name: int | str, solref):
        """Set a robot geom's contact solver reference ``[timeconst, dampratio]`` (contact softness)."""
        gid = self._robot_geom_id(geom_name)
        with self._lock:
            self.model.geom_solref[gid] = solref
            mujoco.mj_forward(self.model, self.data)

    def set_geom_solimp(self, geom_name: int | str, solimp):
        """Set a robot geom's contact solver impedance (5-vector)."""
        gid = self._robot_geom_id(geom_name)
        with self._lock:
            self.model.geom_solimp[gid] = solimp
            mujoco.mj_forward(self.model, self.data)

    def set_dof_damping(self, joint_name: int | str, value):
        """Set the joint's dof damping (per-dof array required for multi-dof joints)."""
        with self._lock:
            self._write_dof_field("dof_damping", joint_name, value)
            mujoco.mj_forward(self.model, self.data)

    def set_dof_frictionloss(self, joint_name: int | str, value):
        """Set the joint's dof friction loss (per-dof array required for multi-dof joints)."""
        with self._lock:
            self._write_dof_field("dof_frictionloss", joint_name, value)
            mujoco.mj_forward(self.model, self.data)

    def set_dof_armature(self, joint_name: int | str, value):
        """Set the joint's dof armature (per-dof array required for multi-dof joints)."""
        with self._lock:
            self._write_dof_field("dof_armature", joint_name, value)
            mujoco.mj_forward(self.model, self.data)

    def set_body_mass(self, body_name: int | str, mass: float, *, scale_inertia: bool = True):
        """Set a robot body's mass (cheap live path for small perturbations / randomisation).

        Scales the body's inertia by the same ratio (unless ``scale_inertia=False``) and fixes the
        subtree masses so the model stays consistent, then calls ``mj_forward``.

        WARNING: large mass ratios can produce a near-singular inertia and NaN accelerations. For
        large changes or whole-model rescaling, rebuild the model instead (edit an ``MjSpec`` and
        recompile), or use :meth:`set_total_mass`.
        """
        with self._lock:
            self._write_body_mass(body_name, mass, scale_inertia)
            mujoco.mj_forward(self.model, self.data)

    def set_total_mass(self, total: float):
        """Rescale the robot so its total mass equals ``total``.

        When the robot is the whole model this is ``mj_setTotalmass``. A scoped robot rescales
        only its own bodies (mass and inertia), leaving the rest of the world unchanged.
        """
        with self._lock:
            if self.is_scoped:
                current = float(sum(self.model.body_mass[b] for b in self.link_ids))
                if current <= 0.0:
                    raise ValueError("The robot has no mass to rescale.")
                ratio = float(total) / current
                for b in self.link_ids:
                    self._write_body_mass_by_id(b, self.model.body_mass[b] * ratio)
            else:
                mujoco.mj_setTotalmass(self.model, total)
            mujoco.mj_forward(self.model, self.data)

    def randomize_dynamics(
        self,
        *,
        geom_friction: dict = None,
        geom_solref: dict = None,
        geom_solimp: dict = None,
        dof_damping: dict = None,
        dof_frictionloss: dict = None,
        dof_armature: dict = None,
        body_mass: dict = None,
    ):
        """Batch dynamics randomisation. Each argument is a ``{element: value}`` dict.

        Dispatches over the per-field setters (geom fields keyed by geom, dof fields by joint,
        ``body_mass`` by body, each as a name or id; mass uses the cheap live path), then does a
        single ``mj_forward`` at the end rather than one per field.
        """
        with self._lock:
            for name, value in (geom_friction or {}).items():
                self._write_geom_friction(name, value)
            for name, value in (geom_solref or {}).items():
                self.model.geom_solref[self._robot_geom_id(name)] = value
            for name, value in (geom_solimp or {}).items():
                self.model.geom_solimp[self._robot_geom_id(name)] = value
            for name, value in (dof_damping or {}).items():
                self._write_dof_field("dof_damping", name, value)
            for name, value in (dof_frictionloss or {}).items():
                self._write_dof_field("dof_frictionloss", name, value)
            for name, value in (dof_armature or {}).items():
                self._write_dof_field("dof_armature", name, value)
            for name, value in (body_mass or {}).items():
                self._write_body_mass(name, value, scale_inertia=True)
            mujoco.mj_forward(self.model, self.data)

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
