"""Differential inverse-kinematics helper (damped least squares) for a single site.

Standalone: depends only on ``mujoco`` / ``numpy``. This solves a single-frame, fixed-base target.
Whole-body / floating-base / multi-frame IK is not implemented here; MuJoCo already gives the
Jacobian for free and `mink <https://github.com/kevinzakka/mink>`_ solves the general case (see
``examples/demo_mink_wholebody_ik.py``).

Exported as a helper, not as part of the main object:

    from mujoco_robot.ik import differential_ik
"""

from typing import Sequence

import numpy as np
import mujoco

from .mujoco_robot import xyzw_to_wxyz

_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
_SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)


def _clamp_to_limits(model: mujoco.MjModel, qpos: np.ndarray, joint_ids: Sequence[int] = None):
    """Clamp limited 1-dof joints in ``qpos`` to their ranges (in place).

    Only ``joint_ids`` are clamped if given; otherwise every joint is.
    """
    for j in range(model.njnt) if joint_ids is None else joint_ids:
        if model.jnt_limited[j] and int(model.jnt_type[j]) in (_HINGE, _SLIDE):
            adr = int(model.jnt_qposadr[j])
            lo, hi = model.jnt_range[j]
            qpos[adr] = min(max(qpos[adr], lo), hi)


def _actuated_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Positions of the movable (non-free) joints, in joint-id order (MujocoRobot's order)."""
    return np.array(
        [
            float(data.qpos[int(model.jnt_qposadr[j])])
            for j in range(model.njnt)
            if int(model.jnt_type[j]) != _FREE
        ]
    )


def differential_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
    target_pos,
    target_quat=None,
    *,
    pos_tol: float = 1e-4,
    rot_tol: float = 1e-3,
    max_iters: int = 100,
    damping: float = 1e-2,
    max_step: float = 0.5,
    joint_names: Sequence[str] = None,
) -> np.ndarray:
    """Damped-least-squares inverse kinematics for a single site (fixed-base robots).

    Solves on a COPY of ``data`` (the live simulation is never mutated) for joint positions that
    place the named site at ``target_pos`` and, if given, ``target_quat`` (otherwise the target is
    position-only). Iterates ``dq = Jᵀ (J Jᵀ + damping² I)⁻¹ e`` with a per-step norm clamp, then
    integrates and clamps to joint limits.

    Args:
        model (mujoco.MjModel): The model.
        data (mujoco.MjData): The current state (copied internally; not modified).
        site_name (str): Name of the site to drive to the target.
        target_pos: Desired site position ``[x, y, z]`` in the world.
        target_quat: Desired site orientation quaternion ``[x, y, z, w]``, or None for
            position-only tracking.
        pos_tol (float): Position convergence tolerance (metres).
        rot_tol (float): Orientation convergence tolerance (radians).
        max_iters (int): Maximum iterations.
        damping (float): Damped-least-squares damping factor.
        max_step (float): Maximum L2 norm of a per-iteration joint step (radians).
        joint_names (Sequence[str], optional): Solve over only these hinge/slide joints (e.g. a
            scoped robot's ``actuated_joint_names``); every other joint is held where it is.
            Defaults to None (every joint in the model).

    Returns:
        np.ndarray: Actuated-joint positions in MujocoRobot's joint order (free base excluded),
            ready for ``robot.reset_actuated_joint_positions`` / ``robot.set_joint_positions``.
            With ``joint_names``, the positions of exactly those joints, in that order.
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"No site named '{site_name}'.")

    joint_ids = cols = None
    if joint_names is not None:
        joint_ids = []
        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"No joint named '{name}'.")
            if int(model.jnt_type[jid]) not in (_HINGE, _SLIDE):
                raise ValueError(f"Joint '{name}' is not a hinge or slide joint.")
            joint_ids.append(jid)
        cols = [int(model.jnt_dofadr[j]) for j in joint_ids]

    # Work on a copy so the live sim is untouched.
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    mujoco.mj_forward(model, scratch)

    target_pos = np.asarray(target_pos, dtype=float)
    track_rot = target_quat is not None
    target_wxyz = xyzw_to_wxyz(target_quat) if track_rot else None

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    err = np.zeros(6)
    site_quat = np.zeros(4)
    neg_site_quat = np.zeros(4)
    err_quat = np.zeros(4)

    for _ in range(max_iters):
        err[:3] = target_pos - scratch.site_xpos[site_id]
        if track_rot:
            # World-frame orientation error: target * inv(current), as an angular velocity.
            mujoco.mju_mat2Quat(site_quat, scratch.site_xmat[site_id])
            mujoco.mju_negQuat(neg_site_quat, site_quat)
            mujoco.mju_mulQuat(err_quat, target_wxyz, neg_site_quat)
            mujoco.mju_quat2Vel(err[3:], err_quat, 1.0)

        if np.linalg.norm(err[:3]) < pos_tol and (
            not track_rot or np.linalg.norm(err[3:]) < rot_tol
        ):
            break

        mujoco.mj_jacSite(model, scratch, jacp, jacr, site_id)
        jac = np.vstack([jacp, jacr]) if track_rot else jacp
        if cols is not None:
            jac = jac[:, cols]
        e = err if track_rot else err[:3]

        jjt = jac @ jac.T
        dq = jac.T @ np.linalg.solve(jjt + damping**2 * np.eye(jjt.shape[0]), e)
        norm = np.linalg.norm(dq)
        if norm > max_step:
            dq = dq * (max_step / norm)

        if cols is not None:
            dq_full = np.zeros(model.nv)
            dq_full[cols] = dq
            dq = dq_full
        mujoco.mj_integratePos(model, scratch.qpos, dq, 1.0)
        _clamp_to_limits(model, scratch.qpos, joint_ids)
        mujoco.mj_forward(model, scratch)

    if joint_ids is not None:
        return np.array([float(scratch.qpos[int(model.jnt_qposadr[j])]) for j in joint_ids])
    return _actuated_qpos(model, scratch)
