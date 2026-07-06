"""Whole-body inverse-kinematics demo for mujoco_robot, using mink.

A Unitree Go1 quadruped (floating base) keeps its four feet planted while its trunk tracks a
target that bobs up and down: a squat. This is a whole-body / multi-frame problem (base pose plus
four foot frames solved together), which is exactly what mink solves; mujoco_robot just loads the
robot, applies the solution, and renders.

Requires the optional IK backend:
    pip install "mujoco_robot[ik]"

Run:
    python demo_mink_wholebody_ik.py            # with the viewer
    python demo_mink_wholebody_ik.py --headless # prints trunk height + foot drift only
"""

import argparse
import time

import numpy as np

from mujoco_robot import MujocoRobot
from mujoco_robot.mujoco_robot import wxyz_to_xyzw
from mujoco_robot.utils.robot_loader_utils import get_mjcf_from_awesome_robot_descriptions

FEET = ["FR", "FL", "RR", "RL"]  # Go1 foot sites
SQUAT_DEPTH = 0.08  # metres
SQUAT_PERIOD = 4.0  # seconds per squat cycle


def _apply_config(robot: MujocoRobot, q: np.ndarray):
    """Apply a mink configuration (full qpos) to the robot: base pose + joint positions."""
    robot.reset_base_pose(q[:3], wxyz_to_xyzw(q[3:7]))
    robot.reset_actuated_joint_positions(q[7:])


def main(render: bool = True, duration: float = None):
    try:
        import mink
    except ImportError:
        print('This demo needs mink. Install the optional backend with:  pip install "mujoco_robot[ik]"')
        return

    robot = MujocoRobot(
        mjcf_path=get_mjcf_from_awesome_robot_descriptions("go1_mj_description"),
        use_fixed_base=False,  # floating base
        run_async=False,
        place_on_ground=False,
        load_ground_plane=True,
        render=render,
        verbose=True,
    )
    model = robot.model

    config = mink.Configuration(model)
    config.update_from_keyframe("home")  # a standing pose
    _apply_config(robot, config.q)

    trunk_task = mink.FrameTask("trunk", "body", position_cost=1.0, orientation_cost=1.0, lm_damping=1.0)
    foot_tasks = [
        mink.FrameTask(foot, "site", position_cost=1.0, orientation_cost=0.0, lm_damping=1.0)
        for foot in FEET
    ]
    posture_task = mink.PostureTask(model, cost=1e-2)
    tasks = [trunk_task, *foot_tasks, posture_task]
    for task in tasks:
        task.set_target_from_configuration(config)

    trunk0 = config.get_transform_frame_to_world("trunk", "body")
    rot0 = mink.SO3(trunk0.rotation().wxyz)
    trans0 = trunk0.translation().copy()
    foot0 = {f: config.get_transform_frame_to_world(f, "site").translation().copy() for f in FEET}

    omega = 2.0 * np.pi / SQUAT_PERIOD
    dt = robot.get_timestep()

    def keep_running(elapsed: float) -> bool:
        if render:
            return robot.is_viewer_running()
        return duration is None or elapsed < duration

    print("\nGo1 squatting: trunk tracks a bobbing target while the feet stay planted. Ctrl+C to exit.\n")
    start = time.time()
    last_log = 0.0
    try:
        while keep_running(time.time() - start):
            t = time.time() - start
            dz = -SQUAT_DEPTH * 0.5 * (1.0 - np.cos(omega * t))  # 0 -> down -> 0
            trunk_task.set_target(
                mink.SE3.from_rotation_and_translation(rot0, trans0 + np.array([0.0, 0.0, dz]))
            )
            vel = mink.solve_ik(config, tasks, dt, "daqp", damping=1e-3)
            config.integrate_inplace(vel, dt)
            _apply_config(robot, config.q)
            if render and robot.viewer is not None:
                robot.viewer.sync()

            now = time.time()
            if now - last_log > 0.5:
                trunk_z = config.get_transform_frame_to_world("trunk", "body").translation()[2]
                drift = max(
                    np.linalg.norm(config.get_transform_frame_to_world(f, "site").translation() - foot0[f])
                    for f in FEET
                )
                print(f"t={t:4.1f}s | trunk_z={trunk_z:.3f} m | max foot drift={drift * 1e3:.2f} mm")
                last_log = now
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="run without the passive viewer")
    parser.add_argument(
        "--duration", type=float, default=None, help="seconds to run (headless); default runs 8s"
    )
    args = parser.parse_args()
    main(render=not args.headless, duration=args.duration if args.duration else 8.0)
