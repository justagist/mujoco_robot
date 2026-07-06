"""Single-arm inverse-kinematics demo for mujoco_robot.

A Universal Robots UR5e tracks a moving Cartesian target with its tool flange, using the
``differential_ik`` helper each frame (damped least squares, single site, no extra dependencies).
This mirrors the original headless/interactive single-arm IK demos.

Run:
    python demo_single_arm_ik.py            # with the viewer
    python demo_single_arm_ik.py --headless # prints tracking only
"""

import argparse
import time

import numpy as np

from mujoco_robot import MujocoRobot
from mujoco_robot.ik import differential_ik
from mujoco_robot.utils.robot_loader_utils import get_mjcf_from_awesome_robot_descriptions

# UR5e "home" keyframe (6 revolute joints; no gripper). It ships with an "attachment_site".
HOME = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
EE_SITE = "attachment_site"
TRAJ_RADIUS = 0.15  # metres
TRAJ_PERIOD = 6.0  # seconds per revolution


def main(render: bool = True, duration: float = None):
    robot = MujocoRobot(
        mjcf_path=get_mjcf_from_awesome_robot_descriptions("ur5e_mj_description"),
        ee_names=["wrist_3_link"],
        default_joint_positions=HOME,
        run_async=False,
        place_on_ground=False,
        load_ground_plane=True,
        render=render,
        verbose=True,
    )
    robot.set_position_control_mode()

    site_id = robot.get_site_id(EE_SITE)
    centre, fixed_ori = robot.get_site_pose(site_id)
    centre = centre.copy()
    fixed_ori = fixed_ori.copy()
    omega = 2.0 * np.pi / TRAJ_PERIOD
    dt = robot.get_timestep()

    def keep_running(elapsed: float) -> bool:
        if render:
            return robot.is_viewer_running()
        return duration is None or elapsed < duration

    print("\nTracking a circular target with differential IK (position control). Ctrl+C to exit.\n")
    start = time.time()
    last_log = 0.0
    try:
        while keep_running(time.time() - start):
            t = time.time() - start
            target = centre + np.array(
                [0.0, TRAJ_RADIUS * np.sin(omega * t), TRAJ_RADIUS * (1.0 - np.cos(omega * t))]
            )
            # UR5e has no gripper, so every actuated joint is an arm joint.
            q = differential_ik(robot.model, robot.data, EE_SITE, target, target_quat=fixed_ori)
            robot.set_joint_positions(q)
            robot.step()

            now = time.time()
            if now - last_log > 1.0:
                reached = robot.get_site_pose(site_id)[0]
                error = np.linalg.norm(reached - target)
                print(f"t={t:5.1f}s | target={np.round(target, 3)} | error={error:.4f} m")
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
