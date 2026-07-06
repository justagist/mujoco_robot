"""Joint-space position control demo for mujoco_robot.

Loads a KUKA iiwa14, prints its structure, and drives the arm along a smooth sinusoidal joint
trajectory using position control. A passive viewer shows the motion (unless ``--headless``).

Run:
    python demo_joint_position_control.py            # with the viewer
    python demo_joint_position_control.py --headless # no viewer (prints tracking only)
"""

import argparse
import time

import numpy as np

from mujoco_robot import MujocoRobot
from mujoco_robot.utils.robot_loader_utils import get_mjcf_from_awesome_robot_descriptions

# iiwa14 "home" keyframe (7 revolute joints; no gripper).
HOME = [0.0, 0.785, 0.0, -1.571, 0.0, 0.0, 0.0]
ARM = [f"joint{i}" for i in range(1, 8)]


def main(render: bool = True, duration: float = None):
    robot = MujocoRobot(
        mjcf_path=get_mjcf_from_awesome_robot_descriptions("iiwa14_mj_description"),
        ee_names=["link7"],
        default_joint_positions=HOME,
        run_async=False,  # step the simulation manually in the loop below
        place_on_ground=False,
        load_ground_plane=True,
        render=render,
        verbose=True,
    )
    robot.set_position_control_mode()

    home = robot.get_actuated_joint_positions(ARM).copy()
    amplitude = np.deg2rad([30, 25, 40, 35, 45, 40, 60])  # per-joint amplitude (radians)
    frequency = 0.2  # Hz
    dt = robot.get_timestep()

    def keep_running(elapsed: float) -> bool:
        if render:
            return robot.is_viewer_running()
        return duration is None or elapsed < duration

    print("\nRunning a sinusoidal joint trajectory with position control. Ctrl+C to exit.\n")
    start = time.time()
    last_log = 0.0
    try:
        while keep_running(time.time() - start):
            t = time.time() - start
            target = home + amplitude * np.sin(2 * np.pi * frequency * t)
            robot.set_joint_positions(target, actuated_joint_names=ARM)
            robot.step()

            now = time.time()
            if now - last_log > 1.0:
                ee_pos = robot.get_link_pose(robot.ee_ids[0])[0]
                err = np.linalg.norm(target - robot.get_actuated_joint_positions(ARM))
                print(
                    f"t={t:5.1f}s | ee_pos={np.round(ee_pos, 3)} "
                    f"| joint tracking err={err:.4f} rad"
                )
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
        "--duration", type=float, default=None, help="seconds to run (headless); default runs 5s"
    )
    args = parser.parse_args()
    main(render=not args.headless, duration=args.duration if args.duration else 5.0)
