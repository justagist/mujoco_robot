"""Task-space (Cartesian) control demo for mujoco_robot.

Loads a Franka Panda in torque mode and uses a Cartesian impedance controller to make the
end-effector trace a circle in the world Y-Z plane, while the arm holds itself up (gravity
compensation) and regulates its posture in the nullspace. A passive viewer shows the motion
(unless ``--headless``).

Run:
    python demo_task_space_control.py            # with the viewer
    python demo_task_space_control.py --headless # no viewer (prints tracking only)
"""

import argparse
import time

import numpy as np

from mujoco_robot import MujocoRobot
from mujoco_robot.utils.robot_loader_utils import get_mjcf_from_awesome_robot_descriptions

from impedance_controllers import CartesianImpedanceController

# A valid within-limits "ready" configuration (7 arm joints + 2 fingers).
NEUTRAL = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.02, 0.02]

# Cartesian impedance gains: [x, y, z, roll, pitch, yaw]. Gravity compensation is handled by the
# controller, so moderate (compliant) stiffness is enough for smooth, stable tracking.
Kp = np.array([600.0, 600.0, 600.0, 30.0, 30.0, 30.0])
Kd = 2.0 * np.sqrt(Kp)  # critically damped
NULLSPACE_Kp = np.array([10.0] * 9)  # gentle posture regulation

TRAJ_RADIUS = 0.12  # metres
TRAJ_PERIOD = 6.0  # seconds per revolution


def main(render: bool = True, duration: float = None):
    robot = MujocoRobot(
        mjcf_path=get_mjcf_from_awesome_robot_descriptions("panda_mj_description"),
        default_joint_positions=NEUTRAL,
        enable_torque_mode=True,  # torque control for impedance
        ee_names=["hand"],
        run_async=False,
        place_on_ground=False,
        load_ground_plane=True,
        render=render,
        verbose=True,
    )

    controller = CartesianImpedanceController(
        robot=robot,
        kp=Kp,
        kd=Kd,
        null_kp=NULLSPACE_Kp,
        nullspace_pos_target=NEUTRAL,
    )

    robot.step()  # settle one step, then read the starting end-effector pose
    center_pos, fixed_ori = robot.get_link_pose(link_id=robot.ee_ids[0])
    center_pos = center_pos.copy()
    fixed_ori = fixed_ori.copy()

    omega = 2.0 * np.pi / TRAJ_PERIOD
    dt = robot.get_timestep()

    def keep_running(elapsed: float) -> bool:
        if render:
            return robot.is_viewer_running()
        return duration is None or elapsed < duration

    print("\nTracing a circle in the Y-Z plane with Cartesian impedance control. Ctrl+C to exit.\n")
    start = time.time()
    last_log = 0.0
    try:
        while keep_running(time.time() - start):
            t = time.time() - start
            goal_pos = center_pos + np.array(
                [0.0, TRAJ_RADIUS * np.sin(omega * t), TRAJ_RADIUS * (1.0 - np.cos(omega * t))]
            )
            controller.set_target(goal_pos=goal_pos, goal_ori=fixed_ori)
            joint_cmds, error = controller.compute_cmd()
            robot.set_actuated_joint_commands(tau=joint_cmds)
            robot.step()

            now = time.time()
            if now - last_log > 1.0:
                print(f"t={t:5.1f}s | pos error: {error[0]:.4f} m | ori error: {error[1]:.4f} rad")
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
