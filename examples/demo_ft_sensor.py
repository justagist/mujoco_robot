"""Force-torque sensor demo for mujoco_robot.

A Franka Panda (with a force-torque sensor on its ``hand``) repeatedly presses its hand down
onto the ground and releases, using a Cartesian impedance controller. The measured wrench is
printed live, and in the viewer a red arrow at the hand shows the measured contact force: it is
short in free space (just the static weight of the hand/fingers) and grows sharply on contact.

Run:
    python demo_ft_sensor.py            # with the viewer + force arrow
    python demo_ft_sensor.py --headless # prints the wrench only
"""

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation

from mujoco_robot import MujocoRobot
from mujoco_robot.utils.robot_loader_utils import get_mjcf_from_awesome_robot_descriptions

from impedance_controllers import CartesianImpedanceController

NEUTRAL = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.02, 0.02]
Kp = np.array([800.0, 800.0, 800.0, 40.0, 40.0, 40.0])
Kd = 2.0 * np.sqrt(Kp)

PRESS_DEPTH = 0.55  # metres below the start height (well past the floor -> firm contact)
PRESS_PERIOD = 4.0  # seconds per press-and-release cycle
ARROW_SCALE = 0.004  # metres of arrow per newton


def _draw_force_arrow(robot, site_id, force_world):
    """Draw a red arrow at the hand showing the measured contact force (viewer only)."""
    import mujoco

    scene = robot.viewer.user_scn
    scene.ngeom = 0
    magnitude = np.linalg.norm(force_world)
    if magnitude < 1e-6 or scene.maxgeom < 1:
        return
    start = np.array(robot.data.site_xpos[site_id])
    end = start + force_world * ARROW_SCALE
    geom = scene.geoms[0]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        np.array([1.0, 0.1, 0.1, 1.0], dtype=np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.012, start, end)
    scene.ngeom = 1


def main(render: bool = True, duration: float = None):
    robot = MujocoRobot(
        mjcf_path=get_mjcf_from_awesome_robot_descriptions("panda_mj_description"),
        default_joint_positions=NEUTRAL,
        enable_torque_mode=True,
        ee_names=["hand"],
        ft_sensor_links=["hand"],  # inject the force-torque sensor on the hand
        run_async=False,
        place_on_ground=False,
        load_ground_plane=True,
        render=render,
        verbose=True,
    )
    controller = CartesianImpedanceController(
        robot=robot, kp=Kp, kd=Kd, null_kp=np.array([5.0] * 9), nullspace_pos_target=NEUTRAL
    )

    robot.step()
    start_pos, fixed_ori = robot.get_link_pose(link_id=robot.ee_ids[0])
    start_pos, fixed_ori = start_pos.copy(), fixed_ori.copy()
    site_id = robot.get_site_id("hand_ft_site")
    omega = 2.0 * np.pi / PRESS_PERIOD
    dt = robot.get_timestep()

    def keep_running(elapsed: float) -> bool:
        if render:
            return robot.is_viewer_running()
        return duration is None or elapsed < duration

    print("\nPressing the hand onto the ground; watch the force-torque reading. Ctrl+C to exit.\n")
    start = time.time()
    last_log = 0.0
    try:
        while keep_running(time.time() - start):
            t = time.time() - start
            # 0 -> PRESS_DEPTH -> 0 : descend into the floor, then release
            depth = PRESS_DEPTH * 0.5 * (1.0 - np.cos(omega * t))
            goal = start_pos - np.array([0.0, 0.0, depth])
            controller.set_target(goal_pos=goal, goal_ori=fixed_ori)
            tau, _ = controller.compute_cmd()
            robot.set_actuated_joint_commands(tau=tau)

            wrench = robot.get_link_ft_measurement("hand")
            if render:
                force_world = Rotation.from_quat(robot.get_site_pose(site_id)[1]).as_matrix().dot(
                    wrench[:3]
                )
                _draw_force_arrow(robot, site_id, force_world)

            robot.step()

            now = time.time()
            if now - last_log > 0.5:
                # the contact is on the finger geoms; the hand FT sensor senses it up the chain
                touching = int(
                    robot.get_contact_states_of_links(
                        [robot.get_link_id(n) for n in ("hand", "left_finger", "right_finger")]
                    ).any()
                )
                print(
                    f"t={t:4.1f}s | contact={touching} | force={np.round(wrench[:3], 2)} N "
                    f"| torque={np.round(wrench[3:], 2)} N·m"
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
        "--duration", type=float, default=None, help="seconds to run (headless); default runs 8s"
    )
    args = parser.parse_args()
    main(render=not args.headless, duration=args.duration if args.duration else 8.0)
