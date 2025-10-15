import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

import omnigibson as og
import torch as th

class ROS2OGBridge(Node):
    def __init__(self, env, robot):
        super().__init__('ros2_og_bridge')
        self.env = env
        self.robot = robot
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10)
        self.subscription  # prevent unused var warning
        self.get_logger().info("Subscribed to /cmd_vel")

    def cmd_vel_callback(self, msg: Twist):
        # Extract v and omega from ROS2 message
        v = msg.linear.x
        omega = msg.angular.z

        # Package action for DifferentialDriveController
        action = th.tensor([v, omega], dtype=th.float32)

        # Step environment with this action
        self.env.step(action=action)


def main():
    # === Initialize OmniGibson ===
    scene_cfg = dict(type="InteractiveTraversableScene", scene_model="Rs_int")
    robot_cfg = dict(
        type="Fetch",
        obs_modalities=["rgb"],
        action_type="continuous",
        action_normalize=True,
    )
    cfg = dict(scene=scene_cfg, robots=[robot_cfg])
    env = og.Environment(configs=cfg)
    robot = env.robots[0]

    # Load DifferentialDriveController for base
    controller_choices = {"base": "DifferentialDriveController"}
    controller_config = {comp: {"name": name} for comp, name in controller_choices.items()}
    robot.reload_controllers(controller_config=controller_config)
    env.reset()
    robot.reset()

    # === Initialize ROS2 Node ===
    rclpy.init()
    node = ROS2OGBridge(env, robot)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            # Even if no new Twist, simulator still needs stepping
            env.step(action=th.tensor([0.0, 0.0]))
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        og.shutdown()


if __name__ == '__main__':
    main()
