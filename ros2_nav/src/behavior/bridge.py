import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
import omnigibson as og
import torch as th
import cv2
import os
import yaml
from omnigibson.macros import gm


# Performance optimizations
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.ENABLE_TRANSITION_RULES = False


def generate_map_yaml(w, h):
    data = {
        "image": "trav_map.png",
        "resolution": 0.01,
        "origin": [-w/200.0, -h/200.0, 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }
    with open("./maps/trav_map.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"YAML file saved as src/behavior/maps/trav_map.yaml")

class ROS2OGBridge(Node):
    def __init__(self, env, robot):
        super().__init__('ros2_og_bridge')
        self.v = 0.0
        self.omega = 0.0
        self.env = env
        self.robot = robot

        # Subscribe to cmd_vel
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        self.subscription  # prevent unused var warning
        self.get_logger().info("Subscribed to /cmd_vel")

        # Publisher for robot pose
        self.pose_publisher = self.create_publisher(PoseStamped, '/robot_pose', 10)
        self.get_logger().info("Publishing to /robot_pose")

    def cmd_vel_callback(self, msg: Twist):
        # Update linear and angular velocity from ROS2 messages
        self.v = msg.linear.x
        self.omega = msg.angular.z

    def publish_pose(self):
        # Get robot position and orientation from OmniGibson
        pos, orn = self.robot.get_position_orientation()
        
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = float(pos[0])
        pose_msg.pose.position.y = float(pos[1])
        pose_msg.pose.position.z = float(pos[2])
        pose_msg.pose.orientation.x = float(orn[0])
        pose_msg.pose.orientation.y = float(orn[1])
        pose_msg.pose.orientation.z = float(orn[2])
        pose_msg.pose.orientation.w = float(orn[3])

        self.pose_publisher.publish(pose_msg)


def main():
    # Specify scene and robot configuration
    scene_model = "Rs_int"
    trav_map_erosion = 2

    scene_cfg = dict(type="InteractiveTraversableScene", scene_model=scene_model)
    robot_cfg = dict(
        type="Fetch",
        obs_modalities=["rgb"],
        action_type="continuous",
        action_normalize=True,
    )
    cfg = dict(scene=scene_cfg, robots=[robot_cfg])
    
    # Initialize OmniGibson environment
    env = og.Environment(configs=cfg)
    robot = env.robots[0]

    # Generate traversability map /data/og_dataset/scenes/{scene_model}/layout
    trav_map = cv2.imread(f"~/omnigibson/data/og_dataset/scenes/{scene_model}/layout/floor_trav_0.png")
    trav_map = cv2.erode(trav_map, th.ones((trav_map_erosion, trav_map_erosion)).cpu().numpy())

    cv2.imwrite("./maps/trav_map.png", trav_map)
    height, width = trav_map.shape[:2]
    generate_map_yaml(width, height)

    # Load DifferentialDriveController
    controller_choices = {"base": "DifferentialDriveController"}
    controller_config = {comp: {"name": name} for comp, name in controller_choices.items()}
    robot.reload_controllers(controller_config=controller_config)
    env.reset()
    robot.reset()

    # Initialize ROS2
    rclpy.init()
    node = ROS2OGBridge(env, robot)

    try:
        while rclpy.ok():
            # Process incoming cmd_vel messages
            rclpy.spin_once(node, timeout_sec=0.01)

            # Read the latest velocities
            v = node.v  
            omega = node.omega

            # Fill full action tensor for DifferentialDriveController
            full_action = th.zeros(robot.action_dim, dtype=th.float32)
            full_action[0] = v
            full_action[1] = omega

            # Step the environment
            env.step(action={robot.name: full_action})

            # Publish current pose
            node.publish_pose()

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        og.shutdown()


if __name__ == "__main__":
    main()
