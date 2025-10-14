#!/usr/bin/env python3

"""
ROS 2 node that publishes RGB-D, semantic data, and ground truth pose from OmniGibson.
Configured for Fetch robot with head camera.
Uses keyboard control for robot movement.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch as th
from scipy.spatial.transform import Rotation
import time

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController

# Performance optimizations
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.ENABLE_TRANSITION_RULES = False


class OmniGibsonFetchRGBDPublisher(Node):
    def __init__(self):
        super().__init__('omnigibson_fetch_rgbd_publisher')
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Target publishing rate
        self.declare_parameter('publish_rate', 15.0)
        self.target_rate = self.get_parameter('publish_rate').value
        
        # Create publishers for sensor data
        self.rgb_publisher = self.create_publisher(Image, '/camera/rgb/image_raw', 10)
        self.depth_publisher = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.semantic_publisher = self.create_publisher(Image, '/camera/semantic/image_raw', 10)
        self.instance_publisher = self.create_publisher(Image, '/camera/instance/image_raw', 10)
        self.rgb_info_publisher = self.create_publisher(CameraInfo, '/camera/rgb/camera_info', 10)
        self.depth_info_publisher = self.create_publisher(CameraInfo, '/camera/depth/camera_info', 10)
        
        # Create publishers for ground truth pose
        self.gt_odom_publisher = self.create_publisher(Odometry, '/ground_truth/odom', 10)
        self.gt_tf_broadcaster = TransformBroadcaster(self)
        
        # Camera parameters
        self.image_width = 640
        self.image_height = 480
        self.frame_id = "camera_link"
        
        # Camera info message (created once)
        self.camera_info_msg = None
        
        # Initialize OmniGibson environment
        self.init_omnigibson()
        
        # Create timer for publishing at target rate
        timer_period = 1.0 / self.target_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # Performance tracking
        self.frame_count = 0
        self.last_time = time.time()
        self.processing_times = []
        
        self.get_logger().info(f'OmniGibson Fetch Publisher initialized (target: {self.target_rate} Hz)')
        self.get_logger().info('Publishing sensor data from head camera and ground truth pose')

    def init_omnigibson(self):
        """Initialize OmniGibson environment with Fetch robot."""
        self.get_logger().info('Initializing OmniGibson environment with Fetch robot...')
        
        # Scene configuration
        scene_cfg = {
            "type": "InteractiveTraversableScene",
            "scene_model": "Merom_0_int"
        }
        
        # Fetch robot configuration with head camera
        robot_cfg = {
            "type": "Fetch",
            "obs_modalities": ["rgb", "depth", "depth_linear", "seg_semantic", "seg_instance"],
            "action_type": "continuous", 
            "action_normalize": True,
            # Exclude wrist/gripper cameras, only use head camera
            "exclude_sensor_names": ["eef_link", "gripper", "wrist"],
            "sensor_config": {
                "VisionSensor": {
                    "sensor_kwargs": {
                        "image_height": self.image_height,
                        "image_width": self.image_width
                    }
                }
            }
        }
        
        # Create environment
        cfg = {"scene": scene_cfg, "robots": [robot_cfg]}
        self.env = og.Environment(configs=cfg)
        self.robot = self.env.robots[0]
        
        # Set up robot controllers for Fetch
        # Fetch uses DifferentialDriveController for base (avoids cffi issues)
        controller_config = {
            "base": {"name": "DifferentialDriveController"},
            "trunk": {"name": "JointController"},
            "camera": {"name": "JointController"},
            "arm_right": {"name": "InverseKinematicsController"},
            "gripper_right": {"name": "MultiFingerGripperController"},
        }
        self.robot.reload_controllers(controller_config=controller_config)
        self.env.scene.update_initial_file()
        
        # Set camera pose for viewer
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([1.46949, -3.97358, 2.21529]),
            orientation=th.tensor([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
        )
        
        # Reset environment
        self.env.reset()
        self.robot.reset()
        
        # Lower the trunk (torso) to reduce robot height
        trunk_indices = self.robot.trunk_control_idx
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_pos[trunk_indices] = 0.0  # Fully lowered (min position)
        self.robot.set_joint_positions(current_joint_pos)
        
        # Set arm to a lower pose to avoid camera occlusion
        # Get the arm control indices
        arm_indices = self.robot.arm_control_idx[self.robot.default_arm]
        
        # Set arm to a tucked/lowered position
        # These joint values lower the arm to avoid blocking the camera
        lowered_arm_pose = th.tensor([
            1.32,    # shoulder_pan: rotate arm to side
            0.7,     # shoulder_lift: lift shoulder
            0.0,     # upperarm_roll: no roll
            -2.0,    # elbow_flex: bend elbow down
            0.0,     # forearm_roll: no roll
            -1.57,   # wrist_flex: flex wrist down
            0.0      # wrist_roll: no roll
        ])
        
        # Apply the joint positions
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_pos[arm_indices] = lowered_arm_pose
        self.robot.set_joint_positions(current_joint_pos)
        
        # Step simulation a few times to let the robot settle
        for _ in range(10):
            og.sim.step()
        
        # Create keyboard controller
        self.action_generator = KeyboardRobotController(robot=self.robot)
        self.action_generator.register_custom_keymapping(
            key=lazy.carb.input.KeyboardInput.R,
            description="Reset the robot",
            callback_fn=self.reset_environment,
        )
        
        self.action_generator.print_keyboard_teleop_info()
        self.get_logger().info('OmniGibson environment initialized with Fetch robot')
        self.get_logger().info('Base control: i/k (forward/back), j/l (turn left/right)')
        
        # Log available sensors
        self.get_logger().info('Available sensors:')
        for sensor_name in self.robot.sensors.keys():
            self.get_logger().info(f'  - {sensor_name}')

    def reset_environment(self):
        """Reset the environment callback."""
        self.get_logger().info('Resetting environment...')
        self.env.reset()
        self.robot.reset()

    def create_camera_info_msg(self, timestamp):
        """Create CameraInfo message with camera intrinsics."""
        if self.camera_info_msg is None:
            camera_info = CameraInfo()
            camera_info.width = self.image_width
            camera_info.height = self.image_height
            
            fx = fy = 600.0
            cx = self.image_width / 2.0
            cy = self.image_height / 2.0
            
            camera_info.k = [
                fx, 0.0, cx,
                0.0, fy, cy,
                0.0, 0.0, 1.0
            ]
            
            camera_info.p = [
                fx, 0.0, cx, 0.0,
                0.0, fy, cy, 0.0,
                0.0, 0.0, 1.0, 0.0
            ]
            
            camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            camera_info.distortion_model = "plumb_bob"
            camera_info.header.frame_id = self.frame_id
            
            self.camera_info_msg = camera_info
        
        # Update timestamp
        msg = CameraInfo()
        msg.header.stamp = timestamp
        msg.header.frame_id = self.camera_info_msg.header.frame_id
        msg.width = self.camera_info_msg.width
        msg.height = self.camera_info_msg.height
        msg.k = self.camera_info_msg.k
        msg.p = self.camera_info_msg.p
        msg.d = self.camera_info_msg.d
        msg.distortion_model = self.camera_info_msg.distortion_model
        
        return msg

    def get_ground_truth_pose(self):
        """Get ground truth pose from robot head camera."""
        # Get camera sensor (head camera for Fetch, exclude eef_link wrist camera)
        camera_sensor = None
        for sensor_name, sensor in self.robot.sensors.items():
            # Look specifically for head camera (not wrist/eef camera)
            if ("VisionSensor" in sensor_name or "Camera" in sensor_name) and "eef_link" not in sensor_name:
                camera_sensor = sensor
                break
        
        # If no head camera found, try any vision sensor
        if camera_sensor is None:
            for sensor_name, sensor in self.robot.sensors.items():
                if "VisionSensor" in sensor_name or "Camera" in sensor_name:
                    self.get_logger().warn(f'Using camera: {sensor_name}')
                    camera_sensor = sensor
                    break
        
        # Get pose from camera sensor if available, otherwise use robot base
        if camera_sensor and hasattr(camera_sensor, 'get_position_orientation'):
            pos, quat = camera_sensor.get_position_orientation()
        else:
            pos, quat = self.robot.get_position_orientation()

        # Ensure numpy arrays (float64 for numerical stability)
        if hasattr(pos, 'cpu'):
            pos = pos.cpu().numpy().astype(np.float64)
        else:
            pos = np.array(pos, dtype=np.float64)

        if hasattr(quat, 'cpu'):
            quat = quat.cpu().numpy().astype(np.float64)
        else:
            quat = np.array(quat, dtype=np.float64)

        # Transform matrix (identity - adjust if needed for coordinate system)
        T = np.array([
            [1.0,  0.0, 0.0],  
            [0.0, 1.0,  0.0],   
            [0.0,  0.0,  1.0],
        ], dtype=np.float64)

        # Transform position
        position = T @ pos

        # Convert input quaternion -> rotation matrix
        Rodom = Rotation.from_quat(quat).as_matrix()

        # Change basis for rotation: R_ros = T @ R_sim @ T^T
        Rworld = T @ Rodom @ T.T

        # Convert back to quaternion (xyzw)
        quaternion = Rotation.from_matrix(Rworld).as_quat()

        # Normalize quaternion
        qnorm = np.linalg.norm(quaternion)
        if qnorm > 0:
            quaternion = quaternion / qnorm

        return position, quaternion

    def publish_ground_truth(self, timestamp):
        """Publish ground truth odometry and TF."""
        # Get ground truth pose
        position, quaternion = self.get_ground_truth_pose()
        
        # Create Odometry message
        gt_odom = Odometry()
        gt_odom.header.stamp = timestamp
        gt_odom.header.frame_id = 'odom'
        gt_odom.child_frame_id = 'base_link_gt'
        
        # Set pose
        gt_odom.pose.pose.position.x = float(position[0])
        gt_odom.pose.pose.position.y = float(position[1])
        gt_odom.pose.pose.position.z = float(position[2])
        
        gt_odom.pose.pose.orientation.x = float(quaternion[0])
        gt_odom.pose.pose.orientation.y = float(quaternion[1])
        gt_odom.pose.pose.orientation.z = float(quaternion[2])
        gt_odom.pose.pose.orientation.w = float(quaternion[3])
        
        # Set covariance (very low for ground truth)
        covariance = [0.0] * 36
        covariance[0] = 0.001
        covariance[7] = 0.001
        covariance[14] = 0.001
        covariance[21] = 0.001
        covariance[28] = 0.001
        covariance[35] = 0.001
        
        gt_odom.pose.covariance = covariance
        gt_odom.twist.covariance = [0.0] * 36
        
        # Publish odometry
        self.gt_odom_publisher.publish(gt_odom)
        
        # Publish TF
        gt_tf = TransformStamped()
        gt_tf.header.stamp = timestamp
        gt_tf.header.frame_id = 'odom'
        gt_tf.child_frame_id = 'base_link_gt'
        
        gt_tf.transform.translation.x = float(position[0])
        gt_tf.transform.translation.y = float(position[1])
        gt_tf.transform.translation.z = float(position[2])
        
        gt_tf.transform.rotation.x = float(quaternion[0])
        gt_tf.transform.rotation.y = float(quaternion[1])
        gt_tf.transform.rotation.z = float(quaternion[2])
        gt_tf.transform.rotation.w = float(quaternion[3])
        
        self.gt_tf_broadcaster.sendTransform(gt_tf)

    def timer_callback(self):
        """Timer callback to capture and publish all data."""
        start_time = time.time()
        
        try:
            # Get keyboard action
            action = self.action_generator.get_teleop_action()
            
            # Step simulation
            self.env.step(action=action)
            
            # Get observations
            data, info = self.env.get_obs()
            robot_key = next(iter(data.keys()))
            sensor_dict = data[robot_key]
            
            # Find the head camera data (VisionSensor)
            # Prioritize head camera over wrist camera
            cam_key = None
            for key in sensor_dict.keys():
                # Exclude eef_link (wrist camera), look for head camera
                if ("VisionSensor" in key or "Camera" in key) and "eef_link" not in key:
                    cam_key = key
                    break
            
            # If no head camera found, use any camera but warn
            if cam_key is None:
                for key in sensor_dict.keys():
                    if "VisionSensor" in key or "Camera" in key:
                        if self.frame_count == 0:  # Only warn once
                            self.get_logger().warn(f'No head camera found, using: {key}')
                        cam_key = key
                        break
            
            if cam_key is None:
                self.get_logger().warn('No camera sensor found in observations')
                return
                
            cam_dict = sensor_dict[cam_key]
            
            # Create timestamp
            timestamp = self.get_clock().now().to_msg()
            
            # Process RGB
            rgb = np.array(cam_dict["rgb"])[..., :3].astype(np.uint8)
            if rgb.shape[:2] != (self.image_height, self.image_width):
                rgb_resized = cv2.resize(rgb, (self.image_width, self.image_height), 
                                       interpolation=cv2.INTER_LINEAR)
            else:
                rgb_resized = rgb
            
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_resized, encoding="rgb8")
            rgb_msg.header.stamp = timestamp
            rgb_msg.header.frame_id = self.frame_id
            
            # Process depth
            if "depth_linear" in cam_dict:
                depth = np.array(cam_dict["depth_linear"])
            else:
                depth = np.array(cam_dict["depth"])
            
            if depth.shape != (self.image_height, self.image_width):
                depth_resized = cv2.resize(depth, (self.image_width, self.image_height), 
                                         interpolation=cv2.INTER_NEAREST)
            else:
                depth_resized = depth
                
            depth_mm = (depth_resized * 1000.0).astype(np.uint16)
            
            depth_msg = self.bridge.cv2_to_imgmsg(depth_mm, encoding="16UC1")
            depth_msg.header.stamp = timestamp
            depth_msg.header.frame_id = self.frame_id
            
            # Process semantic
            semantic = cam_dict["seg_semantic"]
            if hasattr(semantic, 'cpu'):
                semantic = semantic.cpu().numpy().astype(np.int32)
            else:
                semantic = semantic.astype(np.int32)
                
            if semantic.shape != (self.image_height, self.image_width):
                semantic_resized = cv2.resize(semantic, (self.image_width, self.image_height), 
                                            interpolation=cv2.INTER_NEAREST)
            else:
                semantic_resized = semantic
            
            semantic_msg = self.bridge.cv2_to_imgmsg(semantic_resized, encoding="32SC1")
            semantic_msg.header.stamp = timestamp
            semantic_msg.header.frame_id = self.frame_id
            
            # Process instance
            instance = cam_dict["seg_instance"]
            if hasattr(instance, 'cpu'):
                instance = instance.cpu().numpy().astype(np.uint8)
            else:
                instance = instance.astype(np.uint8)
                
            if instance.shape != (self.image_height, self.image_width):
                instance_resized = cv2.resize(instance, (self.image_width, self.image_height), 
                                            interpolation=cv2.INTER_NEAREST)
            else:
                instance_resized = instance
            
            instance_msg = self.bridge.cv2_to_imgmsg(instance_resized, encoding="mono8")
            instance_msg.header.stamp = timestamp
            instance_msg.header.frame_id = self.frame_id
            
            # Create camera info
            rgb_info = self.create_camera_info_msg(timestamp)
            depth_info = self.create_camera_info_msg(timestamp)
            
            # Publish all sensor data
            self.rgb_publisher.publish(rgb_msg)
            self.depth_publisher.publish(depth_msg)
            self.semantic_publisher.publish(semantic_msg)
            self.instance_publisher.publish(instance_msg)
            self.rgb_info_publisher.publish(rgb_info)
            self.depth_info_publisher.publish(depth_info)
            
            # Publish ground truth pose
            self.publish_ground_truth(timestamp)
            
            self.frame_count += 1
            
            # Track processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            # Log progress with performance metrics
            if self.frame_count % 150 == 0:
                current_time = time.time()
                elapsed = current_time - self.last_time
                actual_rate = 150 / elapsed if elapsed > 0 else 0
                avg_processing = np.mean(self.processing_times) * 1000
                max_processing = np.max(self.processing_times) * 1000
                
                self.get_logger().info(
                    f'Frame {self.frame_count} | '
                    f'Rate: {actual_rate:.1f} Hz (target: {self.target_rate} Hz) | '
                    f'Processing: avg={avg_processing:.1f}ms, max={max_processing:.1f}ms'
                )
                self.last_time = current_time
                
        except Exception as e:
            self.get_logger().error(f'Error in timer callback: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def destroy_node(self):
        """Clean shutdown."""
        self.get_logger().info('Shutting down...')
        try:
            og.shutdown()
        except:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        publisher = OmniGibsonFetchRGBDPublisher()
        
        print("\n" + "="*70)
        print("OmniGibson ROS 2 Publisher - Fetch Robot with Head Camera")
        print("="*70)
        print("Robot: Fetch (mobile manipulator)")
        print("Camera: Head-mounted vision sensor")
        print("\nPublishing:")
        print("  - RGB: /camera/rgb/image_raw")
        print("  - Depth: /camera/depth/image_raw")
        print("  - Semantic: /camera/semantic/image_raw")
        print("  - Instance: /camera/instance/image_raw")
        print("  - Ground truth: /ground_truth/odom")
        print("  - TF: odom -> base_link_gt")
        print("\nBase Movement Controls (Differential Drive):")
        print("  - i: Move forward")
        print("  - k: Move backward")
        print("  - j: Turn left")
        print("  - l: Turn right")
        print("\nOther Controls:")
        print("  - Arrow keys: Control arm (IK mode)")
        print("  - t: Toggle gripper")
        print("  - Press 'R' to reset")
        print("  - Press 'ESC' to quit")
        print("="*70 + "\n")
        
        rclpy.spin(publisher)
        
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            publisher.destroy_node()
        except:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()