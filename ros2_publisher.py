#!/usr/bin/env python3

"""
ROS 2 node that publishes RGB-D, semantic data, and ground truth pose from OmniGibson.
Optimized for higher publishing rate (~15 Hz).
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
gm.ENABLE_TRANSITION_RULES = False  # Disable transition rules for speed


class OmniGibsonKeyboardRGBDPublisher(Node):
    def __init__(self):
        super().__init__('omnigibson_keyboard_rgbd_publisher')
        
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
        
        self.get_logger().info(f'OmniGibson Publisher initialized (target: {self.target_rate} Hz)')
        self.get_logger().info('Publishing sensor data and ground truth pose')

    def init_omnigibson(self):
        """Initialize OmniGibson environment with RGB-D capable robot."""
        self.get_logger().info('Initializing OmniGibson environment...')
        
        # Scene configuration
        scene_cfg = {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int"
        }
        
        # Robot configuration - optimized sensor settings
        robot_cfg = {
            "type": "Turtlebot",
            "obs_modalities": ["rgb", "depth", "depth_linear", "seg_semantic", "seg_instance"],
            "action_type": "continuous", 
            "action_normalize": True,
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
        
        # Set up robot controllers
        controller_config = {
            "base": {"name": "DifferentialDriveController"},
            "arm_0": {"name": "InverseKinematicsController"},
            "gripper_0": {"name": "MultiFingerGripperController"},
            "camera": {"name": "JointController"},
        }
        self.robot.reload_controllers(controller_config=controller_config)
        self.env.scene.update_initial_file()
        
        # Set camera pose
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([1.46949, -3.97358, 2.21529]),
            orientation=th.tensor([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
        )
        
        # Reset environment
        self.env.reset()
        self.robot.reset()
        
        # Create keyboard controller
        self.action_generator = KeyboardRobotController(robot=self.robot)
        self.action_generator.register_custom_keymapping(
            key=lazy.carb.input.KeyboardInput.R,
            description="Reset the robot",
            callback_fn=self.reset_environment,
        )
        
        self.action_generator.print_keyboard_teleop_info()
        self.get_logger().info('OmniGibson environment initialized')

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
        """Get ground truth pose from robot/camera."""
        # Get pose
        camera_sensor = None
        for sensor in self.robot.sensors.values():
            if hasattr(sensor, 'get_position_orientation'):
                camera_sensor = sensor
                break
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

        # Transform matrix
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
            
            # Get camera data
            cam_key = next(k for k in sensor_dict.keys() if "Camera" in k)
            cam_dict = sensor_dict[cam_key]
            
            # Create timestamp
            timestamp = self.get_clock().now().to_msg()
            
            # Process RGB - optimize with direct conversion
            rgb = np.array(cam_dict["rgb"])[..., :3].astype(np.uint8)
            if rgb.shape[:2] != (self.image_height, self.image_width):
                rgb_resized = cv2.resize(rgb, (self.image_width, self.image_height), 
                                       interpolation=cv2.INTER_LINEAR)  # Changed to INTER_LINEAR for speed
            else:
                rgb_resized = rgb
            
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_resized, encoding="rgb8")
            rgb_msg.header.stamp = timestamp
            rgb_msg.header.frame_id = self.frame_id
            
            # Process depth - optimize conversion
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
            
            # Process semantic - optimize with direct numpy conversion
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
            
            # Process instance - optimize with direct numpy conversion
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
            
            # Create camera info (reuse cached message)
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
            if self.frame_count % 150 == 0:  # Every 10 seconds at 15 Hz
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
        publisher = OmniGibsonKeyboardRGBDPublisher()
        
        print("\n" + "="*70)
        print("OmniGibson ROS 2 Publisher (Optimized for 15 Hz)")
        print("="*70)
        print("Publishing:")
        print("  - Sensor data: /camera/*/image_raw")
        print("  - Ground truth: /ground_truth/odom")
        print("  - TF: odom -> base_link_gt")
        print("\nOptimizations:")
        print("  - Efficient image processing")
        print("  - Reduced unnecessary conversions")
        print("  - Cached camera info messages")
        print("\nControls:")
        print("  - Use keyboard to control the robot")
        print("  - Press 'R' to reset")
        print("  - Press 'ESC' to quit")
        print("="*70 + "\n")
        
        rclpy.spin(publisher)
        
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            publisher.destroy_node()
        except:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()