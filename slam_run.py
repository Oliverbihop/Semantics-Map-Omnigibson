#!/usr/bin/env python3

"""
Enhanced ROS 2 SLAM node with drift reduction features.
- Keyframe-based mapping for stability
- Adaptive covariance estimation
- Outlier rejection for robust odometry
Uses ROS standard world coordinates: X-forward, Y-left, Z-up
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge
import numpy as np
import pygicp
import cv2
from scipy.spatial.transform import Rotation
import message_filters
from collections import deque
import time


class RGBDSLAMNode(Node):
    def __init__(self):
        super().__init__('rgbd_slam_node')
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # SLAM parameters
        self.declare_parameter('downsample_resolution', 0.05)
        self.declare_parameter('max_correspondence_distance', 0.5)
        self.declare_parameter('num_threads', 4)
        self.declare_parameter('skip_frames', 1)  # Changed back to 1 for debugging
        
        # Drift reduction parameters
        self.declare_parameter('use_keyframes', True)
        self.declare_parameter('keyframe_distance', 0.1)  # meters
        self.declare_parameter('keyframe_rotation', 2.0)  # degrees
        self.declare_parameter('max_keyframes', 10000)
        
        self.downsample_resolution = self.get_parameter('downsample_resolution').value
        self.max_correspondence_distance = self.get_parameter('max_correspondence_distance').value
        self.num_threads = self.get_parameter('num_threads').value
        self.skip_frames = self.get_parameter('skip_frames').value
        
        self.use_keyframes = self.get_parameter('use_keyframes').value
        self.keyframe_distance = self.get_parameter('keyframe_distance').value
        self.keyframe_rotation = self.get_parameter('keyframe_rotation').value
        self.max_keyframes = self.get_parameter('max_keyframes').value
        
        # Frame skip counter
        self.frame_skip_counter = 0
        
        # Camera intrinsics
        self.fx = 600.0
        self.fy = 600.0
        self.cx = 320.0
        self.cy = 240.0
        self.camera_info_received = False
        
        # SLAM state (in WORLD coordinates)
        self.previous_cloud = None
        self.current_pose = np.eye(4)
        self.frame_count = 0
        self.processed_count = 0
        self.received_count = 0
        self.skipped_count = 0
        
        # Keyframe management
        self.keyframes = deque(maxlen=self.max_keyframes)
        self.last_keyframe_pose = np.eye(4)
        
        # Motion model for outlier detection
        self.previous_pose = np.eye(4)
        self.velocity_history = deque(maxlen=5)
        
        # Initialize GICP
        self.gicp = pygicp.FastGICP()
        self.gicp.set_num_threads(self.num_threads)
        self.gicp.set_max_correspondence_distance(self.max_correspondence_distance)
        
        # Subscribers
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/rgb/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 
            queue_size=50,
            slop=0.2
        )
        self.ts.registerCallback(self.rgbd_callback)
        
        self.processing = False
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/rgb/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Publishers
        self.odom_publisher = self.create_publisher(Odometry, '/odometry/gicp', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Performance tracking
        self.last_log_time = time.time()
        
        self.get_logger().info('='*70)
        self.get_logger().info('Enhanced RGB-D SLAM Node initialized')
        self.get_logger().info(f'Keyframes enabled: {self.use_keyframes}')
        self.get_logger().info(f'Skip frames: {self.skip_frames}')
        self.get_logger().info(f'Downsample resolution: {self.downsample_resolution}m')
        self.get_logger().info('Waiting for RGB-D data...')
        self.get_logger().info('='*70)

    def manual_voxel_downsample(self, points, voxel_size):
        """Manual voxel grid downsampling."""
        if len(points) == 0:
            return points
        
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        downsampled = points[unique_indices]
        downsampled = np.ascontiguousarray(downsampled, dtype=np.float64)
        
        return downsampled

    def clean_point_cloud(self, points):
        """Clean point cloud to remove invalid values."""
        if points is None or len(points) == 0:
            return np.array([]).reshape(0, 3)
        
        finite_mask = np.all(np.isfinite(points), axis=1)
        points = points[finite_mask]
        
        if len(points) == 0:
            return np.array([]).reshape(0, 3)
        
        range_mask = np.all(np.abs(points) < 50.0, axis=1)
        points = points[range_mask]
        
        if len(points) == 0:
            return np.array([]).reshape(0, 3)
        
        distance = np.linalg.norm(points, axis=1)
        distance_mask = distance > 0.1
        points = points[distance_mask]
        
        points = np.ascontiguousarray(points, dtype=np.float64)
        
        return points

    def camera_info_callback(self, msg):
        """Update camera intrinsics from CameraInfo message."""
        if not self.camera_info_received:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            
            self.camera_info_received = True
            self.get_logger().info(f'Camera intrinsics received: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}')

    def depth_to_point_cloud(self, depth_image, rgb_image=None):
        """Convert depth to 3D point cloud in WORLD coordinates."""
        h, w = depth_image.shape
        
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        valid_mask = (depth_image > 0.1) & (depth_image < 10.0) & np.isfinite(depth_image)
        
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = depth_image[valid_mask]
        
        # Back-project to camera frame
        x_cam = (u_valid - self.cx) * z_valid / self.fx
        y_cam = (v_valid - self.cy) * z_valid / self.fy
        z_cam = z_valid
        
        points_camera = np.stack([x_cam, y_cam, z_cam], axis=-1)
        points_camera = np.ascontiguousarray(points_camera, dtype=np.float64)
        
        valid_points_mask = np.all(np.isfinite(points_camera), axis=1)
        points_camera = points_camera[valid_points_mask]
        
        # Transform to world frame
        points_world = np.zeros_like(points_camera)
        points_world[:, 0] = points_camera[:, 2]   # X = forward
        points_world[:, 1] = -points_camera[:, 0]  # Y = left
        points_world[:, 2] = -points_camera[:, 1]  # Z = up
        
        colors = None
        if rgb_image is not None:
            colors = rgb_image[valid_mask]
            colors = colors[valid_points_mask]
        
        return points_world, colors

    def check_keyframe(self, current_pose):
        """Check if current pose should be a keyframe."""
        if len(self.keyframes) == 0:
            return True
        
        # Compute relative transformation
        delta = np.linalg.inv(self.last_keyframe_pose) @ current_pose
        
        # Translation distance
        translation = np.linalg.norm(delta[:3, 3])
        
        # Rotation angle
        R = delta[:3, :3]
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
        angle_deg = np.degrees(angle)
        
        # Check thresholds
        if translation > self.keyframe_distance or angle_deg > self.keyframe_rotation:
            return True
        
        return False

    def is_motion_outlier(self, transformation):
        """Check if motion is an outlier based on velocity history."""
        if len(self.velocity_history) < 3:
            return False
        
        # Current velocity
        translation = np.linalg.norm(transformation[:3, 3])
        
        # Historical statistics
        velocities = np.array(list(self.velocity_history))
        mean_vel = np.mean(velocities)
        std_vel = np.std(velocities)
        
        # Outlier detection (3-sigma rule)
        threshold = mean_vel + 3 * std_vel
        
        if translation > max(threshold, 1.0):
            self.get_logger().warn(f'Motion outlier detected: {translation:.2f}m vs avg {mean_vel:.2f}m')
            return True
        
        return False

    def compute_adaptive_covariance(self, transformation, fitness_score=None):
        """Compute adaptive covariance based on motion and fitness."""
        base_cov = 0.01
        
        # Increase covariance with larger motions
        translation = np.linalg.norm(transformation[:3, 3])
        translation_factor = 1.0 + translation
        
        # Increase covariance with poor fitness
        if fitness_score is not None:
            fitness_factor = 1.0 / max(fitness_score, 0.1)
        else:
            fitness_factor = 1.0
        
        # Combined adaptive covariance
        adaptive_cov = base_cov * translation_factor * fitness_factor
        
        covariance = np.zeros((6, 6))
        covariance[0, 0] = adaptive_cov
        covariance[1, 1] = adaptive_cov
        covariance[2, 2] = adaptive_cov
        covariance[3, 3] = adaptive_cov * 0.5
        covariance[4, 4] = adaptive_cov * 0.5
        covariance[5, 5] = adaptive_cov * 0.5
        
        return covariance.flatten().tolist()

    def rgbd_callback(self, rgb_msg, depth_msg):
        """Synchronized callback with enhanced SLAM."""
        self.received_count += 1
        
        # Debug: Log first message
        if self.received_count == 1:
            self.get_logger().info('✓ First synchronized RGB-D message received!')
        
        # Skip if already processing
        if self.processing:
            self.skipped_count += 1
            return
        
        # Frame skipping logic
        self.frame_skip_counter += 1
        if self.frame_skip_counter <= self.skip_frames:
            return
        self.frame_skip_counter = 0
        
        self.processing = True
        
        try:
            # Convert messages
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            
            # Convert depth to meters
            if depth_image.dtype == np.uint16:
                depth_image = depth_image.astype(np.float32) / 1000.0
            
            # Clean depth
            if not np.all(np.isfinite(depth_image)):
                depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Generate point cloud in WORLD coordinates
            points, colors = self.depth_to_point_cloud(depth_image, rgb_image)
            
            # Clean point cloud
            points = self.clean_point_cloud(points)
            
            if len(points) < 100:
                self.get_logger().warn(f'Not enough points: {len(points)} < 100')
                return
            
            # Downsample
            if self.downsample_resolution > 0:
                points = self.manual_voxel_downsample(points, self.downsample_resolution)
                points = self.clean_point_cloud(points)
                
                if len(points) < 50:
                    self.get_logger().warn(f'Not enough points after downsample: {len(points)} < 50')
                    return
            
            # Final validation
            if not np.all(np.isfinite(points)) or len(points) < 50:
                self.get_logger().warn('Points failed final validation')
                return
            
            # GICP alignment
            if self.previous_cloud is not None:
                try:
                    if not np.all(np.isfinite(self.previous_cloud)) or len(self.previous_cloud) < 50:
                        self.get_logger().warn('Invalid previous cloud, reinitializing')
                        self.previous_cloud = points
                        return
                    
                    # Set target and source
                    self.gicp.set_input_target(self.previous_cloud)
                    self.gicp.set_input_source(points)
                    
                    # Align
                    transformation = self.gicp.align()
                    
                    # Get fitness score
                    try:
                        fitness_score = self.gicp.get_fitness_score()
                    except:
                        fitness_score = 1.0
                    
                    # Validate transformation
                    if not np.all(np.isfinite(transformation)):
                        self.get_logger().error('Invalid transformation (contains NaN/Inf)')
                        return
                    
                    translation_norm = np.linalg.norm(transformation[:3, 3])
                    
                    # Outlier rejection
                    if self.is_motion_outlier(transformation):
                        return
                    
                    # Update velocity history
                    self.velocity_history.append(translation_norm)
                    
                    # Check for excessive motion
                    if translation_norm > 2.0:
                        self.get_logger().warn(f'Large translation rejected: {translation_norm:.2f}m')
                        return
                    
                    # Update pose
                    self.previous_pose = self.current_pose.copy()
                    self.current_pose = self.current_pose @ transformation
                    
                    # Keyframe management
                    if self.use_keyframes and self.check_keyframe(self.current_pose):
                        self.keyframes.append((self.current_pose.copy(), points.copy()))
                        self.last_keyframe_pose = self.current_pose.copy()
                    
                    # Publish with adaptive covariance
                    covariance = self.compute_adaptive_covariance(transformation, fitness_score)
                    self.publish_odometry(rgb_msg.header.stamp, covariance)
                    
                    # Log progress
                    self.processed_count += 1
                    current_time = time.time()
                    if current_time - self.last_log_time >= 3.0:
                        elapsed = current_time - self.last_log_time
                        rate = self.processed_count / elapsed if elapsed > 0 else 0
                        
                        t = self.current_pose[:3, 3]
                        self.get_logger().info(
                            f'Received: {self.received_count} | Processed: {self.processed_count} | '
                            f'Rate: {rate:.1f} Hz | Pos=[{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}] | '
                            f'Fit: {fitness_score:.3f} | KF: {len(self.keyframes)}'
                        )
                        self.last_log_time = current_time
                        self.processed_count = 0
                        self.received_count = 0
                        
                except Exception as e:
                    self.get_logger().error(f'GICP alignment failed: {str(e)}')
                    import traceback
                    self.get_logger().error(traceback.format_exc())
                    return
            else:
                # First frame - initialize
                self.get_logger().info(f'✓ First frame initialized with {len(points)} points')
                
                # Initialize keyframes
                if self.use_keyframes:
                    self.keyframes.append((self.current_pose.copy(), points.copy()))
                    self.last_keyframe_pose = self.current_pose.copy()
                
                # Publish initial pose at origin
                covariance = [0.01] * 36
                for i in range(6):
                    covariance[i * 6 + i] = 0.01
                
                self.publish_odometry(rgb_msg.header.stamp, covariance)
                self.get_logger().info('✓ Initial odometry published at origin')
            
            # Store previous cloud
            if np.all(np.isfinite(points)) and len(points) >= 50:
                self.previous_cloud = points.copy()
            
        except Exception as e:
            self.get_logger().error(f'Callback error: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())
        finally:
            self.processing = False

    def publish_odometry(self, timestamp, covariance):
        """Publish odometry in WORLD coordinates."""
        # Extract translation and rotation
        translation = self.current_pose[:3, 3]
        rotation_matrix = self.current_pose[:3, :3]
        
        # Convert to quaternion
        rotation = Rotation.from_matrix(rotation_matrix)
        quat = rotation.as_quat()  # [x, y, z, w]
        
        # Create Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = timestamp
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        
        # Position
        odom_msg.pose.pose.position.x = translation[0]
        odom_msg.pose.pose.position.y = translation[1]
        odom_msg.pose.pose.position.z = translation[2]
        
        # Orientation
        odom_msg.pose.pose.orientation.x = quat[0]
        odom_msg.pose.pose.orientation.y = quat[1]
        odom_msg.pose.pose.orientation.z = quat[2]
        odom_msg.pose.pose.orientation.w = quat[3]
        
        # Covariance
        odom_msg.pose.covariance = covariance
        odom_msg.twist.covariance = [1e6] * 36
        
        # Publish
        self.odom_publisher.publish(odom_msg)
        
        # TF
        tf_msg = TransformStamped()
        tf_msg.header.stamp = timestamp
        tf_msg.header.frame_id = 'odom'
        tf_msg.child_frame_id = 'base_link'
        tf_msg.transform.translation.x = translation[0]
        tf_msg.transform.translation.y = translation[1]
        tf_msg.transform.translation.z = translation[2]
        tf_msg.transform.rotation.x = quat[0]
        tf_msg.transform.rotation.y = quat[1]
        tf_msg.transform.rotation.z = quat[2]
        tf_msg.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(tf_msg)

    def destroy_node(self):
        self.get_logger().info('Shutting down Enhanced SLAM node...')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        slam_node = RGBDSLAMNode()
        
        print("\n" + "="*70)
        print("Enhanced RGB-D SLAM - Waiting for data...")
        print("="*70)
        print("Subscribing to:")
        print("  - /camera/rgb/image_raw")
        print("  - /camera/depth/image_raw")
        print("\nPublishing to:")
        print("  - /odometry/gicp")
        print("  - TF: odom -> base_link")
        print("="*70 + "\n")
        
        rclpy.spin(slam_node)
        
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            slam_node.destroy_node()
        except:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()