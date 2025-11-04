#!/usr/bin/env python3

"""
Integrated RGB-D SLAM and Semantic Mapping Node
Combines odometry estimation and mapping in a single process for perfect synchronization.
Only publishes final odometry and visualizes the map in Rerun.
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
import rerun as rr
import open3d as o3d


class IntegratedSLAMNode(Node):
    def __init__(self):
        super().__init__('integrated_slam_node')
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # SLAM parameters
        self.declare_parameter('downsample_resolution', 0.05)
        self.declare_parameter('max_correspondence_distance', 0.5)
        self.declare_parameter('num_threads', 4)
        self.declare_parameter('skip_frames', 1)
        
        # Keyframe parameters
        self.declare_parameter('keyframe_distance', 0.1)  # meters
        self.declare_parameter('keyframe_rotation', 2.0)  # degrees
        self.declare_parameter('max_keyframes', 10000)
        
        # Mapping parameters
        self.declare_parameter('voxel_size', 0.05)
        self.declare_parameter('max_map_points', 1000000)
        self.declare_parameter('min_depth', 0.5)
        self.declare_parameter('max_depth', 10.0)
        self.declare_parameter('downsample_stride', 4)
        
        # Get parameters
        self.downsample_resolution = self.get_parameter('downsample_resolution').value
        self.max_correspondence_distance = self.get_parameter('max_correspondence_distance').value
        self.num_threads = self.get_parameter('num_threads').value
        self.skip_frames = self.get_parameter('skip_frames').value
        
        self.keyframe_distance = self.get_parameter('keyframe_distance').value
        self.keyframe_rotation = self.get_parameter('keyframe_rotation').value
        self.max_keyframes = self.get_parameter('max_keyframes').value
        
        self.voxel_size = self.get_parameter('voxel_size').value
        self.max_map_points = self.get_parameter('max_map_points').value
        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value
        self.downsample_stride = self.get_parameter('downsample_stride').value
        
        # Camera intrinsics
        self.fx = 600.0
        self.fy = 600.0
        self.cx = 320.0
        self.cy = 240.0
        self.image_width = 640
        self.image_height = 480
        self.camera_info_received = False
        
        # SLAM state (in WORLD coordinates: X-forward, Y-left, Z-up)
        self.previous_cloud = None
        self.current_pose = np.eye(4)
        self.frame_count = 0
        self.frame_skip_counter = 0
        
        # Keyframe management
        self.keyframes = deque(maxlen=self.max_keyframes)
        self.last_keyframe_pose = np.eye(4)
        
        # Motion model for outlier detection
        self.previous_pose = np.eye(4)
        self.velocity_history = deque(maxlen=5)
        
        # Accumulated semantic map
        self.map_points = None
        self.map_colors = None
        self.map_semantic_ids = None
        
        # Accumulated instance map
        self.map_instance_points = None
        self.map_instance_colors = None
        self.map_instance_ids = None
        
        # Accumulated RGB map
        self.map_rgb_points = None
        self.map_rgb_colors = None
        
        # Initialize GICP
        self.gicp = pygicp.FastGICP()
        self.gicp.set_num_threads(self.num_threads)
        self.gicp.set_max_correspondence_distance(self.max_correspondence_distance)
        
        # Initialize Rerun
        rr.init("integrated_slam", spawn=True)
        self.get_logger().info('Rerun viewer initialized')
        
        # Subscribers
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/rgb/camera_info',
            self.camera_info_callback,
            10
        )
        
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/rgb/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')
        self.semantic_sub = message_filters.Subscriber(self, Image, '/camera/semantic/image_raw')
        self.instance_sub = message_filters.Subscriber(self, Image, '/camera/instance/image_raw')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.semantic_sub, self.instance_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.integrated_callback)
        
        self.processing = False
        
        # Publishers (only odometry output)
        self.odom_publisher = self.create_publisher(Odometry, '/odometry/gicp', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Performance tracking
        self.last_log_time = time.time()
        self.received_count = 0
        self.processed_count = 0
        
        self.get_logger().info('='*70)
        self.get_logger().info('Integrated SLAM and Mapping Node initialized')
        self.get_logger().info(f'Keyframe distance: {self.keyframe_distance}m')
        self.get_logger().info(f'Keyframe rotation: {self.keyframe_rotation}°')
        self.get_logger().info(f'Map voxel size: {self.voxel_size}m')
        self.get_logger().info('Coordinate system: X-forward, Y-left, Z-up (ROS)')
        self.get_logger().info('Waiting for RGB-D data...')
        self.get_logger().info('='*70)

    def camera_info_callback(self, msg):
        """Update camera intrinsics from CameraInfo message."""
        if not self.camera_info_received:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.image_width = msg.width
            self.image_height = msg.height
            self.camera_info_received = True
            self.get_logger().info(f'Camera intrinsics: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}')

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

    def depth_to_point_cloud_world(self, depth_image):
        """
        Convert depth image to 3D point cloud in WORLD frame.
        Camera frame (RDF): X-right, Y-down, Z-forward
        World frame: X-forward, Y-left, Z-up
        
        Returns:
            points_world: Nx3 array in world coordinates
            pixel_coords: Nx2 array of (v, u) pixel coordinates
        """
        h, w = depth_image.shape
        
        # Downsample for performance
        stride = self.downsample_stride
        depth_ds = depth_image[::stride, ::stride]
        h_ds, w_ds = depth_ds.shape
        
        # Create mesh grid
        u = (np.arange(0, w_ds) * stride).astype(np.float32)
        v = (np.arange(0, h_ds) * stride).astype(np.float32)
        uu, vv = np.meshgrid(u, v)
        
        # Get depth values
        z = depth_ds.flatten()
        
        # Filter valid depth
        valid = np.isfinite(z) & (z > self.min_depth) & (z < self.max_depth)
        
        if not np.any(valid):
            return None, None
        
        # Back-project to 3D in camera frame (RDF)
        u_valid = uu.flatten()[valid]
        v_valid = vv.flatten()[valid]
        z_valid = z[valid]
        
        x_cam = (u_valid - self.cx) / self.fx * z_valid
        y_cam = (v_valid - self.cy) / self.fy * z_valid
        z_cam = z_valid
        
        # Transform to world frame: [Z, -X, -Y]
        points_world = np.zeros((len(x_cam), 3), dtype=np.float32)
        points_world[:, 0] = z_cam   # X = forward (was Z in camera)
        points_world[:, 1] = -x_cam  # Y = left (was -X in camera)
        points_world[:, 2] = -y_cam  # Z = up (was -Y in camera)
        
        # Pixel coordinates (row, col)
        pixel_coords = np.stack([v_valid, u_valid], axis=1).astype(np.int32)
        
        return points_world, pixel_coords

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
        
        translation = np.linalg.norm(transformation[:3, 3])
        
        velocities = np.array(list(self.velocity_history))
        mean_vel = np.mean(velocities)
        std_vel = np.std(velocities)
        
        threshold = mean_vel + 3 * std_vel
        
        if translation > max(threshold, 1.0):
            self.get_logger().warn(f'Motion outlier: {translation:.2f}m vs avg {mean_vel:.2f}m')
            return True
        
        return False

    def compute_adaptive_covariance(self, transformation, fitness_score=None):
        """Compute adaptive covariance based on motion and fitness."""
        base_cov = 0.01
        
        translation = np.linalg.norm(transformation[:3, 3])
        translation_factor = 1.0 + translation
        
        if fitness_score is not None:
            fitness_factor = 1.0 / max(fitness_score, 0.1)
        else:
            fitness_factor = 1.0
        
        adaptive_cov = base_cov * translation_factor * fitness_factor
        
        covariance = np.zeros((6, 6))
        covariance[0, 0] = adaptive_cov
        covariance[1, 1] = adaptive_cov
        covariance[2, 2] = adaptive_cov
        covariance[3, 3] = adaptive_cov * 0.5
        covariance[4, 4] = adaptive_cov * 0.5
        covariance[5, 5] = adaptive_cov * 0.5
        
        return covariance.flatten().tolist()

    def transform_points_to_world(self, points_local, pose):
        """Transform points from local world frame to global world frame."""
        R = pose[:3, :3]
        t = pose[:3, 3]
        points_global = (R @ points_local.T).T + t
        return points_global.astype(np.float32)

    def semantic_id_to_color(self, semantic_ids, max_id=None):
        """Convert semantic IDs to RGB colors using JET colormap."""
        if max_id is None:
            max_id = max(semantic_ids.max(), 1)
        
        normalized = (semantic_ids.astype(np.float32) / max_id * 255).astype(np.uint8)
        colored = cv2.applyColorMap(normalized.reshape(-1, 1), cv2.COLORMAP_JET)
        colors = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).reshape(-1, 3)
        
        return colors

    def instance_id_to_color(self, instance_ids):
        """Convert instance IDs to stable RGB colors."""
        h = (instance_ids.astype(np.uint32) * np.uint32(2654435761)) & np.uint32(0xFFFFFFFF)
        r = (h & np.uint32(0xFF)).astype(np.uint8)
        g = ((h >> np.uint32(8)) & np.uint32(0xFF)).astype(np.uint8)
        b = ((h >> np.uint32(16)) & np.uint32(0xFF)).astype(np.uint8)
        
        colors = np.stack([r, g, b], axis=1)
        colors = np.maximum(colors, 32).astype(np.uint8)
        
        return colors

    def voxel_downsample_map(self, points, colors, labels):
        """Downsample map using voxel grid filter."""
        if len(points) == 0:
            return points, colors, labels
        
        voxel_indices = np.floor(points / self.voxel_size).astype(np.int32)
        _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
        
        return points[unique_idx], colors[unique_idx], labels[unique_idx]

    def update_map(self, points_world, rgb_colors, semantic_labels, instance_labels):
        """Update the accumulated map with new keyframe data."""
        # Convert labels to colors
        semantic_colors = self.semantic_id_to_color(semantic_labels).astype(np.float32) / 255.0
        instance_colors = self.instance_id_to_color(instance_labels).astype(np.float32) / 255.0
        
        # Update semantic map
        if self.map_points is None:
            self.map_points = points_world
            self.map_colors = semantic_colors
            self.map_semantic_ids = semantic_labels
        else:
            self.map_points = np.vstack([self.map_points, points_world])
            self.map_colors = np.vstack([self.map_colors, semantic_colors])
            self.map_semantic_ids = np.concatenate([self.map_semantic_ids, semantic_labels])
            
            # Downsample
            self.map_points, self.map_colors, self.map_semantic_ids = \
                self.voxel_downsample_map(self.map_points, self.map_colors, self.map_semantic_ids)
            
            # Limit size
            if len(self.map_points) > self.max_map_points:
                self.map_points = self.map_points[-self.max_map_points:]
                self.map_colors = self.map_colors[-self.max_map_points:]
                self.map_semantic_ids = self.map_semantic_ids[-self.max_map_points:]
        
        # Update instance map
        if self.map_instance_points is None:
            self.map_instance_points = points_world
            self.map_instance_colors = instance_colors
            self.map_instance_ids = instance_labels
        else:
            self.map_instance_points = np.vstack([self.map_instance_points, points_world])
            self.map_instance_colors = np.vstack([self.map_instance_colors, instance_colors])
            self.map_instance_ids = np.concatenate([self.map_instance_ids, instance_labels])
            
            # Downsample
            self.map_instance_points, self.map_instance_colors, self.map_instance_ids = \
                self.voxel_downsample_map(self.map_instance_points, self.map_instance_colors, self.map_instance_ids)
            
            # Limit size
            if len(self.map_instance_points) > self.max_map_points:
                self.map_instance_points = self.map_instance_points[-self.max_map_points:]
                self.map_instance_colors = self.map_instance_colors[-self.max_map_points:]
                self.map_instance_ids = self.map_instance_ids[-self.max_map_points:]
        
        # Update RGB map
        rgb_colors_normalized = rgb_colors.astype(np.float32) / 255.0
        if self.map_rgb_points is None:
            self.map_rgb_points = points_world.copy()
            self.map_rgb_colors = rgb_colors_normalized
        else:
            self.map_rgb_points = np.vstack([self.map_rgb_points, points_world])
            self.map_rgb_colors = np.vstack([self.map_rgb_colors, rgb_colors_normalized])
            
            # Downsample (reusing voxel_downsample_map, but we don't need labels for RGB)
            voxel_indices = np.floor(self.map_rgb_points / self.voxel_size).astype(np.int32)
            _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
            self.map_rgb_points = self.map_rgb_points[unique_idx]
            self.map_rgb_colors = self.map_rgb_colors[unique_idx]
            
            # Limit size
            if len(self.map_rgb_points) > self.max_map_points:
                self.map_rgb_points = self.map_rgb_points[-self.max_map_points:]
                self.map_rgb_colors = self.map_rgb_colors[-self.max_map_points:]

    def integrated_callback(self, rgb_msg, depth_msg, semantic_msg, instance_msg):
        """Integrated callback that performs SLAM and mapping synchronously."""
        self.received_count += 1
        
        if self.received_count == 1:
            self.get_logger().info('✓ First synchronized message received!')
        
        if self.processing:
            return
        
        # Frame skipping
        self.frame_skip_counter += 1
        if self.frame_skip_counter <= self.skip_frames:
            return
        self.frame_skip_counter = 0
        
        self.processing = True
        
        try:
            # Set Rerun timeline
            rr.set_time_sequence("frame", self.frame_count)
            
            # Convert messages
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            semantic_image = self.bridge.imgmsg_to_cv2(semantic_msg, desired_encoding='passthrough')
            instance_image = self.bridge.imgmsg_to_cv2(instance_msg, desired_encoding='passthrough')
            
            # Convert depth to meters
            if depth_image.dtype == np.uint16:
                depth_image = depth_image.astype(np.float32) / 1000.0
            
            # Clean depth
            if not np.all(np.isfinite(depth_image)):
                depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Generate point cloud in WORLD coordinates (for SLAM)
            points_local, pixel_coords = self.depth_to_point_cloud_world(depth_image)
            
            if points_local is None or len(points_local) == 0:
                self.get_logger().warn('No valid points')
                return
            
            # Clean point cloud
            points_local = self.clean_point_cloud(points_local)
            
            if len(points_local) < 100:
                self.get_logger().warn(f'Not enough points: {len(points_local)} < 100')
                return
            
            # Downsample for SLAM
            if self.downsample_resolution > 0:
                points_slam = self.manual_voxel_downsample(points_local, self.downsample_resolution)
                points_slam = self.clean_point_cloud(points_slam)
                
                if len(points_slam) < 50:
                    self.get_logger().warn(f'Not enough points after downsample: {len(points_slam)} < 50')
                    return
            else:
                points_slam = points_local
            
            # GICP alignment
            is_keyframe = False
            if self.previous_cloud is not None:
                try:
                    # Set target and source
                    self.gicp.set_input_target(self.previous_cloud)
                    self.gicp.set_input_source(points_slam)
                    
                    # Align
                    transformation = self.gicp.align()
                    
                    # Get fitness score
                    try:
                        fitness_score = self.gicp.get_fitness_score()
                    except:
                        fitness_score = 1.0
                    
                    # Validate transformation
                    if not np.all(np.isfinite(transformation)):
                        self.get_logger().error('Invalid transformation')
                        return
                    
                    translation_norm = np.linalg.norm(transformation[:3, 3])
                    
                    # Outlier rejection
                    if self.is_motion_outlier(transformation):
                        return
                    
                    # Update velocity history
                    self.velocity_history.append(translation_norm)
                    
                    if translation_norm > 2.0:
                        self.get_logger().warn(f'Large translation rejected: {translation_norm:.2f}m')
                        return
                    
                    # Update pose
                    self.previous_pose = self.current_pose.copy()
                    self.current_pose = self.current_pose @ transformation
                    
                    # Check if this should be a keyframe
                    is_keyframe = self.check_keyframe(self.current_pose)
                    
                    if is_keyframe:
                        # This is a keyframe - update map with current data
                        # Extract labels at valid pixel locations
                        v_coords = pixel_coords[:, 0] // self.downsample_stride
                        u_coords = pixel_coords[:, 1] // self.downsample_stride
                        
                        semantic_ds = semantic_image[::self.downsample_stride, ::self.downsample_stride]
                        instance_ds = instance_image[::self.downsample_stride, ::self.downsample_stride]
                        rgb_ds = rgb_image[::self.downsample_stride, ::self.downsample_stride]
                        
                        semantic_labels = semantic_ds[v_coords, u_coords]
                        instance_labels = instance_ds[v_coords, u_coords]
                        rgb_colors = rgb_ds[v_coords, u_coords]
                        
                        # Transform points to global world frame
                        points_world = self.transform_points_to_world(points_local, self.current_pose)
                        
                        # Update map
                        self.update_map(points_world, rgb_colors, semantic_labels, instance_labels)
                        
                        # Store keyframe
                        self.keyframes.append((self.current_pose.copy(), points_slam.copy()))
                        self.last_keyframe_pose = self.current_pose.copy()
                    
                    # Publish odometry
                    covariance = self.compute_adaptive_covariance(transformation, fitness_score)
                    self.publish_odometry(rgb_msg.header.stamp, covariance)
                    
                    # Visualization in Rerun
                    self.visualize_rerun(rgb_image, depth_image, semantic_image, instance_image,
                                        points_local, pixel_coords, is_keyframe, fitness_score)
                    
                    self.processed_count += 1
                    
                except Exception as e:
                    self.get_logger().error(f'GICP alignment failed: {str(e)}')
                    import traceback
                    self.get_logger().error(traceback.format_exc())
                    return
            else:
                # First frame - initialize
                self.get_logger().info(f'✓ First frame initialized with {len(points_slam)} points')
                
                # Initialize as keyframe
                is_keyframe = True
                
                # Extract labels
                v_coords = pixel_coords[:, 0] // self.downsample_stride
                u_coords = pixel_coords[:, 1] // self.downsample_stride
                
                semantic_ds = semantic_image[::self.downsample_stride, ::self.downsample_stride]
                instance_ds = instance_image[::self.downsample_stride, ::self.downsample_stride]
                rgb_ds = rgb_image[::self.downsample_stride, ::self.downsample_stride]
                
                semantic_labels = semantic_ds[v_coords, u_coords]
                instance_labels = instance_ds[v_coords, u_coords]
                rgb_colors = rgb_ds[v_coords, u_coords]
                
                # Initialize map (at origin)
                self.update_map(points_local, rgb_colors, semantic_labels, instance_labels)
                
                # Store keyframe
                self.keyframes.append((self.current_pose.copy(), points_slam.copy()))
                self.last_keyframe_pose = self.current_pose.copy()
                
                # Publish initial pose
                covariance = [0.01] * 36
                for i in range(6):
                    covariance[i * 6 + i] = 0.01
                
                self.publish_odometry(rgb_msg.header.stamp, covariance)
                
                # Visualization
                self.visualize_rerun(rgb_image, depth_image, semantic_image, instance_image,
                                    points_local, pixel_coords, is_keyframe, 1.0)
            
            # Store previous cloud
            if np.all(np.isfinite(points_slam)) and len(points_slam) >= 50:
                self.previous_cloud = points_slam.copy()
            
            self.frame_count += 1
            
            # Log progress
            current_time = time.time()
            if current_time - self.last_log_time >= 3.0:
                elapsed = current_time - self.last_log_time
                rate = self.processed_count / elapsed if elapsed > 0 else 0
                
                t = self.current_pose[:3, 3]
                self.get_logger().info(
                    f'Received: {self.received_count} | Processed: {self.processed_count} | '
                    f'Rate: {rate:.1f} Hz | Pos=[{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}] | '
                    f'KF: {len(self.keyframes)} | Map: {len(self.map_points) if self.map_points is not None else 0} pts'
                )
                self.last_log_time = current_time
                self.processed_count = 0
                self.received_count = 0
                
        except Exception as e:
            self.get_logger().error(f'Callback error: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())
        finally:
            self.processing = False

    def visualize_rerun(self, rgb_image, depth_image, semantic_image, instance_image,
                       points_local, pixel_coords, is_keyframe, fitness_score):
        """Visualize data in Rerun."""
        # Log images
        rr.log("images/rgb", rr.Image(rgb_image))
        rr.log("images/depth", rr.DepthImage(depth_image))
        
        semantic_colored = self.semantic_id_to_color(semantic_image.flatten()).reshape(
            semantic_image.shape[0], semantic_image.shape[1], 3)
        rr.log("images/semantic", rr.Image(semantic_colored))
        
        instance_colored = self.instance_id_to_color(instance_image.flatten()).reshape(
            instance_image.shape[0], instance_image.shape[1], 3)
        rr.log("images/instance", rr.Image(instance_colored))
        
        # Log camera pose
        t = self.current_pose[:3, 3]
        R = self.current_pose[:3, :3]
        q = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
        
        rr.log("world/camera", rr.Transform3D(
            translation=t.tolist(),
            rotation=rr.Quaternion(xyzw=q.tolist())
        ))
        
        # Log camera pinhole
        rr.log("world/camera", rr.Pinhole(
            resolution=[self.image_width, self.image_height],
            image_from_camera=[[self.fx, 0.0, self.cx], 
                              [0.0, self.fy, self.cy], 
                              [0.0, 0.0, 1.0]],
            camera_xyz=rr.ViewCoordinates.RDF,
        ))
        
        # Log accumulated maps
        if self.map_points is not None:
            rr.log("world/map/semantic", rr.Points3D(positions=self.map_points, colors=self.map_colors))
        
        if self.map_instance_points is not None:
            rr.log("world/map/instance", rr.Points3D(positions=self.map_instance_points, colors=self.map_instance_colors))
        
        if self.map_rgb_points is not None:
            rr.log("world/map/rgb", rr.Points3D(positions=self.map_rgb_points, colors=self.map_rgb_colors))
        
        # Log keyframe indicator
        if is_keyframe:
            rr.log("info/keyframe", rr.TextLog(f"KEYFRAME {len(self.keyframes)}", level=rr.TextLogLevel.INFO))

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

    def save_maps(self, filename_prefix='integrated_slam_map'):
        """Save accumulated maps to disk as PLY files."""
        if self.map_points is not None:
            semantic_pcd = o3d.geometry.PointCloud()
            semantic_pcd.points = o3d.utility.Vector3dVector(self.map_points)
            semantic_pcd.colors = o3d.utility.Vector3dVector(self.map_colors)
            o3d.io.write_point_cloud(f'{filename_prefix}_semantic.ply', semantic_pcd) 
            self.get_logger().info(f'Saved semantic map: {len(self.map_points)} points')
        
        if self.map_instance_points is not None:
            instance_pcd = o3d.geometry.PointCloud()
            instance_pcd.points = o3d.utility.Vector3dVector(self.map_instance_points)
            instance_pcd.colors = o3d.utility.Vector3dVector(self.map_instance_colors)
            o3d.io.write_point_cloud(f'{filename_prefix}_instance.ply', instance_pcd)
            self.get_logger().info(f'Saved instance map: {len(self.map_instance_points)} points')
        
        if self.map_rgb_points is not None:
            rgb_pcd = o3d.geometry.PointCloud()
            rgb_pcd.points = o3d.utility.Vector3dVector(self.map_rgb_points)
            rgb_pcd.colors = o3d.utility.Vector3dVector(self.map_rgb_colors)
            o3d.io.write_point_cloud(f'{filename_prefix}_rgb.ply', rgb_pcd)
            self.get_logger().info(f'Saved RGB map: {len(self.map_rgb_points)} points')

    def destroy_node(self):
        self.save_maps()
        self.get_logger().info('Shutting down Integrated SLAM node...')
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        slam_node = IntegratedSLAMNode()
        
        print("\n" + "="*70)
        print("Integrated RGB-D SLAM and Semantic Mapping")
        print("="*70)
        print("\nCoordinate System:")
        print("  Camera frame: X-right, Y-down, Z-forward (RDF)")
        print("  World frame:  X-forward, Y-left, Z-up (ROS)")
        print("\nSubscribing to:")
        print("  - /camera/rgb/image_raw")
        print("  - /camera/depth/image_raw")
        print("  - /camera/semantic/image_raw")
        print("  - /camera/instance/image_raw")
        print("\nPublishing to:")
        print("  - /odometry/gicp")
        print("  - TF: odom -> base_link")
        print("\nVisualization:")
        print("  - Rerun viewer with semantic/instance maps")
        print("  - Camera trajectory and keyframes")
        print("\nPress Ctrl+C to save maps and exit")
        print("="*70 + "\n")
        
        rclpy.spin(slam_node)
        
    except KeyboardInterrupt:
        print("\nSaving maps...")
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