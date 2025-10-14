#!/usr/bin/env python3

"""
ROS 2 Semantic Mapping Node with Rerun Visualization
MERGED: Correct transformations for both Ground Truth and SLAM modes
- Ground Truth mode: Uses OmniGibson convention (X-right, Y-up, Z-back local frame)
- SLAM mode: Uses ROS world coordinates (X-forward, Y-left, Z-up)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import message_filters
import rerun as rr


class SemanticMappingNode(Node):
    def __init__(self):
        super().__init__('semantic_mapping_node')
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Mapping parameters
        self.declare_parameter('voxel_size', 0.05)
        self.declare_parameter('max_points', 1000000)
        self.declare_parameter('min_depth', 0.5)
        self.declare_parameter('max_depth', 10.0)
        self.declare_parameter('downsample_stride', 4)
        self.declare_parameter('use_ground_truth', False)  # Flag to use ground truth or SLAM
        
        self.voxel_size = self.get_parameter('voxel_size').value
        self.max_points = self.get_parameter('max_points').value
        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value
        self.downsample_stride = self.get_parameter('downsample_stride').value
        self.use_ground_truth = self.get_parameter('use_ground_truth').value
        
        # Camera intrinsics
        self.fx = 600.0
        self.fy = 600.0
        self.cx = 320.0
        self.cy = 240.0
        self.image_width = 640
        self.image_height = 480
        self.camera_info_received = False
        
        # Current pose from odometry
        self.current_pose = np.eye(4)
        self.pose_received = False
        
        # Accumulated semantic point cloud map
        self.map_points = None
        self.map_colors = None
        self.map_semantic_ids = None
        
        # Accumulated instance point cloud map
        self.map_instance_points = None
        self.map_instance_colors = None
        self.map_instance_ids = None
        
        # Frame counter
        self.frame_count = 0
        
        # Initialize Rerun viewer
        rr.init("semantic_slam_map", spawn=True)
        self.get_logger().info('Rerun viewer initialized')
        
        # Subscribers
        # Subscribe to odometry (either ground truth or SLAM based on flag)
        if self.use_ground_truth:
            odom_topic = '/ground_truth/odom'
            self.get_logger().info('Using GROUND TRUTH pose for mapping')
            self.get_logger().info('Coordinate convention: OmniGibson (Local: X-right, Y-up, Z-back)')
        else:
            odom_topic = '/odometry/gicp'
            self.get_logger().info('Using SLAM (GICP) pose for mapping')
            self.get_logger().info('Coordinate convention: ROS World (X-forward, Y-left, Z-up)')
        
        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odometry_callback,
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/rgb/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Synchronized subscribers
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/rgb/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')
        self.semantic_sub = message_filters.Subscriber(self, Image, '/camera/semantic/image_raw')
        self.instance_sub = message_filters.Subscriber(self, Image, '/camera/instance/image_raw')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.semantic_sub, self.instance_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.sensor_callback)
        
        self.get_logger().info('Semantic Mapping Node initialized')
        self.get_logger().info(f'Camera frame: X-right, Y-down, Z-forward (RDF)')
        self.get_logger().info(f'Voxel size: {self.voxel_size}m')
        self.get_logger().info(f'Max points: {self.max_points}')
        self.get_logger().info('Waiting for odometry and sensor data...')

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

    def odometry_callback(self, msg):
        """Update current pose from odometry."""
        # Extract translation
        t = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        
        # Extract rotation quaternion
        q = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])
        
        # Convert quaternion to rotation matrix
        R = Rotation.from_quat(q).as_matrix()
        
        # Build 4x4 transformation matrix
        self.current_pose = np.eye(4)
        self.current_pose[:3, :3] = R
        self.current_pose[:3, 3] = t
        
        if not self.pose_received:
            self.pose_received = True
            pose_source = "Ground Truth" if self.use_ground_truth else "SLAM"
            self.get_logger().info(f'Received first {pose_source} pose')
        
        # Log camera pose to Rerun
        rr.set_time_sequence("frame", self.frame_count)
        rr.log("world/camera", rr.Transform3D(
            translation=t.tolist(),
            rotation=rr.Quaternion(xyzw=q.tolist())
        ))

    def depth_to_point_cloud_camera_frame(self, depth_image):
        """
        Convert depth image to 3D point cloud in CAMERA frame.
        
        Camera frame: X-right, Y-down, Z-forward (RDF convention)
        
        Args:
            depth_image: Depth image in meters (H x W)
            
        Returns:
            points_camera: Nx3 numpy array in CAMERA frame (RDF)
            pixel_coords: Nx2 numpy array of (v, u) pixel coordinates
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
        
        # Back-project to 3D in CAMERA frame (X-right, Y-down, Z-forward)
        u_valid = uu.flatten()[valid]
        v_valid = vv.flatten()[valid]
        z_valid = z[valid]
        
        x_cam = (u_valid - self.cx) / self.fx * z_valid
        y_cam = (v_valid - self.cy) / self.fy * z_valid
        z_cam = z_valid
        
        # Stack into Nx3 in camera frame (RDF)
        points_camera = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32)
        
        # Pixel coordinates (row, col)
        pixel_coords = np.stack([v_valid, u_valid], axis=1).astype(np.int32)
        
        return points_camera, pixel_coords

    def transform_camera_to_world(self, points_camera, pose):
        """
        Transform points from camera frame to world frame.
        
        Camera frame: X-right, Y-down, Z-forward (RDF)
        
        TWO DIFFERENT TRANSFORMATIONS:
        
        1. Ground Truth mode (OmniGibson convention):
           - Local frame: X-right, Y-up, Z-back (RUB)
           - Transform: [X, -Y, -Z] to convert Camera RDF -> Local RUB
           - Then apply pose directly
        
        2. SLAM mode (ROS world convention):
           - World frame: X-forward, Y-left, Z-up
           - Transform: [Z, -X, -Y] to convert Camera RDF -> World coords
           - Then apply pose
        
        Args:
            points_camera: Nx3 array in camera frame (X-right, Y-down, Z-forward)
            pose: 4x4 transformation matrix
            
        Returns:
            points_world: Nx3 array in world frame
        """
        if self.use_ground_truth:
            # Ground Truth mode: OmniGibson convention
            # Camera RDF -> Local RUB transformation
            # Camera: X-right, Y-down, Z-forward
            # Local:  X-right, Y-up, Z-back
            #
            # Transformation:
            #   X_local = X_camera  (right stays right)
            #   Y_local = -Y_camera (up = -down)
            #   Z_local = -Z_camera (back = -forward)
            
            points_local = np.zeros_like(points_camera)
            points_local[:, 0] = points_camera[:, 0]   # X = right
            points_local[:, 1] = -points_camera[:, 1]  # Y = up (was -down)
            points_local[:, 2] = -points_camera[:, 2]  # Z = back (was -forward)
            
            # Apply pose transformation
            R = pose[:3, :3]
            t = pose[:3, 3]
            points_world = (R @ points_local.T).T + t
            
        else:
            # SLAM mode: ROS world convention
            # Camera RDF -> ROS World (X-forward, Y-left, Z-up)
            #
            # Transformation:
            #   X_world = Z_camera  (forward = camera forward)
            #   Y_world = -X_camera (left = -right)
            #   Z_world = -Y_camera (up = -down)
            
            points_local_world = np.zeros_like(points_camera)
            points_local_world[:, 0] = points_camera[:, 2]   # X = forward (was Z)
            points_local_world[:, 1] = -points_camera[:, 0]  # Y = left (was -X)
            points_local_world[:, 2] = -points_camera[:, 1]  # Z = up (was -Y)
            
            # Apply SLAM pose
            R = pose[:3, :3]
            t = pose[:3, 3]
            points_world = (R @ points_local_world.T).T + t
        
        return points_world.astype(np.float32)

    def voxel_downsample(self, points, colors, labels):
        """Downsample point cloud using voxel grid filter."""
        if len(points) == 0:
            return points, colors, labels
        
        voxel_indices = np.floor(points / self.voxel_size).astype(np.int32)
        _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
        
        return points[unique_idx], colors[unique_idx], labels[unique_idx]

    def semantic_id_to_color(self, semantic_ids, max_id=None):
        """Convert semantic IDs to RGB colors using JET colormap."""
        if max_id is None:
            max_id = max(semantic_ids.max(), 1)
        
        normalized = (semantic_ids.astype(np.float32) / max_id * 255).astype(np.uint8)
        colored = cv2.applyColorMap(normalized.reshape(-1, 1), cv2.COLORMAP_JET)
        colors = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).reshape(-1, 3)
        
        return colors

    def instance_id_to_color(self, instance_ids):
        """Convert instance IDs to stable RGB colors using hash function."""
        h = (instance_ids.astype(np.uint32) * np.uint32(2654435761)) & np.uint32(0xFFFFFFFF)
        r = (h & np.uint32(0xFF)).astype(np.uint8)
        g = ((h >> np.uint32(8)) & np.uint32(0xFF)).astype(np.uint8)
        b = ((h >> np.uint32(16)) & np.uint32(0xFF)).astype(np.uint8)
        
        colors = np.stack([r, g, b], axis=1)
        colors = np.maximum(colors, 32).astype(np.uint8)
        
        return colors

    def sensor_callback(self, rgb_msg, depth_msg, semantic_msg, instance_msg):
        """Process synchronized sensor data and update semantic map."""
        if not self.pose_received:
            self.get_logger().warn('No odometry received yet')
            return
        
        if not self.camera_info_received:
            self.get_logger().warn('No camera info received yet')
            return
        
        try:
            # Set Rerun timeline
            rr.set_time_sequence("frame", self.frame_count)
            
            # Convert messages
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            semantic_image = self.bridge.imgmsg_to_cv2(semantic_msg, desired_encoding='passthrough')
            instance_image = self.bridge.imgmsg_to_cv2(instance_msg, desired_encoding='passthrough')
            
            # Log images
            rr.log("images/rgb", rr.Image(rgb_image))
            rr.log("images/depth", rr.DepthImage(depth_image))
            
            semantic_colored = self.semantic_id_to_color(semantic_image.flatten()).reshape(
                semantic_image.shape[0], semantic_image.shape[1], 3)
            rr.log("images/semantic", rr.Image(semantic_colored))
            
            instance_colored = self.instance_id_to_color(instance_image.flatten()).reshape(
                instance_image.shape[0], instance_image.shape[1], 3)
            rr.log("images/instance", rr.Image(instance_colored))
            
            # Convert depth to meters
            if depth_image.dtype == np.uint16:
                depth_image = depth_image.astype(np.float32) / 1000.0
            
            # Generate point cloud in CAMERA frame (RDF)
            points_camera, pixel_coords = self.depth_to_point_cloud_camera_frame(depth_image)
            
            if points_camera is None or len(points_camera) == 0:
                self.get_logger().warn('No valid points')
                return
            
            # Transform to world frame using mode-specific transformation
            points_world = self.transform_camera_to_world(points_camera, self.current_pose)
            
            # Get labels (matching downsampled points)
            semantic_ds = semantic_image[::self.downsample_stride, ::self.downsample_stride]
            instance_ds = instance_image[::self.downsample_stride, ::self.downsample_stride]
            rgb_ds = rgb_image[::self.downsample_stride, ::self.downsample_stride]
            
            # Extract labels at valid pixel locations
            v_coords = pixel_coords[:, 0] // self.downsample_stride
            u_coords = pixel_coords[:, 1] // self.downsample_stride
            
            semantic_labels = semantic_ds[v_coords, u_coords]
            instance_labels = instance_ds[v_coords, u_coords]
            rgb_colors = rgb_ds[v_coords, u_coords]
            
            # Convert IDs to colors
            semantic_colors = self.semantic_id_to_color(semantic_labels)
            instance_colors = self.instance_id_to_color(instance_labels)
            
            # Log current frame
            rr.log("world/current_frame/rgb", rr.Points3D(positions=points_world, colors=rgb_colors))
            rr.log("world/current_frame/semantic", rr.Points3D(positions=points_world, colors=semantic_colors))
            rr.log("world/current_frame/instance", rr.Points3D(positions=points_world, colors=instance_colors))
            
            # Accumulate semantic map
            if self.map_points is None:
                self.map_points = points_world
                self.map_colors = semantic_colors
                self.map_semantic_ids = semantic_labels
            else:
                self.map_points = np.vstack([self.map_points, points_world])
                self.map_colors = np.vstack([self.map_colors, semantic_colors])
                self.map_semantic_ids = np.concatenate([self.map_semantic_ids, semantic_labels])
                
                self.map_points, self.map_colors, self.map_semantic_ids = \
                    self.voxel_downsample(self.map_points, self.map_colors, self.map_semantic_ids)
                
                if len(self.map_points) > self.max_points:
                    self.map_points = self.map_points[-self.max_points:]
                    self.map_colors = self.map_colors[-self.max_points:]
                    self.map_semantic_ids = self.map_semantic_ids[-self.max_points:]
            
            rr.log("world/map/semantic", rr.Points3D(positions=self.map_points, colors=self.map_colors))
            
            # Accumulate instance map
            if self.map_instance_points is None:
                self.map_instance_points = points_world
                self.map_instance_colors = instance_colors
                self.map_instance_ids = instance_labels
            else:
                self.map_instance_points = np.vstack([self.map_instance_points, points_world])
                self.map_instance_colors = np.vstack([self.map_instance_colors, instance_colors])
                self.map_instance_ids = np.concatenate([self.map_instance_ids, instance_labels])
                
                self.map_instance_points, self.map_instance_colors, self.map_instance_ids = \
                    self.voxel_downsample(self.map_instance_points, self.map_instance_colors, self.map_instance_ids)
                
                if len(self.map_instance_points) > self.max_points:
                    self.map_instance_points = self.map_instance_points[-self.max_points:]
                    self.map_instance_colors = self.map_instance_colors[-self.max_points:]
                    self.map_instance_ids = self.map_instance_ids[-self.max_points:]
            
            rr.log("world/map/instance", rr.Points3D(positions=self.map_instance_points, colors=self.map_instance_colors))
            
            # Log camera pinhole
            rr.log("world/camera", rr.Pinhole(
                resolution=[self.image_width, self.image_height],
                image_from_camera=[[self.fx, 0.0, self.cx], 
                                  [0.0, self.fy, self.cy], 
                                  [0.0, 0.0, 1.0]],
                camera_xyz=rr.ViewCoordinates.RDF,
            ))
            
            self.frame_count += 1
            
            if self.frame_count % 30 == 0:
                t = self.current_pose[:3, 3]
                mode = "GT" if self.use_ground_truth else "SLAM"
                self.get_logger().info(
                    f'Frame {self.frame_count} [{mode}]: '
                    f'Pose=[X:{t[0]:.2f}m, Y:{t[1]:.2f}m, Z:{t[2]:.2f}m], '
                    f'Semantic: {len(self.map_points)} pts, '
                    f'Instance: {len(self.map_instance_points)} pts'
                )
            
        except Exception as e:
            self.get_logger().error(f'Error: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def save_maps(self, filename_prefix='semantic_map'):
        """Save accumulated maps to disk."""
        if self.map_points is not None:
            np.save(f'{filename_prefix}_points.npy', self.map_points)
            np.save(f'{filename_prefix}_colors.npy', self.map_colors)
            np.save(f'{filename_prefix}_labels.npy', self.map_semantic_ids)
            self.get_logger().info(f'Saved semantic map: {len(self.map_points)} points')
        
        if self.map_instance_points is not None:
            np.save(f'{filename_prefix}_instance_points.npy', self.map_instance_points)
            np.save(f'{filename_prefix}_instance_colors.npy', self.map_instance_colors)
            np.save(f'{filename_prefix}_instance_labels.npy', self.map_instance_ids)
            self.get_logger().info(f'Saved instance map: {len(self.map_instance_points)} points')

    def destroy_node(self):
        self.get_logger().info('Shutting down...')
        self.save_maps()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        mapping_node = SemanticMappingNode()
        
        print("\n" + "="*70)
        print("Semantic Mapping with Dual-Mode Support")
        print("="*70)
        print("Camera frame: X-right, Y-down, Z-forward (RDF)")
        print("\nMODE-SPECIFIC TRANSFORMATIONS:")
        print("  Ground Truth: [X, -Y, -Z] → Local RUB (OmniGibson)")
        print("  SLAM (GICP):  [Z, -X, -Y] → World FLU (ROS)")
        print("\nSubscribing to:")
        print("  - Pose: /ground_truth/odom OR /odometry/gicp")
        print("  - Sensors: /camera/*/image_raw")
        print("\nUsage:")
        print("  SLAM mode (default):  python3 slam_map.py")
        print("  Ground Truth mode:    python3 slam_map.py --ros-args -p use_ground_truth:=true")
        print("\nPress Ctrl+C to save and exit")
        print("="*70 + "\n")
        
        rclpy.spin(mapping_node)
        
    except KeyboardInterrupt:
        print("\nSaving maps...")
    finally:
        try:
            mapping_node.destroy_node()
        except:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()