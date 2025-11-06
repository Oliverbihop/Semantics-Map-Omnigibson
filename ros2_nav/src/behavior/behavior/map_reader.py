import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
import cv2
import numpy as np
import os
import time
from geometry_msgs.msg import Twist, PoseStamped
import random
from visualization_msgs.msg import Marker
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

class MapReader(Node):
    def __init__(self):
        super().__init__('map_reader')
        
        # Create publisher to /cmd_vel
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(Marker, '/robot_marker', 10)
        self.subcriber_ = self.create_subscription(PoseStamped, '/robot_pose', self.pose_callback, 10)
        # Publish at 10 Hz
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Parameter: image path
        self.declare_parameter(
            'image_path',
            '/home/luan/ros2_behavior/src/behavior/maps/trav_map.png'
        )
        img_path = self.get_parameter('image_path').get_parameter_value().string_value

        if not os.path.exists(img_path):
            self.get_logger().error(f"Image file not found: {img_path}")
            return

        # Read grayscale image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.get_logger().error("Failed to load image")
            return

        self.get_logger().info(f"Loaded traversibility map {img.shape} from {img_path}")

        # Convert image → occupancy grid
        # Convention: 0=free, 100=occupied, -1=unknown
        occ_grid = np.zeros_like(img, dtype=np.int8)

        # Threshold: treat darker pixels as obstacles
        occ_grid[img < 127] = 100     # Occupied
        occ_grid[img >= 127] = 0      # Free

        # Store for publishing
        self.grid_msg = OccupancyGrid()
        self.grid_msg.header = Header()
        self.grid_msg.header.frame_id = "map"   # RViz uses "map" frame
        self.grid_msg.info.resolution = 0.01 # 5cm per pixel (set to match your map)
        self.grid_msg.info.width = img.shape[1]
        self.grid_msg.info.height = img.shape[0]
        self.grid_msg.info.origin.position.x = 0.0
        self.grid_msg.info.origin.position.y = 0.0
        self.grid_msg.info.origin.position.z = 0.001
        self.grid_msg.info.origin.orientation.w = 1.0

        # Flatten image row-major
        self.grid_msg.data = occ_grid.flatten().tolist()

        # Publisher
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        self.pub = self.create_publisher(OccupancyGrid, 'map', qos)
        # self.pub = self.create_publisher(OccupancyGrid, 'map', 10)

        # Timer to republish
        self.timer = self.create_timer(1.0, self.publish_map)

    def publish_map(self):
        self.grid_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.grid_msg)
        self.get_logger().info("Published occupancy grid map")
        
    def pose_callback(self, msg: PoseStamped):
        # Create a Marker
        marker = Marker()
        marker.header.frame_id = msg.header.frame_id  # usually "world" or "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "robot"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Position
        marker.pose.position.x = msg.pose.position.x
        marker.pose.position.y = msg.pose.position.y
        marker.pose.position.z = msg.pose.position.z

        # Orientation (optional, sphere doesn't need it)
        marker.pose.orientation.x = msg.pose.orientation.x
        marker.pose.orientation.y = msg.pose.orientation.y
        marker.pose.orientation.z = msg.pose.orientation.z
        marker.pose.orientation.w = msg.pose.orientation.w

        # Scale (size of the sphere)
        marker.scale.x = 0.1  # 10 cm
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        # Color (red)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Lifetime (0 = forever)
        marker.lifetime.sec = 0

        # Publish the marker
        self.marker_pub.publish(marker)
    	
        
        
    def timer_callback(self):
        msg = Twist()
        # Random linear velocity between 0.0 and 0.5 m/s
        msg.linear.x = random.uniform(0.0, 0.5)
        # Random angular velocity between -1.0 and 1.0 rad/s
        msg.angular.z = random.uniform(-1.0, 1.0)
       	#self.publisher_.publish(msg)
        # self.get_logger().info(f'Publishing: linear.x={msg.linear.x:.2f}, angular.z={msg.angular.z:.2f}')


def main(args=None):
    rclpy.init(args=args)
    node = MapReader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

