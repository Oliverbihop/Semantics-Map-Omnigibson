import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from visualization_msgs.msg import Marker
import cv2
import numpy as np
import os
import time

class Nav2NavigationNode(Node):
    def __init__(self):
        super().__init__('nav2_navigation_node')
        
        # Control mode: only nav2_direct
        self.control_mode = 'nav2_direct'
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(Marker, '/robot_marker', 10)
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        
        # Subscribers
        self.pose_subscriber = self.create_subscription(
            PoseStamped, 
            '/robot_pose', 
            self.pose_callback, 
            10
        )
        
        # Subscribe to Nav2's velocity commands to republish them
        self.nav2_cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel_nav',  # Nav2's velocity output (remap this in launch file)
            self.nav2_cmd_vel_callback,
            10
        )

        self.goal_subscriber = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_pose_callback,
            10
        )
        
        # Nav2 Action Client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Control variables
        self.current_pose = None
        self.navigation_goal_handle = None
        
        # Check if Nav2 is available (with timeout)
        self.nav2_available = False
        self.get_logger().info("Checking for Nav2 navigation server...")
        
        if self.nav_client.wait_for_server(timeout_sec=5.0):
            self.nav2_available = True
            self.get_logger().info("Nav2 navigation server is ready!")
        else:
            self.get_logger().warning("Nav2 navigation server not available. Navigation features disabled.")
            self.get_logger().info("To enable navigation, launch Nav2 with:")
            self.get_logger().info("  ros2 launch nav2_bringup navigation_launch.py")
        
        # Parameter: image path
        self.declare_parameter(
            'image_path',
            '~/ros2_nav/src/behavior/maps/trav_map.png'
        )
        
        # Load and publish map
        self.load_and_publish_map()
        
        # Set initial pose after a short delay
        self.create_timer(2.0, self.set_initial_pose_once)
        #self.set_initial_pose_once()
        
        # Example: Navigate to a goal after 5 seconds
        # self.create_timer(5.0, self.navigate_to_example_goal)

    def goal_pose_callback(self, msg: PoseStamped):
        """Receive a goal pose from /goal_pose and send it to Nav2"""
        if not self.nav2_available:
            self.get_logger().warning("Nav2 not available. Cannot send goal.")
            return

        x = msg.pose.position.x
        y = msg.pose.position.y

        # Convert orientation from quaternion to yaw
        import math
        q = msg.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))

        self.get_logger().info(f"Received goal from /goal_pose: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")
        
        self.navigate_to_pose(x, y, yaw, frame_id=msg.header.frame_id)

    def nav2_cmd_vel_callback(self, msg: Twist):
        """Receive velocity commands from Nav2 and republish to /cmd_vel"""
        # Republish Nav2's velocity commands to /cmd_vel
        self.cmd_vel_pub.publish(msg)
        
        # Optional: Log the commands being published
        # self.get_logger().info(f"Publishing cmd_vel: linear.x={msg.linear.x:.3f}, angular.z={msg.angular.z:.3f}")

    def load_and_publish_map(self):
        """Load image and convert to occupancy grid"""
        img_path = self.get_parameter('image_path').get_parameter_value().string_value

        if not os.path.exists(img_path):
            self.get_logger().error(f"Image file not found: {img_path}")
            return

        # Read grayscale image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.get_logger().error("Failed to load image")
            return

        self.get_logger().info(f"Loaded traversability map {img.shape} from {img_path}")

        # Convert image → occupancy grid
        occ_grid = np.zeros_like(img, dtype=np.int8)
        occ_grid[img < 127] = 100     # Occupied (dark pixels)
        occ_grid[img >= 127] = 0      # Free (light pixels)

        # Create occupancy grid message
        self.grid_msg = OccupancyGrid()
        self.grid_msg.header = Header()
        self.grid_msg.header.frame_id = "map"
        self.grid_msg.info.resolution = 0.01  # 1cm per pixel
        self.grid_msg.info.width = img.shape[1]
        self.grid_msg.info.height = img.shape[0]
        self.grid_msg.info.origin.position.x = -img.shape[1]/200.0
        self.grid_msg.info.origin.position.y = -img.shape[0]/200.0
        self.grid_msg.info.origin.position.z = 0.0
        self.grid_msg.info.origin.orientation.w = 1.0

        # Flatten image row-major
        self.grid_msg.data = occ_grid.flatten().tolist()

        # Publisher for map with correct QoS
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
        
        map_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        
        self.map_pub = self.create_publisher(OccupancyGrid, 'map', map_qos)
        
        # Timer to republish map periodically
        self.create_timer(1.0, self.publish_map)

    def publish_map(self):
        """Publish the occupancy grid map"""
        self.grid_msg.header.stamp = self.get_clock().now().to_msg()
        self.map_pub.publish(self.grid_msg)

    def set_initial_pose_once(self):
        """Set the initial pose estimate for AMCL (called once)"""
        
        initial_pose = PoseWithCovarianceStamped()
        initial_pose.header.frame_id = "map"
        initial_pose.header.stamp = self.get_clock().now().to_msg()
        
        # Set initial position (adjust based on your robot's starting position)
        initial_pose.pose.pose.position.x = self.current_pose.pose.position.x
        initial_pose.pose.pose.position.y = self.current_pose.pose.position.y
        initial_pose.pose.pose.position.z = self.current_pose.pose.position.z
        initial_pose.pose.pose.orientation.w = self.current_pose.pose.orientation.w
        
        # Set covariance (uncertainty)
        initial_pose.pose.covariance = [0.25] * 36  # Reasonable uncertainty
        
        self.initial_pose_pub.publish(initial_pose)
        self.get_logger().info("Published initial pose estimate")

    def pose_callback(self, msg: PoseStamped):
        """Handle robot pose updates"""
        self.current_pose = msg
        
        # Create and publish visualization marker
        marker = Marker()
        marker.header.frame_id = msg.header.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "robot"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Position and orientation
        marker.pose = msg.pose

        # Scale (size of the sphere)
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        # Color (red)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.lifetime.sec = 0  # Forever

        self.marker_pub.publish(marker)

    def navigate_to_pose(self, x, y, yaw=0.0, frame_id="map"):
        """Navigate to a specific pose using Nav2"""
        if not self.nav_client.server_is_ready():
            self.get_logger().error("Nav2 server not available!")
            return False

        # Create navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = frame_id
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        
        # Set target position
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0
        
        # Set target orientation (from yaw angle)
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = np.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = np.cos(yaw / 2.0)

        self.get_logger().info(f"Navigating to: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")

        # Send goal
        future = self.nav_client.send_goal_async(
            goal_msg, 
            feedback_callback=self.navigation_feedback_callback
        )
        future.add_done_callback(self.navigation_goal_response_callback)
        
        return True

    def navigation_goal_response_callback(self, future):
        """Handle goal response from Nav2"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Navigation goal rejected!')
            return

        self.get_logger().info('Navigation goal accepted')
        self.navigation_goal_handle = goal_handle
        
        # Wait for result
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.navigation_result_callback)

    def navigation_feedback_callback(self, feedback_msg):
        """Handle navigation feedback"""
        feedback = feedback_msg.feedback
        # You can access navigation progress here if needed
        # self.get_logger().info(f"Navigation feedback: {feedback}")

    def navigation_result_callback(self, future):
        """Handle navigation result"""
        result = future.result().result
        if result:
            self.get_logger().info('Navigation completed successfully!')
        else:
            self.get_logger().error('Navigation failed!')
        
        self.navigation_goal_handle = None

    def cancel_navigation(self):
        """Cancel current navigation goal"""
        if self.navigation_goal_handle is not None:
            self.get_logger().info("Cancelling navigation...")
            cancel_future = self.navigation_goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self.cancel_done_callback)

    def cancel_done_callback(self, future):
        """Handle navigation cancellation result"""
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info('Navigation goal successfully canceled')
        else:
            self.get_logger().error('Failed to cancel navigation goal')

    def navigate_to_example_goal(self):
        """Example function to demonstrate navigation"""
        # Check if we have localization working
        if self.current_pose is None:
            self.get_logger().warning("No robot pose available yet. Waiting for localization...")
            # Try again in 2 seconds
            self.create_timer(2.0, self.navigate_to_example_goal)
            return
        
        self.get_logger().info("Control mode: Nav2 Direct - Nav2 controls robot directly")
        
        # Navigate to point (-0.55, 0.2) with 45-degree orientation
        # self.navigate_to_pose(0.78, -2.62, np.pi/4)
        self.navigate_to_pose(-1.63, 0.48, np.pi/4)

    def is_navigating(self):
        """Check if robot is currently navigating"""
        return self.navigation_goal_handle is not None


def main(args=None):
    rclpy.init(args=args)
    node = Nav2NavigationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
        if node.is_navigating():
            node.cancel_navigation()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
