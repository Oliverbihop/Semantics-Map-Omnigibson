# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from tf2_ros import TransformBroadcaster
# from geometry_msgs.msg import TransformStamped
# import math

# class TFPublisher(Node):
#     def __init__(self):
#         super().__init__('tf_publisher')
        
#         self.tf_broadcaster = TransformBroadcaster(self)
        
#         # Publish transforms at 50 Hz
#         self.timer = self.create_timer(0.02, self.publish_transforms)
        
#         # Robot position (you can make this dynamic)
#         self.x = 0.0
#         self.y = 0.0
#         self.theta = 0.0

#     def publish_transforms(self):
#         # Publish odom -> base_link transform
#         t = TransformStamped()
#         t.header.stamp = self.get_clock().now().to_msg()
#         t.header.frame_id = 'odom'
#         t.child_frame_id = 'base_link'
        
#         # Set position
#         t.transform.translation.x = self.x
#         t.transform.translation.y = self.y
#         t.transform.translation.z = 0.0
        
#         # Set orientation (quaternion from yaw)
#         t.transform.rotation.x = 0.0
#         t.transform.rotation.y = 0.0
#         t.transform.rotation.z = math.sin(self.theta / 2.0)
#         t.transform.rotation.w = math.cos(self.theta / 2.0)
        
#         self.tf_broadcaster.sendTransform(t)
        
#         # Publish map -> odom transform (static for now)
#         t2 = TransformStamped()
#         t2.header.stamp = self.get_clock().now().to_msg()
#         t2.header.frame_id = 'map'
#         t2.child_frame_id = 'odom'
        
#         t2.transform.translation.x = 0.0
#         t2.transform.translation.y = 0.0
#         t2.transform.translation.z = 0.0
#         t2.transform.rotation.x = 0.0
#         t2.transform.rotation.y = 0.0
#         t2.transform.rotation.z = 0.0
#         t2.transform.rotation.w = 1.0
        
#         self.tf_broadcaster.sendTransform(t2)

# def main():
#     rclpy.init()
#     node = TFPublisher()
    
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()


import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

class PoseToTF(Node):
    def __init__(self):
        super().__init__('pose_to_tf')
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_broadcaster = StaticTransformBroadcaster(self)

        # Publish a static transform: map → odom
        static_tf = TransformStamped()
        static_tf.header.stamp = self.get_clock().now().to_msg()
        static_tf.header.frame_id = "map"
        static_tf.child_frame_id = "odom"
        static_tf.transform.translation.x = 0.0
        static_tf.transform.translation.y = 0.0
        static_tf.transform.translation.z = 0.0
        static_tf.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(static_tf)

        # Subscribe to your /robot_pose
        self.sub = self.create_subscription(
            PoseStamped, '/robot_pose', self.pose_callback, 10)

    def pose_callback(self, msg):
        
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "odom"          # now relative to odom
        t.child_frame_id = "base_link"
        t.transform.translation.x = msg.pose.position.x + 0.0
        t.transform.translation.y = msg.pose.position.y + 0.0
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation

        print(t.transform)

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = PoseToTF()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()


