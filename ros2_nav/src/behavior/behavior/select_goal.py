import json
import numpy as np
import rclpy
import sys
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

def load_analysis_data(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def get_object_by_id(objects, obj_id):
    for obj in objects:
        if obj['id'] == obj_id:
            return obj
    return None

def get_approach_point(bbox_points, offset_dist=0.5):
    points = np.array(bbox_points)
    bottom_face = points[:4]
    
    center_x = np.mean(bottom_face[:, 0])
    center_y = np.mean(bottom_face[:, 1])
    min_y = np.min(bottom_face[:, 1])
    
    return center_x, min_y - offset_dist

class GoalPosePublisher(Node):
    def __init__(self):
        super().__init__('goal_pose_publisher')
        self.publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
    def publish_goal(self, x, y):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        
        self.publisher.publish(msg)
        self.get_logger().info(f'Published goal: x={x:.3f}, y={y:.3f}')

def main():

    
    if len(sys.argv) < 2:
        print("Usage: ros2 run behavior goal_publisher <object_id>")
        return
    obj_id = int(sys.argv[1])
    rclpy.init()
    data = load_analysis_data('./analysis.json')

    obj = get_object_by_id(data['objects'], obj_id)
    
    if obj is None:
        print(f"Object ID {obj_id} not found.")
        return
        
    goal_x, goal_y = get_approach_point(obj['bbox']['points'], offset_dist=0.5)
    
    node = GoalPosePublisher()
    node.publish_goal(goal_x, goal_y)
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
