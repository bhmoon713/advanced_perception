#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped

# T: base_link -> wrist_rgbd_camera_depth_optical_frame (from your tf2_echo)
T = np.array([
    [ 1.000,  0.000, -0.000, -0.238],
    [ 0.000, -0.866,  0.500, -0.300],
    [-0.000, -0.500, -0.866,  0.200],
    [ 0.000,  0.000,  0.000,  1.000],
], dtype=float)

T_INV = np.linalg.inv(T)  # camera -> base_link

def transform_point_inv(p_xyz):
    """camera (x,y,z) -> base_link (x,y,z) using inverse matrix."""
    pt = np.array([p_xyz[0], p_xyz[1], p_xyz[2], 1.0], dtype=float)
    out = T_INV @ pt
    return out[0], out[1], out[2]

class InverseTransformNode(Node):
    def __init__(self):
        super().__init__('inverse_transform_node')

        # params for topic names (override in launch if you like)
        self.declare_parameter('in_topic', '/point_in_camera')
        self.declare_parameter('out_topic', '/point_in_base_link')

        in_topic = self.get_parameter('in_topic').get_parameter_value().string_value
        out_topic = self.get_parameter('out_topic').get_parameter_value().string_value

        self.pub_ = self.create_publisher(PointStamped, out_topic, 10)
        self.sub_ = self.create_subscription(PointStamped, in_topic, self.cb_point, 10)

        self.get_logger().info(
            f'Listening on "{in_topic}" (frame_id=camera), publishing to "{out_topic}" (frame_id=base_link)'
        )

    def cb_point(self, msg: PointStamped):
        # Expect msg.header.frame_id == 'wrist_rgbd_camera_depth_optical_frame'
        x_b, y_b, z_b = transform_point_inv((msg.point.x, msg.point.y, msg.point.z))

        out = PointStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'base_link'
        out.point.x = float(x_b)
        out.point.y = float(y_b)
        out.point.z = float(z_b)
        self.pub_.publish(out)

        self.get_logger().info(
            f'camera({msg.point.x:.3f}, {msg.point.y:.3f}, {msg.point.z:.3f})'
            f'  ->  base_link({x_b:.3f}, {y_b:.3f}, {z_b:.3f})'
        )

def main():
    rclpy.init()
    node = InverseTransformNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
