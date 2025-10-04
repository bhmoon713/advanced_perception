#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from geometry_msgs.msg import PointStamped
from std_msgs.msg import Int32, Float32MultiArray

from custom_msgs.msg import InferenceResult


class CircleDetector(Node):
    def __init__(self):
        super().__init__('circle_detector')
        self.bridge = CvBridge()

        # Parameters
        self.declare_parameter('camera_topic', '/wrist_rgbd_depth_sensor/image_raw')
        self.declare_parameter('dp', 1.2)          # inverse ratio of accumulator to image resolution
        self.declare_parameter('minDist', 40.0)    # min center distance between circles (px)
        self.declare_parameter('param1', 100.0)    # Canny high threshold
        self.declare_parameter('param2', 30.0)     # accumulator threshold (smaller -> more circles)
        self.declare_parameter('minRadius', 20)    # px
        self.declare_parameter('maxRadius', 30)    # px

        cam_topic = self.get_parameter('camera_topic').value
        self.sub = self.create_subscription(Image, cam_topic, self.camera_cb, 10)

        # Publishers (existing)
        self.pub_img = self.create_publisher(Image, '/circle_detector/output_1', 1)
        self.pub_result = self.create_publisher(InferenceResult, '/circle_inference_result_1', 10)

        # NEW: circle info publishers
        self.pub_center = self.create_publisher(PointStamped, '/circle_center_1', 10)
        self.pub_count  = self.create_publisher(Int32, '/circle_count_1', 10)
        self.pub_radii  = self.create_publisher(Float32MultiArray, '/circle_radii_px_1', 10)

        self.get_logger().info(f"Subscribed to: {cam_topic}")

    def _get_params(self):
        return (
            float(self.get_parameter('dp').value),
            float(self.get_parameter('minDist').value),
            float(self.get_parameter('param1').value),
            float(self.get_parameter('param2').value),
            int(self.get_parameter('minRadius').value),
            int(self.get_parameter('maxRadius').value),
        )

    def camera_cb(self, msg: Image):
        # If this is a color image, 'bgr8' is fine. If it's mono/depth, adjust encoding accordingly.
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            # Fallback for mono images (grayscale cameras)
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)

        dp, minDist, param1, param2, minRadius, maxRadius = self._get_params()
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius
        )

        annotated = img.copy()
        radii_list = []

        if circles is not None:
            circles = np.uint16(np.around(circles[0]))

            # Publish circle count for this frame
            self.pub_count.publish(Int32(data=int(len(circles))))

            for (x, y, r) in circles:
                radii_list.append(float(r))

                # Draw overlay
                cv2.circle(annotated, (x, y), r, (0, 255, 0), 2)   # circle outline
                cv2.circle(annotated, (x, y), 2, (0, 0, 255), 3)   # center point
                cv2.putText(annotated, f"r={r}px", (x + 6, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                # Build and publish InferenceResult (kept from your original)
                inf = InferenceResult()
                inf.class_name = "circle"

                left   = max(0, int(x - r))
                top    = max(0, int(y - r))
                right  = min(w - 1, int(x + r))
                bottom = min(h - 1, int(y + r))

                inf.left = left
                inf.top = top
                inf.right = right
                inf.bottom = bottom
                inf.box_width = right - left
                inf.box_height = bottom - top

                inf.x = float(x)             # center x (px)
                inf.y = float(y)             # center y (px)
                # radius not in message; publish separately below

                self.pub_result.publish(inf)

                # Publish center as PointStamped (image frame)
                pt = PointStamped()
                pt.header = msg.header
                # center in pixel space; z=0 because this is 2D pixel coordinate
                pt.point.x = float(x)
                pt.point.y = float(y)
                pt.point.z = 0.0
                self.pub_center.publish(pt)

                # Log info
                self.get_logger().info(f"Circle: center=({x}, {y}), radius={r}px")
        else:
            self.pub_count.publish(Int32(data=0))
            self.get_logger().info("No circles detected.")

        # Publish all radii for this frame (even if empty)
        rmsg = Float32MultiArray()
        rmsg.data = radii_list
        self.pub_radii.publish(rmsg)

        # Publish annotated image
        self.pub_img.publish(self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = CircleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
