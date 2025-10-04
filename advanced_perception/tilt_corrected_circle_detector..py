#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from geometry_msgs.msg import PointStamped
from std_msgs.msg import Int32, Float32MultiArray

from custom_msgs.msg import InferenceResult


def euler_to_R(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """R = Rz(yaw) * Ry(pitch) * Rx(roll) in camera optical frame (x-right, y-down, z-forward)."""
    cx, cy, cz = np.cos(roll_rad), np.cos(pitch_rad), np.cos(yaw_rad)
    sx, sy, sz = np.sin(roll_rad), np.sin(pitch_rad), np.sin(yaw_rad)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx


class CircleDetector(Node):
    def __init__(self):
        super().__init__('circle_detector')
        self.bridge = CvBridge()

        # --- Inputs ---
        self.declare_parameter('camera_topic', '/wrist_rgbd_depth_sensor/image_raw')
        cam_topic = self.get_parameter('camera_topic').value
        default_info_topic = cam_topic.replace('/image_raw', '/camera_info')
        self.declare_parameter('camera_info_topic', default_info_topic)
        cam_info_topic = self.get_parameter('camera_info_topic').value

        # --- Hough params ---
        self.declare_parameter('dp', 1.2)
        self.declare_parameter('minDist', 40.0)
        self.declare_parameter('param1', 100.0)
        self.declare_parameter('param2', 30.0)
        self.declare_parameter('minRadius', 20)
        self.declare_parameter('maxRadius', 30)

        # --- Rectification (defaults match your current setting) ---
        self.declare_parameter('use_rectification', True)
        self.declare_parameter('rect_roll_deg', 30.0)
        self.declare_parameter('rect_pitch_deg', 0.0)
        self.declare_parameter('rect_yaw_deg', 0.0)

        # --- Shift (translation) ---
        self.declare_parameter('shift_x_px', 0)        # +right / -left
        self.declare_parameter('shift_y_px', 300)      # +down  / -up
        self.declare_parameter('shift_border_mode', 'constant')  # constant|replicate|reflect
        self.declare_parameter('shift_fill_bgr', [0, 0, 0])

        # --- Auto-expand canvas to avoid cropping ---
        self.declare_parameter('expand_canvas', True)
        self.declare_parameter('expand_pad_px', 0)

        # --- Optional fixed-size output window + pan ---
        self.declare_parameter('output_mode', 'expanded')  # 'expanded' or 'fixed'
        self.declare_parameter('fixed_width', 0)
        self.declare_parameter('fixed_height', 0)
        self.declare_parameter('pan_x_px', 0)
        self.declare_parameter('pan_y_px', 0)

        # --- Crop mode (new) ---
        self.declare_parameter('crop_mode', 'none')  # 'none' | 'upper_half' | 'lower_half'

        # Subs
        self.sub_img = self.create_subscription(Image, cam_topic, self.camera_cb, 10)
        self.sub_info = self.create_subscription(CameraInfo, cam_info_topic, self.camera_info_cb, 10)

        # Pubs
        self.pub_tilt_corrected = self.create_publisher(Image, '/tilt_corrected_output', 1)  # processed image (used for detection)
        self.pub_img_annot = self.create_publisher(Image, '/circle_detector/output', 1)      # annotated version of the same
        self.pub_result = self.create_publisher(InferenceResult, '/circle_inference_result', 10)
        self.pub_center = self.create_publisher(PointStamped, '/circle_center', 10)
        self.pub_count  = self.create_publisher(Int32,  '/circle_count', 10)
        self.pub_radii  = self.create_publisher(Float32MultiArray, '/circle_radii_px', 10)

        # Rectification state
        self._maps_ready = False
        self._map1 = self._map2 = None
        self._rect_size = None
        self._K = None
        self._D = None

        self.get_logger().info(f"Subscribed to image: {cam_topic}")
        self.get_logger().info(f"Subscribed to camera info: {cam_info_topic}")

    # ---------------- helpers ----------------
    def _get_hough_params(self):
        return (
            float(self.get_parameter('dp').value),
            float(self.get_parameter('minDist').value),
            float(self.get_parameter('param1').value),
            float(self.get_parameter('param2').value),
            int(self.get_parameter('minRadius').value),
            int(self.get_parameter('maxRadius').value),
        )

    def _get_rect_params_rad(self):
        roll  = np.deg2rad(float(self.get_parameter('rect_roll_deg').value))
        pitch = np.deg2rad(float(self.get_parameter('rect_pitch_deg').value))
        yaw   = np.deg2rad(float(self.get_parameter('rect_yaw_deg').value))
        return roll, pitch, yaw

    def _border_mode_cv(self):
        mode = str(self.get_parameter('shift_border_mode').value).lower()
        if mode == 'replicate':
            return cv2.BORDER_REPLICATE, None
        if mode == 'reflect':
            return cv2.BORDER_REFLECT, None
        fill = self.get_parameter('shift_fill_bgr').value
        if not isinstance(fill, (list, tuple)) or len(fill) != 3:
            fill = [0, 0, 0]
        return cv2.BORDER_CONSTANT, (int(fill[0]), int(fill[1]), int(fill[2]))

    # ---------------- camera info ----------------
    def camera_info_cb(self, info: CameraInfo):
        self._K = np.array(info.k, dtype=float).reshape(3, 3)
        self._D = np.array(info.d, dtype=float).reshape(-1,)
        self._rect_size = (info.width, info.height)

        if not bool(self.get_parameter('use_rectification').value):
            self._maps_ready = False
            self._map1 = self._map2 = None
            self.get_logger().info("Rectification disabled (use_rectification:=false).")
            return

        roll, pitch, yaw = self._get_rect_params_rad()
        R_rect = euler_to_R(roll, pitch, yaw)
        P = self._K.copy()

        try:
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                cameraMatrix=self._K, distCoeffs=self._D,
                R=R_rect, newCameraMatrix=P,
                size=self._rect_size, m1type=cv2.CV_16SC2
            )
            self._maps_ready = True
            self.get_logger().info(
                f"Rectify map ready (size={self._rect_size}, roll={np.rad2deg(roll):.1f}°, "
                f"pitch={np.rad2deg(pitch):.1f}°, yaw={np.rad2deg(yaw):.1f}°)."
            )
        except Exception as e:
            self._maps_ready = False
            self.get_logger().warning(f"Rectify map build failed: {e}")

    # ---------------- image ----------------
    def camera_cb(self, msg: Image):
        # decode
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 1) Rectify
        if bool(self.get_parameter('use_rectification').value) and self._maps_ready:
            rectified = cv2.remap(img, self._map1, self._map2, interpolation=cv2.INTER_LINEAR)
        else:
            rectified = img

        # 2) Shift with auto-expand (no crop)
        dx = int(self.get_parameter('shift_x_px').value)
        dy = int(self.get_parameter('shift_y_px').value)
        expand = bool(self.get_parameter('expand_canvas').value)
        pad = int(self.get_parameter('expand_pad_px').value)
        borderMode, borderValue = self._border_mode_cv()

        h, w = rectified.shape[:2]
        if dx == 0 and dy == 0 and not expand:
            expanded = rectified
        else:
            if expand:
                min_x = min(0, dx);  max_x = max(w, w + dx)
                min_y = min(0, dy);  max_y = max(h, h + dy)
                new_w = (max_x - min_x) + 2 * pad
                new_h = (max_y - min_y) + 2 * pad
                off_x = dx - min_x + pad
                off_y = dy - min_y + pad
                M_expand = np.float32([[1, 0, off_x], [0, 1, off_y]])
                expanded = cv2.warpAffine(rectified, M_expand, (int(new_w), int(new_h)),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=borderMode,
                                          borderValue=borderValue if borderValue is not None else 0)
            else:
                M_expand = np.float32([[1, 0, dx], [0, 1, dy]])
                expanded = cv2.warpAffine(rectified, M_expand, (w, h),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=borderMode,
                                          borderValue=borderValue if borderValue is not None else 0)

        # 3) Optional fixed-size output window + pan
        output_mode = str(self.get_parameter('output_mode').value)
        if output_mode == 'fixed':
            target_w = int(self.get_parameter('fixed_width').value)  or expanded.shape[1]
            target_h = int(self.get_parameter('fixed_height').value) or expanded.shape[0]
            pan_x = int(self.get_parameter('pan_x_px').value)
            pan_y = int(self.get_parameter('pan_y_px').value)
            cx = expanded.shape[1] // 2 + pan_x
            cy = expanded.shape[0] // 2 + pan_y
            M_pan = np.float32([[1, 0, -(cx - target_w // 2)],
                                [0, 1, -(cy - target_h // 2)]])
            out_img = cv2.warpAffine(expanded, M_pan, (int(target_w), int(target_h)),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=borderMode,
                                     borderValue=borderValue if borderValue is not None else 0)
        else:
            out_img = expanded

        # 4) Crop mode (upper/lower half)
        H_all, W_all = out_img.shape[:2]
        crop_mode = str(self.get_parameter('crop_mode').value).lower()
        if crop_mode == 'upper_half':
            proc = out_img[0:H_all // 2, 0:W_all]
        elif crop_mode == 'lower_half':
            proc = out_img[H_all // 2:H_all, 0:W_all]
        else:
            proc = out_img

        # Publish processed image used for detection
        tilt_msg = self.bridge.cv2_to_imgmsg(proc, encoding='bgr8')
        tilt_msg.header = msg.header
        self.pub_tilt_corrected.publish(tilt_msg)

        # 5) Detect circles on 'proc'
        H, W = proc.shape[:2]
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)

        dp, minDist, param1, param2, minRadius, maxRadius = self._get_hough_params()
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT,
            dp=dp, minDist=minDist, param1=param1, param2=param2,
            minRadius=minRadius, maxRadius=maxRadius
        )

        annotated = proc.copy()
        radii_list = []

        if circles is not None:
            circles = np.uint16(np.around(circles[0]))
            self.pub_count.publish(Int32(data=int(len(circles))))
            for (x, y, r) in circles:
                radii_list.append(float(r))
                cv2.circle(annotated, (x, y), r, (0, 255, 0), 2)
                cv2.circle(annotated, (x, y), 2, (0, 0, 255), 3)
                cv2.putText(annotated, f"r={r}px", (x + 6, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                left   = max(0, int(x - r))
                top    = max(0, int(y - r))
                right  = min(W - 1, int(x + r))
                bottom = min(H - 1, int(y + r))

                inf = InferenceResult()
                inf.class_name = "circle"
                inf.left, inf.top, inf.right, inf.bottom = left, top, right, bottom
                inf.box_width, inf.box_height = right - left, bottom - top
                inf.x, inf.y = float(x), float(y)
                self.pub_result.publish(inf)

                pt = PointStamped()
                pt.header = msg.header
                pt.point.x, pt.point.y, pt.point.z = float(x), float(y), 0.0
                self.pub_center.publish(pt)

                self.get_logger().info(f"[proc] center=({x},{y}) r={r}px")
        else:
            self.pub_count.publish(Int32(data=0))
            self.get_logger().info("[proc] no circles detected")

        # annotated image out
        self.pub_radii.publish(Float32MultiArray(data=radii_list))
        self.pub_img_annot.publish(self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = CircleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
