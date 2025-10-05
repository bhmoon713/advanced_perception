#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from math import atan2, sin, cos, radians

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32, Int32MultiArray
from cv_bridge import CvBridge


class CupHoleInspector(Node):
    """
    Subscribes to /circle_detector/output_cropped (a horizontal strip of 200x200 crops),
    classifies the 4 main holes in EACH tile as FILLED/EMPTY, and publishes:

      - /cup_hole_inspector/output           (Image): annotated strip (green=FILLED, red=EMPTY)
      - /cup_hole_inspector/status           (Int32MultiArray): 0/1 per hole per tile (1=FILLED, 0=EMPTY)
      - /cup_hole_inspector/filled_count     (Int32): total filled holes in the strip

    NEW:
      - Subscribes to robot pose (/barista_1/odom by default), computes yaw, and rotates the
        configured hole offsets by that yaw before testing each hole. This adapts to robot rotation.

    Tuning:
      - hole_offsets_px: list[int] flattened [dx1,dy1,...,dx4,dy4] from tile center (px)
      - hole_radius_px: int radius of each cup hole mask (px)
      - red_hsv_*: HSV ranges for red (for EMPTY detection)
      - min_red_fraction: fraction of red pixels in hole mask to consider it EMPTY
      - sat_threshold: minimum mean saturation in mask for red to count
      - tile_width/height: expected crop size (default 200x200)

    Rotation params:
      - pose_topic (default '/barista_1/odom')
      - use_rotation (bool, default True)
      - yaw_sign (int, default -1)     # +1 or -1 depending on your camera/world convention
      - yaw_offset_deg (float, default 0.0)  # add small calibration bias if needed
    """

    def __init__(self):
        super().__init__("cup_hole_inspector")
        self.bridge = CvBridge()

        # --- Image topic(s)
        self.declare_parameter("cropped_topic", "/circle_detector/output_cropped")
        cropped_topic = self.get_parameter("cropped_topic").value

        self.sub = self.create_subscription(Image, cropped_topic, self.on_image, 10)
        self.pub_annot = self.create_publisher(Image, "/cup_hole_inspector/output", 1)
        self.pub_status = self.create_publisher(Int32MultiArray, "/cup_hole_inspector/status", 10)
        self.pub_count = self.create_publisher(Int32, "/cup_hole_inspector/filled_count", 10)

        # --- Tile & hole geometry
        self.declare_parameter("tile_width", 200)
        self.declare_parameter("tile_height", 200)
        self.declare_parameter("hole_radius_px", 18)

        # Default hole offsets (px) from the tile center (cx,cy): TL, TR, BL, BR
        default_offsets = [-28, -32,   28, -32,   -28, 32,   28, 32]
        self.declare_parameter("hole_offsets_px", default_offsets)

        # --- Red detection (HSV, OpenCV H:0..179)
        self.declare_parameter("red_hsv_low1",  [0,   80, 50])
        self.declare_parameter("red_hsv_high1", [10, 255, 255])
        self.declare_parameter("red_hsv_low2",  [170, 80, 50])
        self.declare_parameter("red_hsv_high2", [179, 255, 255])

        # --- Decision thresholds
        self.declare_parameter("min_red_fraction", 0.10)  # ≥10% red → EMPTY
        self.declare_parameter("sat_threshold",    60.0)

        # --- Rotation / pose inputs
        self.declare_parameter("pose_topic", "/barista_1/odom")
        self.declare_parameter("use_rotation", True)
        self.declare_parameter("yaw_sign", 1)          # scene appears to rotate opposite robot
        self.declare_parameter("yaw_offset_deg", 90.0)   # fine-tune bias, if camera not perfectly aligned

        self.yaw_rad = 0.0
        self._last_yaw_log_time = 0.0
        pose_topic = self.get_parameter("pose_topic").value
        self.pose_sub = self.create_subscription(Odometry, pose_topic, self.on_odom, 10)

        self.get_logger().info(f"Listening to crops on: {cropped_topic}")
        self.get_logger().info(f"Using pose topic for yaw: {pose_topic}")

    # ---------- pose/yaw ----------
    def on_odom(self, msg: Odometry):
        q = msg.pose.pose.orientation
        # yaw from quaternion
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = atan2(siny_cosp, cosy_cosp)  # radians, +CCW

        # Apply sign and fixed offset (deg -> rad)
        yaw_sign = int(self.get_parameter("yaw_sign").value)
        yaw_offset_deg = float(self.get_parameter("yaw_offset_deg").value)
        self.yaw_rad = yaw_sign * yaw + radians(yaw_offset_deg)

        # Throttled log
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_yaw_log_time > 1.5:  # every ~1.5s
            self._last_yaw_log_time = now
            self.get_logger().info(f"yaw(rad)={self.yaw_rad:.3f}  yaw(deg)={np.degrees(self.yaw_rad):.1f}")

    # ---------- helpers ----------
    def _hole_centers_rotated(self, W: int, H: int):
        """
        Compute rotated hole centers:
          1) start with configured offsets (x right, y down),
          2) convert to math coords (x right, y up) by dy_up = -dy,
          3) rotate by yaw_rad,
          4) convert back to image coords (y down).
        """
        cx, cy = W // 2, H // 2
        offsets = list(self.get_parameter("hole_offsets_px").value)
        if len(offsets) % 2 != 0:
            self.get_logger().warn("hole_offsets_px must have even length; using safe defaults.")
            offsets = [-28, -32, 28, -32, -28, 32, 28, 32]

        use_rotation = bool(self.get_parameter("use_rotation").value)
        theta = float(self.yaw_rad) if use_rotation else 0.0
        ct, st = cos(theta), sin(theta)

        centers = []
        for i in range(0, len(offsets), 2):
            dx = float(offsets[i + 0])
            dy = float(offsets[i + 1])

            # y-down (image) -> y-up (math)
            dx_u = dx
            dy_u = -dy

            # rotate in math coords
            rx_u =  dx_u * ct - dy_u * st
            ry_u =  dx_u * st + dy_u * ct

            # back to image coords (y-down)
            rx = int(round(rx_u))
            ry = int(round(-ry_u))

            centers.append((cx + rx, cy + ry))
        return centers

    def _classify_hole(self, tile_bgr: np.ndarray, center: tuple, radius_px: int):
        """Return (is_filled: bool, red_fraction: float, sat_mean: float)."""
        H, W = tile_bgr.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(mask, center, radius_px, 255, -1)

        hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)

        low1  = np.array(self.get_parameter("red_hsv_low1").value,  dtype=np.uint8)
        high1 = np.array(self.get_parameter("red_hsv_high1").value, dtype=np.uint8)
        low2  = np.array(self.get_parameter("red_hsv_low2").value,  dtype=np.uint8)
        high2 = np.array(self.get_parameter("red_hsv_high2").value, dtype=np.uint8)

        red1 = cv2.inRange(hsv, low1, high1)
        red2 = cv2.inRange(hsv, low2, high2)
        red_mask = cv2.bitwise_or(red1, red2)

        # Limit to hole region
        red_in_hole = cv2.bitwise_and(red_mask, red_mask, mask=mask)
        hole_area = int(np.count_nonzero(mask))
        red_area  = int(np.count_nonzero(red_in_hole))
        red_fraction = (red_area / hole_area) if hole_area > 0 else 0.0

        sat = hsv[:, :, 1]
        sat_mean = float(cv2.mean(sat, mask=mask)[0])

        min_red_fraction = float(self.get_parameter("min_red_fraction").value)
        sat_threshold    = float(self.get_parameter("sat_threshold").value)

        is_empty = (red_fraction >= min_red_fraction) and (sat_mean >= sat_threshold)
        is_filled = not is_empty
        return is_filled, red_fraction, sat_mean

    # ---------- image callback ----------
    def on_image(self, msg: Image):
        try:
            strip = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            strip = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            strip = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)

        tile_w = int(self.get_parameter("tile_width").value)
        tile_h = int(self.get_parameter("tile_height").value)
        radius = int(self.get_parameter("hole_radius_px").value)

        H, W = strip.shape[:2]
        if H < tile_h:
            self.get_logger().warn(f"Incoming crop strip height {H} < expected {tile_h}. Resizing to proceed.")
            strip = cv2.resize(strip, (max(W, tile_w), tile_h), interpolation=cv2.INTER_AREA)
            H, W = strip.shape[:2]

        num_tiles = max(1, W // tile_w)

        annotated = np.zeros_like(strip)
        status_values = []  # 1=FILLED, 0=EMPTY
        total_filled = 0

        # Pre-compute centers for this tile size with current yaw
        hole_centers_rel = self._hole_centers_rotated(tile_w, tile_h)

        for i in range(num_tiles):
            x0 = i * tile_w
            x1 = x0 + tile_w
            tile = strip[0:tile_h, x0:x1].copy()

            tile_annot = tile.copy()

            for idx, (cx, cy) in enumerate(hole_centers_rel, start=1):
                is_filled, red_frac, sat_mean = self._classify_hole(tile, (cx, cy), radius)

                color = (0, 200, 0) if is_filled else (0, 0, 255)
                label = "FILLED" if is_filled else "EMPTY"

                # annotate only on /cup_hole_inspector/output
                cv2.circle(tile_annot, (cx, cy), radius, color, 2)
                cv2.putText(tile_annot, label, (cx - 32, cy - radius - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                cv2.circle(tile_annot, (cx, cy), 2, color, -1)

                status_values.append(1 if is_filled else 0)
                total_filled += int(is_filled)

                self.get_logger().info(
                    f"[tile {i}] hole {idx}: {label} (red_frac={red_frac:.2f}, sat={sat_mean:.1f})"
                )

            annotated[0:tile_h, x0:x1] = tile_annot

        # Publish annotated strip
        out_img = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out_img.header = msg.header
        self.pub_annot.publish(out_img)

        # Publish status and counts
        arr = Int32MultiArray()
        arr.data = status_values
        self.pub_status.publish(arr)
        self.pub_count.publish(Int32(data=total_filled))

        self.get_logger().info(
            f"[summary] tiles={num_tiles}, holes/tile=4, filled={total_filled}, empty={len(status_values)-total_filled}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = CupHoleInspector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
