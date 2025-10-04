#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import cv2
import numpy as np

from sensor_msgs.msg import Image
from std_msgs.msg import Int32, Int32MultiArray
from cv_bridge import CvBridge


class CupHoleInspector(Node):
    """
    Subscribes to /circle_detector/output_cropped (a horizontal strip of 200x200 crops),
    classifies the 4 main holes in EACH tile as FILLED/EMPTY, and publishes:

      - /cup_hole_inspector/output           (Image): annotated strip (green=FILLED, red=EMPTY)
      - /cup_hole_inspector/status           (Int32MultiArray): 0/1 per hole per tile (1=FILLED, 0=EMPTY)
      - /cup_hole_inspector/filled_count     (Int32): total filled holes in the strip

    Tuning:
      - hole_offsets_px: list[int] flattened [dx1,dy1,...,dx4,dy4] from tile center (px)
      - hole_radius_px: int radius of each cup hole mask (px)
      - red_hsv_low1/high1 & red_hsv_low2/high2: HSV ranges for red
      - min_red_fraction: fraction of red pixels in hole mask to consider it EMPTY
      - sat_threshold: minimum mean saturation in mask for red to count (helps ignore glare)
      - tile_width/height: expected crop size (default 200x200)
    """

    def __init__(self):
        super().__init__("cup_hole_inspector")
        self.bridge = CvBridge()

        # Topics
        self.declare_parameter("cropped_topic", "/circle_detector/output_cropped")
        cropped_topic = self.get_parameter("cropped_topic").value

        self.sub = self.create_subscription(Image, cropped_topic, self.on_image, 10)
        self.pub_annot = self.create_publisher(Image, "/cup_hole_inspector/output", 1)
        self.pub_status = self.create_publisher(Int32MultiArray, "/cup_hole_inspector/status", 10)
        self.pub_count = self.create_publisher(Int32, "/cup_hole_inspector/filled_count", 10)

        # Tile & hole geometry
        self.declare_parameter("tile_width", 200)
        self.declare_parameter("tile_height", 200)
        self.declare_parameter("hole_radius_px", 18)

        # Default hole offsets (px) from the tile center (cx,cy): TL, TR, BL, BR
        # Adjust these if your layout shifts a bit. Positive x→right, positive y→down.
        default_offsets = [-28, -32,   28, -32,   -28, 32,   28, 32]
        # default_offsets = [-42, -45, 42, -45, -42, 45, 42, 45]
        
        self.declare_parameter("hole_offsets_px", default_offsets)

        # Red detection (HSV, OpenCV H:0..179)
        self.declare_parameter("red_hsv_low1",  [0,   80, 50])
        self.declare_parameter("red_hsv_high1", [10, 255, 255])
        self.declare_parameter("red_hsv_low2",  [170, 80, 50])
        self.declare_parameter("red_hsv_high2", [179, 255, 255])

        # Decision thresholds
        self.declare_parameter("min_red_fraction", 0.10)  # ≥10% red in the hole region → EMPTY
        self.declare_parameter("sat_threshold",    60.0)  # need some saturation to count as red

        self.get_logger().info(f"Listening to crops on: {cropped_topic}")

    # ---------- helpers ----------
    def _hole_centers(self, W: int, H: int):
        """Compute hole centers from tile center and configured offsets."""
        cx, cy = W // 2, H // 2
        offsets = list(self.get_parameter("hole_offsets_px").value)
        if len(offsets) % 2 != 0:
            self.get_logger().warn("hole_offsets_px must have even length; using defaults for safety.")
            offsets = [-28, -32,   28, -32,   -28, 32,   28, 32]
        centers = []
        for i in range(0, len(offsets), 2):
            dx, dy = int(offsets[i]), int(offsets[i + 1])
            centers.append((cx + dx, cy + dy))
        return centers

    def _classify_hole(self, tile_bgr: np.ndarray, center: tuple, radius_px: int):
        """Return (is_filled: bool, red_fraction: float)."""
        H, W = tile_bgr.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(mask, center, radius_px, 255, -1)

        # Extract masked region in HSV
        hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)

        low1  = np.array(self.get_parameter("red_hsv_low1").value,  dtype=np.uint8)
        high1 = np.array(self.get_parameter("red_hsv_high1").value, dtype=np.uint8)
        low2  = np.array(self.get_parameter("red_hsv_low2").value,  dtype=np.uint8)
        high2 = np.array(self.get_parameter("red_hsv_high2").value, dtype=np.uint8)

        red1 = cv2.inRange(hsv, low1, high1)
        red2 = cv2.inRange(hsv, low2, high2)
        red_mask = cv2.bitwise_or(red1, red2)

        # Limit to the hole region
        red_in_hole = cv2.bitwise_and(red_mask, red_mask, mask=mask)
        hole_area = int(np.count_nonzero(mask))
        red_area = int(np.count_nonzero(red_in_hole))
        red_fraction = (red_area / hole_area) if hole_area > 0 else 0.0

        # Add a saturation guard (ignore noisy red with very low saturation)
        sat = hsv[:, :, 1]
        sat_mean = float(cv2.mean(sat, mask=mask)[0])

        min_red_fraction = float(self.get_parameter("min_red_fraction").value)
        sat_threshold = float(self.get_parameter("sat_threshold").value)

        is_empty = (red_fraction >= min_red_fraction) and (sat_mean >= sat_threshold)
        is_filled = not is_empty
        return is_filled, red_fraction, sat_mean

    # ---------- callback ----------
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

        # Number of tiles in the horizontal strip
        num_tiles = max(1, W // tile_w)

        annotated = np.zeros_like(strip)
        status_values = []  # 1=FILLED, 0=EMPTY for each hole, per tile in order

        total_filled = 0
        hole_centers_rel = None  # computed per tile (same for all tiles)

        for i in range(num_tiles):
            x0 = i * tile_w
            x1 = x0 + tile_w
            tile = strip[0:tile_h, x0:x1].copy()

            # Lazy compute centers for this tile size
            if hole_centers_rel is None:
                hole_centers_rel = self._hole_centers(tile_w, tile_h)

            # Annotate copy
            tile_annot = tile.copy()

            for idx, (cx, cy) in enumerate(hole_centers_rel, start=1):
                is_filled, red_frac, sat_mean = self._classify_hole(tile, (cx, cy), radius)

                # Draw visual cue (on annotated only)
                color = (0, 200, 0) if is_filled else (0, 0, 255)
                cv2.circle(tile_annot, (cx, cy), radius, color, 2)
                label = "FILLED" if is_filled else "EMPTY"
                cv2.putText(tile_annot, f"{label}",
                            (cx - 32, cy - radius - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                # Optional tiny dot at center
                cv2.circle(tile_annot, (cx, cy), 2, color, -1)

                status_values.append(1 if is_filled else 0)
                total_filled += int(is_filled)

                # Debug log per hole
                self.get_logger().info(
                    f"[tile {i}] hole {idx}: {label} (red_frac={red_frac:.2f}, sat={sat_mean:.1f})"
                )

            # Paste this annotated tile back into the output strip
            annotated[0:tile_h, x0:x1] = tile_annot

        # Publish annotated strip
        out_img = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out_img.header = msg.header
        self.pub_annot.publish(out_img)

        # Publish status vector and counts
        arr = Int32MultiArray()
        arr.data = status_values  # length = num_tiles * #holes (default 4)
        self.pub_status.publish(arr)
        self.pub_count.publish(Int32(data=total_filled))

        self.get_logger().info(f"[summary] tiles={num_tiles}, holes/tile=4, filled={total_filled}, empty={len(status_values)-total_filled}")


def main(args=None):
    rclpy.init(args=args)
    node = CupHoleInspector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
