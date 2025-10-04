#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import cv2

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


def Rx(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c,-s],
                     [0, s, c]], dtype=float)

def Ry(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=float)

def Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s, 0],
                     [s, c, 0],
                     [0, 0, 1]], dtype=float)


class TopDownRectifier(Node):
    """
    Create a true top-down (bird’s-eye) view of a planar surface from a tilted pinhole camera.
    Uses CameraInfo intrinsics, camera height, and tilt angle relative to the plane.
    Publishes: /tilt_corrected_output (sensor_msgs/Image)
    """
    def __init__(self):
        super().__init__('topdown_rectifier')
        self.bridge = CvBridge()

        # ----- Parameters -----
        self.declare_parameter('camera_topic', '/wrist_rgbd_depth_sensor/image_raw')
        cam_topic = self.get_parameter('camera_topic').value

        default_info = cam_topic.replace('/image_raw', '/camera_info')
        self.declare_parameter('camera_info_topic', default_info)

        # Physical setup
        # Angle **to the plane** (e.g., 30° means grazing; 90° would be straight down)
        self.declare_parameter('tilt_axis', 'pitch')     # 'pitch' (about X) or 'roll' (about Y)
        self.declare_parameter('tilt_deg', 60.0)         # angle between optical axis and plane (deg)
        self.declare_parameter('height_m', 0.7)          # camera center to plane (meters)
        self.declare_parameter('yaw_deg', 0.0)           # rotate around world Z to align top-down axes

        # Output scaling
        self.declare_parameter('pixels_per_meter', 500.0)   # controls resolution of the top-down
        self.declare_parameter('margin_m', 0.02)            # extra border around auto-computed bounds (meters)

        # Optional: clamp output size (0 = auto)
        self.declare_parameter('max_out_width',  0)      # pixels
        self.declare_parameter('max_out_height', 0)      # pixels

        # ----- Subscriptions / Publications -----
        self.sub_img  = self.create_subscription(Image, cam_topic, self.on_image, 10)
        self.sub_info = self.create_subscription(CameraInfo,
                                                 self.get_parameter('camera_info_topic').value,
                                                 self.on_info, 10)
        self.pub_out  = self.create_publisher(Image, '/tilt_corrected_output', 1)

        self.K = None
        self.D = None
        self.size = None        # (w, h)

        self.get_logger().info(f"Subscribed image: {cam_topic}")
        self.get_logger().info(f"Subscribed camera info: {self.get_parameter('camera_info_topic').value}")

    # ---------------- CameraInfo ----------------
    def on_info(self, msg: CameraInfo):
        try:
            self.K = np.array(msg.k, dtype=float).reshape(3, 3)
            self.D = np.array(msg.d, dtype=float).reshape(-1)
            self.size = (int(msg.width), int(msg.height))
        except Exception as e:
            self.get_logger().error(f"CameraInfo parse error: {e}")

    # ---------------- Image ----------------
    def on_image(self, msg: Image):
        if self.K is None or self.size is None:
            self.get_logger().warn("Waiting for CameraInfo...")
            return

        # Decode image and undistort to ideal pinhole
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        w, h = self.size
        newK = self.K.copy()  # keep same intrinsics for convenience
        undistorted = cv2.undistort(img, self.K, self.D, None, newK)

        # ----- Build camera pose relative to the plane -----
        # World: plane is Z=0, Z up. Camera at (0,0,height) in world.
        height_m = float(self.get_parameter('height_m').value)
        tilt_deg = float(self.get_parameter('tilt_deg').value)
        yaw_deg  = float(self.get_parameter('yaw_deg').value)
        tilt_axis = str(self.get_parameter('tilt_axis').value).lower()

        # We start from a "top-down" pose (optical axis pointing straight down):
        # Camera-to-world rotation at top-down:
        #   X_c -> +X_w,  Y_c -> -Y_w,  Z_c -> -Z_w
        R_wc_top = np.array([[ 1,  0,  0],
                             [ 0, -1,  0],
                             [ 0,  0, -1]], dtype=float)

        # Convert the provided "angle to plane" to a "tilt from vertical".
        # If plane angle is 30°, the camera is 60° away from top-down.
        tilt_from_vertical_deg = max(0.0, 90.0 - tilt_deg)
        tilt = np.deg2rad(tilt_from_vertical_deg)
        yaw  = np.deg2rad(yaw_deg)

        if tilt_axis == 'pitch':
            R_wc = R_wc_top @ Rx(tilt) @ Rz(yaw)
        elif tilt_axis == 'roll':
            R_wc = R_wc_top @ Ry(tilt) @ Rz(yaw)
        else:
            # default to pitch if unknown
            R_wc = R_wc_top @ Rx(tilt) @ Rz(yaw)

        # World->camera rotation and translation
        R_cw = R_wc.T
        C_w = np.array([0.0, 0.0, height_m])   # camera center in world
        t = -R_cw @ C_w                        # translation (world origin in camera coords)

        # ----- Homography plane->image:  H = K [r1 r2 t] -----
        r1 = R_cw[:, 0].reshape(3, 1)
        r2 = R_cw[:, 1].reshape(3, 1)
        H_plane2img = newK @ np.hstack([r1, r2, t.reshape(3, 1)])
        H_img2plane = np.linalg.inv(H_plane2img)

        # ----- Project the 4 image corners to the plane to get output bounds -----
        corners = np.array([[0,   0,   1],
                            [w-1, 0,   1],
                            [w-1, h-1, 1],
                            [0,   h-1, 1]], dtype=float).T  # 3x4
        plane_pts = H_img2plane @ corners   # 3x4
        plane_pts /= plane_pts[2:3, :]      # normalize
        Xs, Ys = plane_pts[0, :], plane_pts[1, :]

        # Add margin and build pixel scaling
        m = float(self.get_parameter('margin_m').value)
        xmin, xmax = Xs.min() - m, Xs.max() + m
        ymin, ymax = Ys.min() - m, Ys.max() + m

        ppm = float(self.get_parameter('pixels_per_meter').value)
        out_w = int(np.clip(np.ceil((xmax - xmin) * ppm), 10, 5000))
        out_h = int(np.clip(np.ceil((ymax - ymin) * ppm), 10, 5000))

        # Optional caps
        maxW = int(self.get_parameter('max_out_width').value)
        maxH = int(self.get_parameter('max_out_height').value)
        if maxW > 0:
            out_w = min(out_w, maxW)
        if maxH > 0:
            out_h = min(out_h, maxH)

        # Map plane meters -> output pixels (flip Y so up is up)
        # X_pix = ppm * (X - xmin)
        # Y_pix = ppm * (ymax - Y)
        A = np.array([[ ppm,   0.0, -ppm * xmin],
                      [ 0.0, -ppm,  ppm *  ymax],
                      [ 0.0,  0.0,  1.0      ]], dtype=float)

        # Final mapping from input image to BEV output
        M = A @ H_img2plane

        # ----- Warp -----
        bev = cv2.warpPerspective(undistorted, M, (out_w, out_h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))

        # Publish
        out_msg = self.bridge.cv2_to_imgmsg(bev, encoding='bgr8')
        out_msg.header = msg.header  # preserve timestamp
        self.pub_out.publish(out_msg)

        # Debug log (once in a while)
        self.get_logger().debug(
            f"Top-down size: {out_w}x{out_h}px, X:[{xmin:.3f},{xmax:.3f}]m, Y:[{ymin:.3f},{ymax:.3f}]m"
        )


def main(args=None):
    rclpy.init(args=args)
    node = TopDownRectifier()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
