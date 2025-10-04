#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import cv2

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Int32, Float32MultiArray
from cv_bridge import CvBridge

from custom_msgs.msg import InferenceResult

# --- rotation helpers ---
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

class CircleDetectorBEV(Node):
    def __init__(self):
        super().__init__('circle_detector_bev')
        self.bridge = CvBridge()

        # ---- I/O topics ----
        self.declare_parameter('camera_topic', '/wrist_rgbd_depth_sensor/image_raw')
        cam_topic = self.get_parameter('camera_topic').value
        default_info = cam_topic.replace('/image_raw', '/camera_info')
        self.declare_parameter('camera_info_topic', default_info)

        self.sub_img  = self.create_subscription(Image, cam_topic, self.on_image, 10)
        self.sub_info = self.create_subscription(CameraInfo,
                                                 self.get_parameter('camera_info_topic').value,
                                                 self.on_info, 10)
        self.pub_bev     = self.create_publisher(Image, '/tilt_corrected_output', 1)
        self.pub_annot   = self.create_publisher(Image, '/circle_detector/output', 1)
        self.pub_inf     = self.create_publisher(InferenceResult, '/circle_inference_result', 10)
        self.pub_center  = self.create_publisher(PointStamped, '/circle_center', 10)
        self.pub_count   = self.create_publisher(Int32, '/circle_count', 10)
        self.pub_radii   = self.create_publisher(Float32MultiArray, '/circle_radii_px', 10)
        self.pub_cropped = self.create_publisher(Image, '/circle_detector/output_cropped', 1)

        # ---- Physical setup ----
        self.declare_parameter('tilt_axis', 'pitch')
        self.declare_parameter('tilt_deg', 60.0)
        self.declare_parameter('height_m', 0.7484)
        self.declare_parameter('yaw_deg', 0.0)

        # ---- BEV output scaling/extent ----
        self.declare_parameter('pixels_per_meter', 500.0)
        self.declare_parameter('margin_m', 0.02)
        self.declare_parameter('max_out_width',  0)
        self.declare_parameter('max_out_height', 0)

        # ---- Hough params ----
        self.declare_parameter('dp', 1.2)
        self.declare_parameter('minDist', 40.0)
        self.declare_parameter('param1', 100.0)
        self.declare_parameter('param2', 30.0)

        # ---- Allowed small radii ----
        self.declare_parameter('allowed_radii', [17, 18])  # 19 ignored silently

        # ---- Axis overlays (kept for annotated image) ----
        self.declare_parameter('draw_axes', True)
        self.declare_parameter('axes_mode', 'meters')
        self.declare_parameter('axis_x_m', 0.0)
        self.declare_parameter('axis_y_m', 0.3)
        self.declare_parameter('axis_x_px', 100)
        self.declare_parameter('axis_y_px', 100)
        self.declare_parameter('axis_thickness', 1)
        self.declare_parameter('axis_color_b', 255)
        self.declare_parameter('axis_color_g', 255)
        self.declare_parameter('axis_color_r', 255)

        # ---- Crop params ----
        self.CROP_SIZE = (200, 200)  # (w, h)

        self.K = None
        self.D = None
        self.size = None

        self.get_logger().info(f"Subscribed image: {cam_topic}")
        self.get_logger().info(f"Subscribed camera info: {self.get_parameter('camera_info_topic').value}")

    def on_info(self, msg: CameraInfo):
        try:
            self.K = np.array(msg.k, dtype=float).reshape(3, 3)
            self.D = np.array(msg.d, dtype=float).reshape(-1)
            self.size = (int(msg.width), int(msg.height))
        except Exception as e:
            self.get_logger().error(f"CameraInfo parse error: {e}")

    def on_image(self, msg: Image):
        if self.K is None or self.size is None:
            self.get_logger().warn("Waiting for CameraInfo...")
            return

        # Decode & undistort
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        w, h = self.size
        newK = self.K.copy()
        und = cv2.undistort(img, self.K, self.D, None, newK)

        # ---- Homography ----
        height_m = float(self.get_parameter('height_m').value)
        tilt_deg = float(self.get_parameter('tilt_deg').value)
        yaw_deg  = float(self.get_parameter('yaw_deg').value)
        tilt_axis = str(self.get_parameter('tilt_axis').value).lower()

        R_wc_top = np.array([[ 1,  0,  0],
                             [ 0, -1,  0],
                             [ 0,  0, -1]], dtype=float)

        tilt_from_vertical_deg = max(0.0, 90.0 - tilt_deg)
        tilt = np.deg2rad(tilt_from_vertical_deg)
        yaw  = np.deg2rad(yaw_deg)

        if tilt_axis == 'pitch':
            R_wc = R_wc_top @ Rx(tilt) @ Rz(yaw)
        elif tilt_axis == 'roll':
            R_wc = R_wc_top @ Ry(tilt) @ Rz(yaw)
        else:
            R_wc = R_wc_top @ Rx(tilt) @ Rz(yaw)

        R_cw = R_wc.T
        C_w = np.array([0.0, 0.0, height_m], dtype=float)
        t = -R_cw @ C_w

        r1 = R_cw[:, 0].reshape(3, 1)
        r2 = R_cw[:, 1].reshape(3, 1)
        H_plane2img = newK @ np.hstack([r1, r2, t.reshape(3, 1)])
        H_img2plane = np.linalg.inv(H_plane2img)

        corners = np.array([[0,   0,   1],
                            [w-1, 0,   1],
                            [w-1, h-1, 1],
                            [0,   h-1, 1]], dtype=float).T
        plane_pts = H_img2plane @ corners
        plane_pts /= plane_pts[2:3, :]
        Xs, Ys = plane_pts[0, :], plane_pts[1, :]

        m = float(self.get_parameter('margin_m').value)
        xmin, xmax = Xs.min() - m, Xs.max() + m
        ymin, ymax = Ys.min() - m, Ys.max() + m

        ppm = float(self.get_parameter('pixels_per_meter').value)
        out_w = int(np.clip(np.ceil((xmax - xmin) * ppm), 10, 10000))
        out_h = int(np.clip(np.ceil((ymax - ymin) * ppm), 10, 10000))

        maxW = int(self.get_parameter('max_out_width').value)
        maxH = int(self.get_parameter('max_out_height').value)
        if maxW > 0: out_w = min(out_w, maxW)
        if maxH > 0: out_h = min(out_h, maxH)

        A = np.array([[ ppm,   0.0, -ppm * xmin],
                      [ 0.0, -ppm,  ppm *  ymax],
                      [ 0.0,  0.0,  1.0      ]], dtype=float)
        M = A @ H_img2plane  # image -> BEV

        bev = cv2.warpPerspective(und, M, (out_w, out_h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))

        # Publish clean BEV
        bev_msg = self.bridge.cv2_to_imgmsg(bev, encoding='bgr8')
        bev_msg.header = msg.header
        self.pub_bev.publish(bev_msg)

        # Keep copies for different uses
        bev_clean   = bev.copy()  # used for CROPS (no overlays)
        annotated   = bev.copy()  # we draw on this for /circle_detector/output

        # ---- Circle detection ----
        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)

        dp        = float(self.get_parameter('dp').value)
        minDist   = float(self.get_parameter('minDist').value)
        param1    = float(self.get_parameter('param1').value)
        param2    = float(self.get_parameter('param2').value)
        H_img, W_img = bev.shape[:2]

        # ================== SMALL circles (17/18; 19 ignored) ==================
        allowed_small = set(int(x) for x in self.get_parameter('allowed_radii').value)
        circles_small = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT,
            dp=dp, minDist=minDist, param1=param1, param2=param2,
            minRadius=15, maxRadius=22
        )

        detections_small = []
        radii_all = []

        def publish_small(x, y, r_int):
            radii_all.append(float(r_int))
            left   = max(0, int(x - r_int))
            top    = max(0, int(y - r_int))
            right  = min(W_img - 1, int(x + r_int))
            bottom = min(H_img - 1, int(y + r_int))

            inf = InferenceResult()
            # inf.class_name = f"circle_r{r_int}"
            inf.class_name = f"circle_small"
            inf.left, inf.top, inf.right, inf.bottom = left, top, right, bottom
            inf.box_width, inf.box_height = right - left, bottom - top
            inf.x, inf.y = float(x), float(y)
            self.pub_inf.publish(inf)

            pt = PointStamped()
            pt.header = msg.header
            pt.point.x, pt.point.y, pt.point.z = float(x), float(y), 0.0
            self.pub_center.publish(pt)

        if circles_small is not None:
            c_small = np.uint16(np.around(circles_small[0]))
            for (x, y, r) in c_small:
                r_int = int(r)
                if r_int == 19:
                    continue
                if r_int not in allowed_small:
                    continue

                # draw on annotated (green)
                cv2.circle(annotated, (x, y), r_int, (0, 255, 0), 2)
                cv2.circle(annotated, (x, y), 2, (0, 255, 0), 3)
                cv2.putText(annotated, f"r={r_int}", (x+6, y-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

                publish_small(x, y, r_int)
                detections_small.append((int(x), int(y), r_int))

        # per-circle + summary logs (small)
        for i, (x, y, r_int) in enumerate(detections_small, start=1):
            self.get_logger().info(f"[CIRCLE {i}] r={r_int}px at (x={x}, y={y})")
        det_str = ", ".join([f"(x={x},y={y},r={r})" for (x,y,r) in detections_small]) or "none"
        self.get_logger().info(f"[CIRCLES] count={len(detections_small)}; {det_str}")

        # ================== LARGE circles (88/89) → crops from CLEAN ==================
        circles_large = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT,
            dp=dp, minDist=minDist, param1=param1, param2=param2,
            minRadius=85, maxRadius=92
        )

        def centered_crop_with_padding(src, cx, cy, box_w, box_h):
            """Return a box_w×box_h crop centered at (cx,cy). Pads with black if needed."""
            half_w, half_h = box_w // 2, box_h // 2
            x0, y0 = int(cx) - half_w, int(cy) - half_h
            x1, y1 = x0 + box_w, y0 + box_h
            Hs, Ws = src.shape[:2]
            x0c, y0c = max(0, x0), max(0, y0)
            x1c, y1c = min(Ws, x1), min(Hs, y1)
            canvas = np.zeros((box_h, box_w, 3), dtype=src.dtype)
            if x0c < x1c and y0c < y1c:
                crop = src[y0c:y1c, x0c:x1c]
                dst_x, dst_y = x0c - x0, y0c - y0
                canvas[dst_y:dst_y+crop.shape[0], dst_x:dst_x+crop.shape[1]] = crop
            return canvas

        crops = []
        if circles_large is not None:
            c_large = np.uint16(np.around(circles_large[0]))
            for (x, y, r) in c_large:
                r_int = int(r)
                if r_int not in (88, 89):
                    continue

                # draw on annotated (red)
                cv2.circle(annotated, (x, y), r_int, (0, 0, 255), 2)
                cv2.circle(annotated, (x, y), 2, (0, 0, 255), 3)
                cv2.putText(annotated, f"R={r_int}", (x+6, y-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

                # crop from CLEAN (no text/overlays)
                crops.append(centered_crop_with_padding(
                    bev_clean, x, y, self.CROP_SIZE[0], self.CROP_SIZE[1]
                ))

        # Build horizontal strip for /output_cropped (NO annotations on crops)
        if len(crops) == 0:
            cropped_out = np.zeros((self.CROP_SIZE[1], self.CROP_SIZE[0], 3), dtype=np.uint8)
        else:
            gap = 5
            total_w = len(crops) * self.CROP_SIZE[0] + (len(crops)-1) * gap
            total_h = self.CROP_SIZE[1]
            cropped_out = np.zeros((total_h, total_w, 3), dtype=np.uint8)
            x_off = 0
            for crop in crops:
                cropped_out[0:self.CROP_SIZE[1], x_off:x_off+self.CROP_SIZE[0]] = crop
                x_off += self.CROP_SIZE[0] + gap

        # ---------- AXIS / ORIGIN OVERLAY (only on annotated) ----------
        if bool(self.get_parameter('draw_axes').value):
            axes_mode = str(self.get_parameter('axes_mode').value).lower()
            thickness = int(self.get_parameter('axis_thickness').value)
            color = (
                int(self.get_parameter('axis_color_b').value),
                int(self.get_parameter('axis_color_g').value),
                int(self.get_parameter('axis_color_r').value),
            )

            def plane_to_bev_px(X, Y):
                u = ppm * (X - xmin)
                v = ppm * (ymax - Y)
                return int(round(u)), int(round(v))

            if axes_mode == 'meters':
                X_m = float(self.get_parameter('axis_x_m').value)
                Y_m = float(self.get_parameter('axis_y_m').value)
                x_px, y_px = plane_to_bev_px(X_m, Y_m)
            else:
                x_px = int(self.get_parameter('axis_x_px').value)
                y_px = int(self.get_parameter('axis_y_px').value)

            if 0 <= x_px < W_img:
                cv2.line(annotated, (x_px, 0), (x_px, H_img-1), color, thickness, lineType=cv2.LINE_AA)
            if 0 <= y_px < H_img:
                cv2.line(annotated, (0, y_px), (W_img-1, y_px), color, thickness, lineType=cv2.LINE_AA)
            if 0 <= y_px < H_img:
                cv2.line(annotated, (0, y_px-100), (W_img-1, y_px-100), color, thickness, lineType=cv2.LINE_AA)

            if 0 <= x_px < W_img and 0 <= y_px < H_img:
                cv2.putText(annotated, f"axes@({x_px},{y_px}) [{axes_mode}]",
                            (min(x_px+6, W_img-120), max(y_px-6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # ---------- Publish ----------
        # clean BEV already published to /tilt_corrected_output
        self.pub_annot.publish(self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8'))

        crop_msg = self.bridge.cv2_to_imgmsg(cropped_out, encoding='bgr8')
        crop_msg.header = msg.header
        self.pub_cropped.publish(crop_msg)

        self.pub_count.publish(Int32(data=len(detections_small)))
        self.pub_radii.publish(Float32MultiArray(data=radii_all))

def main(args=None):
    rclpy.init(args=args)
    node = CircleDetectorBEV()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
