import numpy as np

# Transformation matrix from base_link -> wrist_rgbd_camera_depth_optical_frame
T = np.array([
    [1.000,  0.000, -0.000, -0.238],
    [0.000, -0.866,  0.500, -0.300],
    [-0.000, -0.500, -0.866,  0.200],
    [0.000,  0.000,  0.000,  1.000]
])

# Inverse transformation (camera -> base_link)
T_inv = np.linalg.inv(T)

def inverse_transform(x, y, z):
    """Apply inverse transformation: camera frame -> base_link."""
    point = np.array([x, y, z, 1.0])  # homogeneous coordinates
    transformed = T_inv @ point
    return transformed[:3]

if __name__ == "__main__":
    print("Enter point in wrist_rgbd_camera_depth_optical_frame (x y z): ")
    try:
        x, y, z = map(float, input().split())
        bx, by, bz = inverse_transform(x, y, z)
        print(f"\nPoint in camera frame: ({x:.3f}, {y:.3f}, {z:.3f})")
        print(f"Inverse transformed -> base_link frame: ({bx:.3f}, {by:.3f}, {bz:.3f})")
    except Exception as e:
        print("Error:", e)
        print("Please enter values like: 0.1 0.2 0.3")
