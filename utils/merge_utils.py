import os
import cv2
import numpy as np

def load_lidar_with_intensity(ply_path: str) -> np.ndarray:
    """Load 3D LiDAR points from .ply file with intensity"""
    with open(ply_path, 'r') as f:
        while True:
            line = f.readline()
            if line.strip() == "end_header":
                break
        data = np.loadtxt(f)
    return data[:, :3], data[:, 3]

def load_camera_image(image_path: str) -> np.ndarray:
    """Load camera image using OpenCV"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return image

def project_point_to_image(
    point_cam: np.ndarray,
    intrinsics: np.ndarray
) -> tuple[int, int] | None:
    """Project a 3D point (camera frame) into 2D image coordinates"""
    X, Y, Z = point_cam
    if Z <= 0:
        return None  # behind the camera

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return int(u), int(v)

def is_point_visible(
    uv: tuple[int, int],
    image_shape: tuple[int, int]
) -> bool:
    """Check if projected pixel lies within image bounds"""
    u, v = uv
    height, width = image_shape[:2]
    return 0 <= u < width and 0 <= v < height

def transform_lidar_to_camera(
    points_lidar: np.ndarray,
    extrinsics: np.ndarray
) -> np.ndarray:
    """Transform LiDAR points to the camera frame"""
    points_h = np.hstack((points_lidar, np.ones((points_lidar.shape[0], 1))))  # Nx4
    points_cam = (extrinsics @ points_h.T).T[:, :3]
    return points_cam

def sample_color_from_image(
    image: np.ndarray,
    uv: tuple[int, int]
) -> list[float]:
    """Sample RGB color from image at given pixel (normalized to [0, 1])"""
    u, v = uv
    b, g, r = image[v, u]  # OpenCV uses BGR ordering
    return [r / 255.0, g / 255.0, b / 255.0]

def project_lidar_to_image(
    image_path: str,
    lidar_path: str,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    output_path: str
) -> None:
    """
    Projects LiDAR points onto a camera image and saves the image with
    projected points overlaid.

    Only draws points that fall within the image view.

    Args:
        image_path: Path to camera image (BGR).
        lidar_path: Path to LiDAR .ply file.
        intrinsics: 3x3 camera intrinsics matrix.
        extrinsics: 4x4 LiDAR-to-camera extrinsics matrix.
        output_path: Where to save the overlay image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    points, _ = load_lidar_with_intensity(lidar_path)
    points_cam = transform_lidar_to_camera(points, extrinsics)

    for pt_cam in points_cam:
        uv = project_point_to_image(pt_cam, intrinsics)
        if uv and is_point_visible(uv, image.shape):
            u, v = uv
            cv2.circle(image, (u, v), 1, (0, 255, 0), -1)  # green dot

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
