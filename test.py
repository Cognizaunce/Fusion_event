# BUG: Point cloud visualization is not working as expected, all points are at the same height

import numpy as np
import open3d as o3d
import cv2
import os

from utils.json_utils import load_json, save_json
from utils.box_utils import create_3d_box, compute_3d_iou
from models.dummy_model import DummyModel


def load_lidar_with_intensity(ply_path):
    with open(ply_path, 'r') as f:
        while True:
            line = f.readline()
            if line.strip() == "end_header":
                break
        data = np.loadtxt(f)
    return data[:, :3], data[:, 3]


def project_point(intrinsics, point_3d):
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    X, Y, Z = point_3d
    if Z <= 0:
        return None
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return int(u), int(v)


def project_lidar_to_image(image_path, lidar_path, intrinsics, extrinsics, output_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found or cannot be read: {image_path}")
    points, intensities = load_lidar_with_intensity(lidar_path)
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    if extrinsics.shape != (4, 4):
        raise ValueError(f"Expected extrinsics to be a 4x4 matrix, but got shape {extrinsics.shape}")
    points_cam = (extrinsics @ points_h.T).T[:, :3]

    for pt, intensity in zip(points_cam, intensities):
        uv = project_point(intrinsics, pt)
        if not uv:
            print(f"Skipped point {pt} due to non-positive Z value.")
        if uv:
            u, v = uv
            if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                if intensities.ptp() == 0:
                    color = 255  # Assign maximum intensity if all values are the same
                else:
                    color = int(255 * (intensity - intensities.min()) / intensities.ptp())
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directories if output_dir is not empty
        os.makedirs(output_dir, exist_ok=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)

def transform_lidar_to_2d_map_frame(pcd, car_center, car_yaw_deg):
    points = np.asarray(pcd.points)  # shape: (N, 3)

    # Rotation: from local car frame to world frame
    yaw_rad = np.radians(car_yaw_deg)
    R = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad)],
        [np.sin(yaw_rad),  np.cos(yaw_rad)]
    ])

    xy = points[:, :2]  # discard z
    rotated_xy = xy @ R.T
    translated_xy = rotated_xy + car_center[:2]

    # Project into 3D (z = 0) so Open3D can render
    z = points[:, 2:]  # preserve Z
    points_3d_global = np.hstack([translated_xy, z])

    pcd_transformed = o3d.geometry.PointCloud()
    pcd_transformed.points = o3d.utility.Vector3dVector(points_3d_global)
    return pcd_transformed


def get_extrinsic_matrix():
    """
    Returns the extrinsic matrix (LiDAR to camera transformation).
    Currently returns identity, but can be replaced with real values.
    """
    extrinsic = np.eye(4)

    # === Replace with real calibration values when available ===
    # rotation = np.array([
    #     [0.9998, 0.0175, 0.0053],
    #     [-0.0176, 0.9998, -0.0041],
    #     [-0.0052, 0.0042, 0.9999]
    # ])
    # translation = np.array([0.2, 0.0, 0.5])
    # extrinsic[:3, :3] = rotation
    # extrinsic[:3, 3] = translation

    return extrinsic


def main():
    scene_id = "scene_001"
    input_data = load_json(f"data/input/{scene_id}.json")
    ground_truth = load_json(f"data/output/{scene_id}.json")

    model = DummyModel(noise_level=0.5)
    predicted_output = model.predict(input_data)
    save_json(predicted_output, f"data/predictions/{scene_id}.json")

    geometries = []
    gt_boxes = []
    pred_boxes = []

    for car in ["CarA", "CarB"]:
        center = input_data[f"{car}_Location"]
        size = input_data[f"{car}_Dimension"]
        rotation = [0, 0, np.radians(input_data[f"{car}_Rotation"])]
        box = create_3d_box(center, size, rotation)
        box.color = (1.0, 0.0, 0.0) if car == "CarA" else (1.0, 0.5, 0.0)
        geometries.append(box)

    for obj in ground_truth:
        center = obj["Location"]
        size = obj["Dimension"]
        rotation = [0, 0, np.radians(obj["Rotation"])]
        box = create_3d_box(center, size, rotation)
        box.color = (0, 1, 0) if obj["object"] == "Car" else (0, 0, 1)
        geometries.append(box)
        gt_boxes.append(box)

    for obj in predicted_output:
        center = obj["Location"]
        size = obj["Dimension"]
        rotation = [0, 0, np.radians(obj["Rotation"])]
        box = create_3d_box(center, size, rotation)
        box.color = (0, 0, 0)
        geometries.append(box)
        pred_boxes.append(box)

    ious = []
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou = compute_3d_iou(pred_box, gt_box)
            ious.append((i, j, iou))
            print(f"Pred {i} vs GT {j}: IoU = {iou:.3f}")

    print(f"\n➡️ Average IoU over scene: {np.mean([x[2] for x in ious]):.3f}")

    # === Load real intrinsics and extrinsics ===
    intrinsics_obj = o3d.camera.PinholeCameraIntrinsic()
    intrinsics_obj.set_intrinsics(
        width=1920,
        height=1080,
        fx=2058.72664,
        fy=2058.72664,
        cx=960.0,
        cy=540.0
    )
    intrinsics = intrinsics_obj.intrinsic_matrix
    extrinsics = get_extrinsic_matrix()

    cam_paths = {
        "CameraA": "data/CameraA/A_001.png",
        "CameraB": "data/CameraB/B_001.png",
    }
    lidar_paths = {
        "LidarA": "data/LidarA/A_001.ply",
        "LidarB": "data/LidarB/B_001.ply",
    }

    for cam_key in cam_paths:
        cam_img = cam_paths[cam_key]
        lidar = lidar_paths[cam_key.replace("Camera", "Lidar")]
        out_path = f"data/predictions/{scene_id}_{cam_key}_projected.png"
        project_lidar_to_image(cam_img, lidar, intrinsics, extrinsics, out_path)
        print(f"Saved projection: {out_path}")


    pcd = o3d.io.read_point_cloud("data/LidarA/A_001.ply")
    car_center = np.array(input_data["CarA_Location"])
    car_yaw_deg = input_data["CarA_Rotation"]

    pcd_local = transform_lidar_to_2d_map_frame(pcd, car_center, car_yaw_deg)
    geometries.append(pcd_local)

    # === Open3D visualization ===
    o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    main()
