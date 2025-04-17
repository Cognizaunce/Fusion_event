from typing import Sequence, Union
import numpy as np
import open3d as o3d
import cv2
import os

import utils.main_utils as utils
import utils.merge_utils as merge_utils
from models.dummy_model import DummyModel


def merge_lidar_and_camera(lidar_path, image_path, intrinsics, extrinsics):
    """
    Projects LiDAR points into the camera image, assigns image-based color
    to visible points, and returns a colorized point cloud.

    Points not visible in the image are added with a default gray color.
    """
    points_lidar, intensity = merge_utils.load_lidar_with_intensity(lidar_path)
    image = merge_utils.load_camera_image(image_path)
    points_cam = merge_utils.transform_lidar_to_camera(points_lidar, extrinsics)

    visible_points = []
    colors = []
    occluded_points = []

    for i, pt_cam in enumerate(points_cam):
        uv = merge_utils.project_point_to_image(pt_cam, intrinsics)
        if uv and merge_utils.is_point_visible(uv, image.shape):
            u, v = uv
            b, g, r = image[v, u]
            color = [r / 255.0, g / 255.0, b / 255.0]
            visible_points.append(points_lidar[i])  # original (LiDAR) position
            colors.append(color)
        else:
            occluded_points.append(points_lidar[i])

    # Merge and build Open3D PointCloud
    merged_points = visible_points + occluded_points
    merged_colors = colors + [[0.5, 0.5, 0.5]] * len(occluded_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(merged_points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(merged_colors))
    return pcd


def transform_pointcloud_to_global_frame(
    pcd: o3d.geometry.PointCloud,
    origin: Union[np.ndarray, Sequence[float]],
    yaw_deg: float
) -> o3d.geometry.PointCloud:
    """
    Applies 2D yaw rotation and translation to move the point cloud from local
    coordinates to a global/world frame. Preserves any colors (e.g., intensity).
    """
    points = np.asarray(pcd.points)

    yaw_rad = np.radians(yaw_deg)
    R = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad)],
        [np.sin(yaw_rad),  np.cos(yaw_rad)]
    ])

    xy = points[:, :2]
    rotated_xy = xy @ R.T
    translated_xy = rotated_xy + np.asarray(origin)[:2]

    z = points[:, 2:].reshape(-1, 1)
    points_3d_global = np.hstack([translated_xy, z + 1.7])

    pcd_transformed = o3d.geometry.PointCloud()
    pcd_transformed.points = o3d.utility.Vector3dVector(points_3d_global)

    # Preserve colors (e.g., grayscale intensity)
    if pcd.has_colors():
        pcd_transformed.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))

    return pcd_transformed

def create_ground_grid(size=20.0, step=1.0, z_height=0.0):
    lines = []
    points = []
    idx = 0
    for i in range(-int(size/2), int(size/2)+1):
        points.append([i, -size/2, z_height])
        points.append([i,  size/2, z_height])
        lines.append([idx, idx + 1])
        idx += 2
        points.append([-size/2, i, z_height])
        points.append([ size/2, i, z_height])
        lines.append([idx, idx + 1])
        idx += 2

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]] * len(lines))
    return line_set


def main():
    # Step 1: Load the datasets.
    #scene_id = [str(i).zfill(3) for i in range(1, 11)]
    scene_id = "001"  # Use first scene ID for testing
    input_data = utils.load_json(f"data/input/scene_{scene_id}.json")
    ground_truth = utils.load_json(f"data/output/scene_{scene_id}.json")

    model = DummyModel(noise_level=0.5)
    predicted_output = model.predict(input_data)
    utils.save_json(predicted_output, f"data/predictions/scene_{scene_id}.json")

    gt_boxes = []
    pred_boxes = []
    geometries = []

    # Step 2: Add ego cars
    for car in ["CarA", "CarB"]:
        center = input_data[f"{car}_Location"]
        size = input_data[f"{car}_Dimension"]
        rotation = [0, 0, np.radians(input_data[f"{car}_Rotation"])]
        box = utils.create_3d_box(center, size, rotation)
        box.color = (1.0, 0.0, 0.0) if car == "CarA" else (1.0, 0.5, 0.0)
        geometries.append(box)

    # Step 3: Add ground truth boxes
    for obj in ground_truth:
        center = obj["Location"]
        size = obj["Dimension"]
        rotation = [0, 0, np.radians(obj["Rotation"])]
        box = utils.create_3d_box(center, size, rotation)
        box.color = (0, 1, 0) if obj["object"] == "Car" else (0, 0, 1)
        geometries.append(box)
        gt_boxes.append(box)

    # Step 4: Define intrinsics and extrinsics
    intrinsics = np.array([
        [2058.72664, 0, 960.0],
        [0, 2058.72664, 540.0],
        [0, 0, 1]
    ])
    extrinsics = utils.create_extrinsics()

    cam_paths = {
        "CameraA": f"data/CameraA/A_{scene_id}.png",
        "CameraB": f"data/CameraB/B_{scene_id}.png",
    }
    lidar_paths = {
        "LidarA": f"data/LidarA/A_{scene_id}.ply",
        "LidarB": f"data/LidarB/B_{scene_id}.ply",
    }

    # Step 5: Project and colorize LiDAR using camera image
    for cam_key in cam_paths:
        cam_img = cam_paths[cam_key]
        lidar = lidar_paths[cam_key.replace("Camera", "Lidar")]
        out_path = f"data/predictions/{scene_id}_{cam_key}_projected.png"

        merge_utils.project_lidar_to_image(cam_img, lidar, intrinsics, extrinsics, out_path)

        # Step 6: Transform LiDAR to global frame add to display features
        car_key = cam_key.replace("Camera", "Car")
        if f"{car_key}_Location" in input_data:
            car_center = np.array(input_data[f"{car_key}_Location"])
            car_yaw_deg = input_data[f"{car_key}_Rotation"]

            pcd_colored = merge_lidar_and_camera(lidar, cam_img, intrinsics, extrinsics)
            pcd_global = transform_pointcloud_to_global_frame(pcd_colored, car_center, car_yaw_deg)
            geometries.append(pcd_global)

    # Step 7: Add predicted boxes
    for obj in predicted_output:
        center = obj["Location"]
        size = obj["Dimension"]
        rotation = [0, 0, np.radians(obj["Rotation"])]
        box = utils.create_3d_box(center, size, rotation)
        box.color = (0, 0, 0)
        geometries.append(box)
        pred_boxes.append(box)

    # Step 8: Compute IoUs
    ious = []
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou = utils.compute_3d_iou(pred_box, gt_box)
            ious.append((i, j, iou))
            print(f"Pred {i} vs GT {j}: IoU = {iou:.3f}")

    print(f"\n➡️ Average IoU over scene: {np.mean([x[2] for x in ious]):.3f}")

    # Step 9: Add ground grid and display scene
    geometries.append(create_ground_grid())
    o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    main()
