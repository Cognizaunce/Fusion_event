from utils.json_utils import load_json, save_json
from utils.box_utils import create_3d_box, compute_3d_iou
from models.dummy_model import DummyModel
import open3d as o3d
import numpy as np
import os
import json

# ==== Load input and ground truth ====
scene_id = "scene_001"
input_data = load_json(f"data/input/{scene_id}.json")
ground_truth = load_json(f"data/output/{scene_id}.json")

# ==== Generate predictions ====
model = DummyModel(noise_level=0.5)
predicted_output = model.predict(input_data)

# Save predictions
os.makedirs("data/predictions", exist_ok=True)
save_json(predicted_output, f"data/predictions/{scene_id}.json")

# ==== Visualize input cars (CarA, CarB) ====
geometries = []
for car in ["CarA", "CarB"]:
    center = input_data[f"{car}_Location"]
    size = input_data[f"{car}_Dimension"]
    rotation = [0, 0, np.radians(input_data[f"{car}_Rotation"])]
    box = create_3d_box(center, size, rotation)
        # Label with distinct color
    if car == "CarA":
        box.color = (1.0, 0.0, 0.0)  # Red
    else:
        box.color = (1.0, 0.5, 0.0)  # Orange
    geometries.append(box)

# ==== Visualize GT vs Prediction and compute IoU ====

# Add GT boxes (green for cars, blue for pedestrians)
gt_boxes = []
for obj in ground_truth:
    center = obj["Location"]
    size = obj["Dimension"]
    rotation = [0, 0, np.radians(obj["Rotation"])]
    box = create_3d_box(center, size, rotation)
    box.color = (0, 1, 0) if obj["object"] == "Car" else (0, 0, 1)
    geometries.append(box)
    gt_boxes.append(box)

# Add prediction boxes (black)
pred_boxes = []
for obj in predicted_output:
    center = obj["Location"]
    size = obj["Dimension"]
    rotation = [0, 0, np.radians(obj["Rotation"])]
    box = create_3d_box(center, size, rotation)
    box.color = (0, 0, 0)
    geometries.append(box)
    pred_boxes.append(box)

    # Compute full IoU Matrix
    ious = []
for i, pred_box in enumerate(pred_boxes):
    for j, gt_box in enumerate(gt_boxes):
        iou = compute_3d_iou(pred_box, gt_box)
        ious.append((i, j, iou))
        print(f"Pred {i} vs GT {j}: IoU = {iou:.3f}")

print(f"\n➡️ Average IoU over scene: {np.mean(ious):.3f}")

# ==== Show visualization ====
o3d.visualization.draw_geometries(geometries)
