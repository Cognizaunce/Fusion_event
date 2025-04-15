from utils.json_utils import load_json, save_json
from utils.box_utils import create_3d_box, compute_3d_iou
from models.dummy_model import DummyModel
import open3d as o3d
import numpy as np
import os

# Load input and ground truth
scene_id = "scene_001"
input_data = load_json(f"data/input/{scene_id}.json")  # Replace with your file path
ground_truth = load_json(f"data/output/{scene_id}.json")  # Replace with your file path

# Create a list to store 3D geometries
geometries = []

# Add vehicles (CarA and CarB) from input data
for car in ["CarA", "CarB"]:
    location_key = f"{car}_Location"
    rotation_key = f"{car}_Rotation"
    dimension_key = f"{car}_Dimension"
    
    center = input_data[location_key] + [0]  # Add z-coordinate as 0
    size = input_data[dimension_key]
    rotation = [0, 0, np.radians(input_data[rotation_key])]  # Convert rotation to radians
    
    box = create_3d_box(center, size, rotation)
    box.color = (1, 0, 0)  # Red for vehicles
    geometries.append(box)

# Add other road agents from output data
for agent in ground_truth:
    center = agent["Location"] + [0]  # Add z-coordinate as 0
    size = agent["Dimension"]
    rotation = [0, 0, np.radians(agent["Rotation"])]  # Convert rotation to radians
    box = create_3d_box(center, size, rotation)
    
    # Set color based on object type
    if agent["object"] == "Car":
        box.color = (0, 1, 0)  # Green for cars
    elif agent["object"] == "Pedestrian":
        box.color = (0, 0, 1)  # Blue for pedestrians
    
    geometries.append(box)

    # Generate predictions
    model = DummyModel(noise_level=0.5)
    predicted_output = model.predict(input_data)

    # [Optional] Save predictions
    os.makedirs("data/predictions", exist_ok=True)
    with open(f"data/predictions/{scene_id}.json", 'w') as f:
        import json
        json.dump(predicted_output, f, indent=2)

# Visualize the scene
o3d.visualization.draw_geometries(geometries)