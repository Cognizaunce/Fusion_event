import json
import open3d as o3d
import numpy as np

# Function to load JSON data
def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# Function to create a 3D box for an object
def create_3d_box(center, size, rotation):
    box = o3d.geometry.OrientedBoundingBox()
    box.center = center
    box.extent = size
    box.R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz(rotation)
    return box

# Load input and output data
input_data = load_json("data/input/scene1.json")  # Replace with your file path
output_data = load_json("data/output/scene1.json")  # Replace with your file path

# Create a list to store 3D geometries
geometries = []

# Add vehicles (CarA and CarB) from input data
for car in ["CarA", "CarB"]:
    car_data = input_data[car]
    center = [car_data["location"]["x"], car_data["location"]["y"], car_data["location"]["z"]]
    size = [car_data["dimensions"]["length"], car_data["dimensions"]["width"], car_data["dimensions"]["height"]]
    rotation = [0, 0, np.radians(car_data["rotation"])]  # Convert rotation to radians
    box = create_3d_box(center, size, rotation)
    box.color = (1, 0, 0)  # Red for vehicles
    geometries.append(box)

# Add other road agents from output data
for agent in output_data:
    center = [agent["location"]["x"], agent["location"]["y"], agent["location"]["z"]]
    size = [agent["dimensions"]["length"], agent["dimensions"]["width"], agent["dimensions"]["height"]]
    rotation = [0, 0, np.radians(agent["rotation"])]  # Convert rotation to radians
    box = create_3d_box(center, size, rotation)
    box.color = (0, 1, 0)  # Green for other agents
    geometries.append(box)

# Visualize the scene
o3d.visualization.draw_geometries(geometries)