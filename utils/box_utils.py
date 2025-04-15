import open3d as o3d
import numpy as np

def create_3d_box(center, size, rotation):
    box = o3d.geometry.OrientedBoundingBox()
    box.center = center
    box.extent = size
    box.R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz(rotation)
    return box

def compute_3d_iou(box1, box2):
    aabb1 = box1.get_axis_aligned_bounding_box()
    aabb2 = box2.get_axis_aligned_bounding_box()
    inter = aabb1.get_intersection(aabb2).volume()
    union = aabb1.volume() + aabb2.volume() - inter
    return inter / union if union > 0 else 0
