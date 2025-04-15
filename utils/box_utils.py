import open3d as o3d
import numpy as np

def create_3d_box(center, size, rotation):
    # Ensure 3D vector inputs
    center = np.array(center, dtype=np.float64)
    if len(center) == 2:
        center = np.append(center, 0.0)  # default Z = 0

    size = np.array(size, dtype=np.float64)
    if len(size) == 2:
        size = np.append(size, 1.7)  # default height for 2D objects (like pedestrians)

    rotation = np.array(rotation, dtype=np.float64)
    if len(rotation) == 2:
        rotation = np.append(rotation, 0.0)

    # Adjust center.z so the box sits on the ground
    center[2] += size[2] / 2.0


    box = o3d.geometry.OrientedBoundingBox()
    box.center = center
    box.extent = size
    box.R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz(rotation)

    return box

def compute_3d_iou(box1, box2):
    aabb1 = box1.get_axis_aligned_bounding_box()
    aabb2 = box2.get_axis_aligned_bounding_box()

    min1 = aabb1.get_min_bound()
    max1 = aabb1.get_max_bound()
    min2 = aabb2.get_min_bound()
    max2 = aabb2.get_max_bound()

    min_inter = np.maximum(min1, min2)
    max_inter = np.minimum(max1, max2)
    inter_dims = np.maximum(0.0, max_inter - min_inter)
    inter_vol = np.prod(inter_dims)

    vol1 = np.prod(max1 - min1)
    vol2 = np.prod(max2 - min2)
    union_vol = vol1 + vol2 - inter_vol

    return inter_vol / union_vol if union_vol > 0 else 0.0
