import numpy as np

def create_extrinsics():
    # Create a 4x4 identity matrix to initialize the extrinsic transformation matrix.
    # This gives us:
    # [[1, 0, 0, 0],
    #  [0, 1, 0, 0],
    #  [0, 0, 1, 0],
    #  [0, 0, 0, 1]]
    # It's a placeholder for rotation + translation in homogeneous coordinates.
    extrinsics = np.eye(4)

    # Replace the top-left 3x3 part of the matrix (the rotation block) with a custom rotation matrix.
    # This matrix rotates from LiDAR/world coordinate frame (Z-up) to camera coordinate frame (Z-forward).
    # Coordinate frame transformation:
    # - X_world (forward) → X_cam (right)  → becomes row 3
    # - Y_world (left)    → Y_cam (down)   → becomes row 2 (with negative sign)
    # - Z_world (up)      → Z_cam (forward)→ becomes row 1
    extrinsics[:3, :3] = np.array([
        [0, -1,  0],   # X_cam = -Y_world
        [0,  0, -1],   # Y_cam = -Z_world
        [1,  0,  0]    # Z_cam =  X_world
    ])

    # Set the translation part (the last column of the first 3 rows) to define the camera's position 
    # relative to the LiDAR (or vice versa, depending on direction of transform).
    # In this case, the LiDAR is 0.2 meters higher than the camera, so the translation is +0.2 in Y_cam,
    # because in the camera coordinate system, Y points downward.
    extrinsics[:3, 3] = [0.0, 0.2, 0.0]

    return extrinsics
