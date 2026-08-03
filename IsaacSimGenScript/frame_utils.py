import os

import numpy as np
import yaml


DEFAULT_CAMERA_OPTICAL_TO_BASE = np.array([
    [0.0, 0.0, 1.0, 0.25],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.25],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)


def _matrix_from_yaml(entry, key_name, yaml_path):
    if entry is None:
        raise KeyError(f"Missing calibration entry '{key_name}' in {yaml_path}")
    data = entry.get("data")
    rows = int(entry.get("rows", 4))
    cols = int(entry.get("cols", 4))
    matrix = np.array(data, dtype=float).reshape(rows, cols)
    return matrix


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_camera_info(root_dir):
    return load_yaml(os.path.join(root_dir, "camera_info.yaml"))


def load_lidar_info(root_dir):
    return load_yaml(os.path.join(root_dir, "lidar_info.yaml"))


def load_camera_optical_to_base(cam_info, yaml_path):
    if "camera_optical_to_base" in cam_info:
        return _matrix_from_yaml(cam_info["camera_optical_to_base"], "camera_optical_to_base", yaml_path)
    if "camera_to_base" in cam_info:
        return _matrix_from_yaml(cam_info["camera_to_base"], "camera_to_base", yaml_path)
    return DEFAULT_CAMERA_OPTICAL_TO_BASE.copy()


def load_lidar_sensor_to_base(lidar_info, yaml_path):
    if "lidar_sensor_to_base" in lidar_info:
        return _matrix_from_yaml(lidar_info["lidar_sensor_to_base"], "lidar_sensor_to_base", yaml_path)
    return _matrix_from_yaml(lidar_info["lidar_to_base"], "lidar_to_base", yaml_path)


def matrix_to_yaml_dict(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return {
        "rows": int(matrix.shape[0]),
        "cols": int(matrix.shape[1]),
        "data": matrix.flatten().tolist(),
    }
