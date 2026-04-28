import os

BASE_SAVE_PATH = os.path.expanduser("~/_my_code/Dataset")

SUB_FOLDERS = [
    "data/rgb", 
    "data/lidar", 
    "data/depth",
    "label/detection/2d_tight", 
    "label/detection/2d_loose",
    "label/detection/3d",
    "label/segmentation/semantic",
]

LIDAR_RECORD_FOV_DEG = 90.0
LIDAR_FOV_COMPLETENESS_BINS = 9
LIDAR_FOV_COMPLETENESS_MIN_POINTS = 8
LIDAR_FOV_COMPLETENESS_MIN_RATIO = 0.20

WHEEL_RADIUS = 0.098
WHEEL_BASE = 0.375

KEEP_CLASSES = [
                "wet_floor_sign", 
                "traffic_cone", 
                "fire_extinguisher", 
                "chair",
                "medical_bag",
                "trash_can",
                "laptop",
                "monitor",
                "plant",
                "rubiks_cube",
                "table",
                "mug",
                ] 
