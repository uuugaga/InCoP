import os

# ==========================================
# 資料夾與儲存路徑
# ==========================================
BASE_SAVE_PATH = os.path.expanduser("~/_my_code/Dataset")

SUB_FOLDERS = [
    "data/rgb", 
    "data/lidar", 
    "data/imu", 
    "data/pose",
    "label/depth", 
    "label/detection_raw/2d_tight_raw", 
    "label/detection_raw/2d_loose_raw",
    "label/detection_raw/3d_raw",
    "label/segmentation/semantic", 
    "label/segmentation/instance",
    "label/occupancy/3d", 
    "label/occupancy/2d_bev",
]

# ==========================================
# 佔用網格 (Occupancy Grid) 設定
# ==========================================
GRID_CFG = {
    'voxel_size': 0.2,
    'range_xy': 10.0,
    'range_z': 10.0,
    'dim': (50, 50, 25)
}

# ==========================================
# 機器人硬體規格 (Jackal)
# ==========================================
WHEEL_RADIUS = 0.098
WHEEL_BASE = 0.375



KEEP_CLASSES = ["box", 
                "wet_floor_sign", 
                "traffic_cone", 
                "fire_extinguisher", 
                "chair",
                "water_dispenser",
                "medical_bag",
                "trash_can",
                "laptop",
                "monitor",
                "potted_plant",
                "rubiks_cube",
                "table",
                ] 