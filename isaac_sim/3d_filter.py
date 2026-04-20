import os
import glob
import numpy as np
import yaml
from collections import defaultdict
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. 參數與路徑設定
# ==========================================
DATASET_ROOT = os.path.expanduser("~/_my_code/Dataset/full_warehouse_v1/dual_case")

# 你想處理的類別
KEEP_CLASSES = [
    "box", "wet_floor_sign", "traffic_cone", "fire_extinguisher", 
    "chair", "water_dispenser", "medical_bag", "trash_can",
    "laptop", "monitor", "potted_plant", "rubiks_cube", "table", "rack"
]

OCCLUDED_DIR_NAME = "detection_occluded_3d"
DIST_THRESHOLD = 0.5  # 判定為「同一個物體」的距離閾值 (公尺)

# ==========================================
# 2. 幾何投影與 FOV 檢查器
# ==========================================
class FOVChecker:
    def __init__(self, robot_dir):
        self.robot_dir = robot_dir
        
        yaml_path = os.path.join(robot_dir, "camera_info.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"找不到相機參數檔: {yaml_path}")
            
        with open(yaml_path, 'r') as f:
            cam_info = yaml.safe_load(f)
            
        self.K = np.array(cam_info["camera_matrix"]["data"]).reshape(3, 3)
        # 自動相容不同的 YAML Key
        self.img_w = cam_info.get("image_width") or cam_info.get("width")
        self.img_h = cam_info.get("image_height") or cam_info.get("height")
        
        # Base to Camera 外參 (根據你的 Visualizer 設定)
        self.T_base_to_cam = np.array([
            [ 0.0,  0.0,  1.0,  0.25],
            [-1.0,  0.0,  0.0,  0.0],
            [ 0.0, -1.0,  0.0,  0.25],
            [ 0.0,  0.0,  0.0,  1.0]
        ])

    def check_fov(self, frame_id, bbox_line):
        """判斷 3D BBox 是否落在相機 FOV 內"""
        parts = bbox_line.strip().split()
        if len(parts) < 9: return False
            
        # 格式: class_name class_id bx by bz l w h yaw
        bx, by, bz, l, w, h, yaw = map(float, parts[2:9])

        pose_path = os.path.join(self.robot_dir, "data", "pose", f"{frame_id}.npy")
        if not os.path.exists(pose_path): return False
            
        pose_data = np.load(pose_path, allow_pickle=True).item()
        pos, q_scipy = np.array(pose_data['pos']), pose_data['orient']
        
        # World -> Camera 轉換
        T_world_to_base = np.eye(4)
        T_world_to_base[:3, :3] = R.from_quat(q_scipy).as_matrix()
        T_world_to_base[:3, 3] = pos
        T_world_to_cam_inv = np.linalg.inv(T_world_to_base @ self.T_base_to_cam)
        
        # 計算 8 個頂點
        dx, dy, dz = l / 2.0, w / 2.0, h / 2.0
        x_c = [dx, dx, -dx, -dx, dx, dx, -dx, -dx]
        y_c = [dy, -dy, -dy, dy, dy, -dy, -dy, dy]
        z_c = [dz, dz, dz, dz, -dz, -dz, -dz, -dz]
        
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        pts_world = np.vstack((
            [bx + x*cos_y - y*sin_y for x, y in zip(x_c, y_c)],
            [by + x*sin_y + y*cos_y for x, y in zip(x_c, y_c)],
            [bz + z for z in z_c],
            np.ones(8)
        ))
        
        pts_cam = T_world_to_cam_inv @ pts_world
        x_cv, y_cv, z_cv = pts_cam[0, :], pts_cam[1, :], pts_cam[2, :]
        
        if not np.any(z_cv > 0.1): return False
            
        z_cv_safe = np.maximum(z_cv, 1e-5)
        u = (self.K[0, 0] * x_cv / z_cv_safe) + self.K[0, 2]
        v = (self.K[1, 1] * y_cv / z_cv_safe) + self.K[1, 2]
        
        u_valid, v_valid = u[z_cv > 0.1], v[z_cv > 0.1]
        
        # 影像範圍檢查
        in_view = (u_valid >= 0) & (u_valid <= self.img_w) & (v_valid >= 0) & (v_valid <= self.img_h)
        if np.any(in_view): return True
            
        # 處理巨大物件覆蓋畫面的情況
        if (np.max(u_valid) >= 0 and np.min(u_valid) <= self.img_w) and \
           (np.max(v_valid) >= 0 and np.min(v_valid) <= self.img_h):
            return True
            
        return False

# ==========================================
# 3. 資料解析與空間匹配
# ==========================================
def parse_label_file(file_path):
    """解析標籤，回傳物件清單，包含原始字串與座標"""
    objects = []
    if not os.path.exists(file_path): return objects
        
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9: continue
            cls_name = parts[0]
            if KEEP_CLASSES and cls_name not in KEEP_CLASSES: continue
            
            objects.append({
                "raw": line.strip(),
                "class": cls_name,
                "pos": np.array([float(parts[2]), float(parts[3]), float(parts[4])])
            })
    return objects

def is_already_seen(target_obj, scene_objs, threshold=DIST_THRESHOLD):
    """基於類別與距離判斷 R1 是否已經看過 R2 的這個物體"""
    for obj in scene_objs:
        if obj["class"] == target_obj["class"]:
            if np.linalg.norm(target_obj["pos"] - obj["pos"]) < threshold:
                return True
    return False

def find_robot_pairs(root_dir):
    search_pattern = os.path.join(root_dir, "**", "label", "detection_raw", "3d_raw")
    all_3d_dirs = glob.glob(search_pattern, recursive=True)
    case_dict = defaultdict(list)
    for path in all_3d_dirs:
        robot_dir = os.path.dirname(os.path.dirname(os.path.dirname(path)))
        # get only case_1 folder
        if "case_1" in robot_dir:
            case_dict[os.path.dirname(robot_dir)].append(robot_dir)
    return {k: sorted(v) for k, v in case_dict.items() if len(v) >= 2}

# ==========================================
# 4. 主流程
# ==========================================
def process_occlusion_labels():
    print("🔍 開始掃描資料夾結構...")
    cases = find_robot_pairs(DATASET_ROOT)
    if not cases:
        print("⚠️ 找不到 Case，請確認路徑。")
        return
        
    print(f"✅ 找到 {len(cases)} 個 Case。")

    for case_path, robot_dirs in cases.items():
        r1_dir, r2_dir = robot_dirs[0], robot_dirs[1]
        r1_name, r2_name = os.path.basename(r1_dir), os.path.basename(r2_dir)
        
        print(f"\n🤖 配對: {r1_name} <---> {r2_name}")
        
        r1_3d_raw = os.path.join(r1_dir, "label", "detection_raw", "3d_raw")
        r2_3d_raw = os.path.join(r2_dir, "label", "detection_raw", "3d_raw")
        
        r1_out_dir = os.path.join(r1_dir, "label", OCCLUDED_DIR_NAME)
        r2_out_dir = os.path.join(r2_dir, "label", OCCLUDED_DIR_NAME)
        os.makedirs(r1_out_dir, exist_ok=True)
        os.makedirs(r2_out_dir, exist_ok=True)

        checker_r1, checker_r2 = FOVChecker(r1_dir), FOVChecker(r2_dir)
        common_files = sorted(list(set(os.listdir(r1_3d_raw)) & set(os.listdir(r2_3d_raw))))
        
        t1, t2 = 0, 0
        for frame_file in common_files:
            fid = os.path.splitext(frame_file)[0]
            r1_objs = parse_label_file(os.path.join(r1_3d_raw, frame_file))
            r2_objs = parse_label_file(os.path.join(r2_3d_raw, frame_file))
            
            # 任務 A: R2 有，但 R1 沒看到的物體
            r1_occ = [obj["raw"] for obj in r2_objs if not is_already_seen(obj, r1_objs) and checker_r1.check_fov(fid, obj["raw"])]
            if r1_occ:
                with open(os.path.join(r1_out_dir, frame_file), 'w') as f:
                    f.write("\n".join(r1_occ) + "\n")
                t1 += len(r1_occ)

            # 任務 B: R1 有，但 R2 沒看到的物體
            r2_occ = [obj["raw"] for obj in r1_objs if not is_already_seen(obj, r2_objs) and checker_r2.check_fov(fid, obj["raw"])]
            if r2_occ:
                with open(os.path.join(r2_out_dir, frame_file), 'w') as f:
                    f.write("\n".join(r2_occ) + "\n")
                t2 += len(r2_occ)

        print(f"   📈 結果: {r1_name} +{t1} | {r2_name} +{t2}")

if __name__ == "__main__":
    process_occlusion_labels()