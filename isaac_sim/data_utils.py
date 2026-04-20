import os
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
import config

class YOLOFormatter:
    @staticmethod
    def to_yolo_2d(bbox_data, img_w, img_h):
        lines = []
        data = bbox_data["data"] if isinstance(bbox_data, dict) else bbox_data
        if data is None or data.size == 0: return lines

        for box in data:
            if float(box['occlusionRatio']) > 0.8:
                continue

            class_name = bbox_data["info"]["idToLabels"][str(box['semanticId'])]['class']
            class_id = int(box['semanticId'])
            
            x_min, y_min = float(box['x_min']), float(box['y_min'])
            x_max, y_max = float(box['x_max']), float(box['y_max'])
            
            w = (x_max - x_min) / img_w
            h = (y_max - y_min) / img_h
            cx = (x_min / img_w) + (w / 2.0)
            cy = (y_min / img_h) + (h / 2.0)
            
            cx, cy, w, h = np.clip([cx, cy, w, h], 0.0, 1.0)
            if w == 0.0 or h == 0.0:
                continue
            lines.append(f"{class_name} {class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        return lines

    @staticmethod
    def to_yolo_3d(bbox_3d_data):
        lines = []
        data = bbox_3d_data["data"] if isinstance(bbox_3d_data, dict) else bbox_3d_data
        
        if data is None or data.size == 0: 
            return lines

        # 取得 Meta Data
        data_info_meta = bbox_3d_data["info"]["idToLabels"]

        for box in data:
            class_id = int(box['semanticId'])
            class_id_str = str(class_id)
                
            class_name = data_info_meta[class_id_str]['class']

            if float(box['occlusionRatio']) > 0.8:
                continue
            
            transform_matrix = box['transform']
            
            # ==========================================
            # 🚀 修正：從矩陣的 Row 中提取 X, Y, Z 的實際縮放比例
            # Isaac Sim 的矩陣是 Row-Major，前三列向量的長度就是 Scale
            # ==========================================
            scale_x = np.linalg.norm(transform_matrix[0, :3])
            scale_y = np.linalg.norm(transform_matrix[1, :3])
            scale_z = np.linalg.norm(transform_matrix[2, :3])

            # 乘上縮放比例，把「公分(Local)」轉成真正的「公尺(World)」
            l = float(box['x_max'] - box['x_min']) * scale_x
            w = float(box['y_max'] - box['y_min']) * scale_y
            h = float(box['z_max'] - box['z_min']) * scale_z

            local_center = np.array([
                (box['x_max'] + box['x_min']) / 2.0,
                (box['y_max'] + box['y_min']) / 2.0,
                (box['z_max'] + box['z_min']) / 2.0,
                1.0
            ])
            
            # 轉換到世界座標 (中心點計算本來就是對的，因為矩陣相乘包含了 Scale)
            world_center = local_center @ transform_matrix
            cx, cy, cz = world_center[:3]

            # 陷阱二修正：剔除縮放比例 (Scale)，還原純旋轉矩陣
            rot_matrix = transform_matrix[:3, :3].copy()
            # 這裡改成對 axis=1 (Row) 正規化，這才是 USD 矩陣的正確解法
            norms = np.linalg.norm(rot_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0 
            rot_matrix = rot_matrix / norms 
            
            if np.linalg.det(rot_matrix) < 0:
                rot_matrix[:, 2] *= -1
            
            r = R.from_matrix(rot_matrix)
            yaw = r.as_euler('xyz')[2]

            # 加入 class_name
            line = f"{class_name} {class_id} {cx:.6f} {cy:.6f} {cz:.6f} {l:.6f} {w:.6f} {h:.6f} {yaw:.6f}"
            lines.append(line)
            
        return lines

class DataProcessor:
    @staticmethod
    def setup_directories(scene, condition, robot_name, case_name=None):
        if case_name:
            root = os.path.join(config.BASE_SAVE_PATH, scene, condition, case_name, robot_name)
        else:
            root = os.path.join(config.BASE_SAVE_PATH, scene, condition, robot_name)
            
        for folder in config.SUB_FOLDERS:
            folder_path = os.path.join(root, folder)
            os.makedirs(folder_path, exist_ok=True)
        return root

    @staticmethod
    def compute_occupancy(pc, grid_cfg):
        half = grid_cfg['range_xy'] / 2.0
        voxel_size = grid_cfg['voxel_size']
        grid_dim = np.array(grid_cfg['dim'])

        dist_sq = pc[:, 0]**2 + pc[:, 1]**2
        mask = (pc[:, 0] >= -half) & (pc[:, 0] < half) & \
            (pc[:, 1] >= -half) & (pc[:, 1] < half) & \
            (pc[:, 2] >= -0.5) & (pc[:, 2] < grid_cfg['range_z'] - 0.5) & \
            (dist_sq > 0.5**2)
        
        pts = pc[mask]
        if pts.size == 0:
            return np.zeros(grid_cfg['dim'], dtype=np.uint8), np.zeros(grid_dim[:2], dtype=np.uint8)

        mask_floor = pts[:, 2] < -0.25
        mask_obs = pts[:, 2] >= -0.25

        indices = ((pts + [half, half, 0.5]) / voxel_size).astype(int)
        indices = np.clip(indices, [0, 0, 0], grid_dim - 1)

        occ_3d = np.zeros(grid_cfg['dim'], dtype=np.uint8)
        idx_f = indices[mask_floor]
        occ_3d[idx_f[:, 0], idx_f[:, 1], idx_f[:, 2]] = 2
        
        idx_o = indices[mask_obs]
        occ_3d[idx_o[:, 0], idx_o[:, 1], idx_o[:, 2]] = 1

        occ_2d = np.any(occ_3d == 1, axis=2).astype(np.uint8)

        return occ_3d, occ_2d
    
    @staticmethod
    def save_pcd(pc_np, pcd_path):
        pc_np = pc_np.astype(np.float32)
        num_points = pc_np.shape[0]
        num_channels = pc_np.shape[1]
        
        header = "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
        if num_channels >= 4:
            header += "FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
            pc_np = pc_np[:, :4]  # 確保只取前 4 個通道 (x, y, z, intensity)
        else:
            header += "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
            pc_np = pc_np[:, :3]  # 確保只取前 3 個通道 (x, y, z)
            
        header += f"WIDTH {num_points}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        header += f"POINTS {num_points}\nDATA binary\n"
        
        os.makedirs(os.path.dirname(pcd_path), exist_ok=True)
        with open(pcd_path, 'wb') as f:
            f.write(header.encode('ascii'))
            f.write(pc_np.tobytes())

    @staticmethod
    def save_frame(data, occ3d, occ2d, root, f_idx):
        img_h, img_w = data["rgb"].shape[:2]
        
        Image.fromarray(data["rgb"][:, :, :3]).save(os.path.join(root, f"data/rgb/{f_idx}.png"))
        
        lidar_data = data["lidar"]["data"] if isinstance(data["lidar"], dict) else data["lidar"]
        DataProcessor.save_pcd(lidar_data, os.path.join(root, f"data/lidar/{f_idx}.pcd"))
        
        np.save(os.path.join(root, f"data/imu/{f_idx}.npy"), data["imu"])
        np.save(os.path.join(root, f"data/pose/{f_idx}.npy"), {"pos": data["pose"][0], "orient": data["pose"][1]})

        labels = {
            "label/detection_raw/2d_tight_raw": YOLOFormatter.to_yolo_2d(data["tight_bbox"], img_w, img_h),
            "label/detection_raw/2d_loose_raw": YOLOFormatter.to_yolo_2d(data["loose_bbox"], img_w, img_h),
            "label/detection_raw/3d_raw": YOLOFormatter.to_yolo_3d(data["bbox_3d"])
        }
        for path, lines in labels.items():
            txt_path = os.path.join(root, f"{path}/{f_idx}.txt")
            with open(txt_path, "w") as f:
                f.write("\n".join(lines))

        np.save(os.path.join(root, f"label/depth/{f_idx}.npy"), data["depth"])
        np.save(os.path.join(root, f"label/segmentation/semantic/{f_idx}.npy"), data["semantic"])
        np.save(os.path.join(root, f"label/segmentation/instance/{f_idx}.npy"), data["instance"])
        np.save(os.path.join(root, f"label/occupancy/3d/{f_idx}.npy"), occ3d)
        np.save(os.path.join(root, f"label/occupancy/2d_bev/{f_idx}.npy"), occ2d)

        print(f"Saved frame {f_idx} to {root}")