import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.patches as patches

# ==========================================
# 設定區域
# ==========================================
BASE_DIR = os.path.expanduser("~/_my_code/Dataset/full_warehouse_v1/dual_case/case_1/Jackal_R1")
FRAME_ID = "0009366"  # 填入你想 debug 的影格編號 (不含副檔名)

class ProjectionDebugger:
    def __init__(self, robot_dir):
        self.robot_dir = robot_dir
        self.load_camera_params()
        
        # 你的預設外參 (Base to Camera)
        # 注意：如果投影歪掉，通常是這裡或座標定義(OpenCV vs ROS)的問題
        self.T_base_to_cam = np.array([
            [ 0.0,  0.0,  1.0,  0.25],
            [-1.0,  0.0,  0.0,  0.0],
            [ 0.0, -1.0,  0.0,  0.25],
            [ 0.0,  0.0,  0.0,  1.0]
        ])

    def load_camera_params(self):
        yaml_path = os.path.join(self.robot_dir, "camera_info.yaml")
        with open(yaml_path, 'r') as f:
            cam_info = yaml.safe_load(f)
        
        # 自動相容 width / image_width
        self.img_w = cam_info.get("image_width") or cam_info.get("width")
        self.img_h = cam_info.get("image_height") or cam_info.get("height")
        self.K = np.array(cam_info["camera_matrix"]["data"]).reshape(3, 3)
        print(f"✅ 已讀取相機參數: {self.img_w}x{self.img_h}")

    def load_frame_data(self, fid):
        # 讀取 RGB (.png)
        img_path = os.path.join(self.robot_dir, f"data/rgb/{fid}.png")
        img = Image.open(img_path)

        # 讀取 Pose (.npy)
        pose_path = os.path.join(self.robot_dir, f"data/pose/{fid}.npy")
        pose_data = np.load(pose_path, allow_pickle=True).item()

        # 讀取 3D Label (.txt) - 假設格式: cls_id bx by bz l w h yaw
        label_path = os.path.join(self.robot_dir, f"label/detection_occluded_3d/{fid}.txt")
        # /home/uuugaga/_my_code/Dataset/full_warehouse_v1/dual_case/case_1/Jackal_R1/label/detection_occluded_3d/0009366.txt
        boxes = []
        print(f"📂 嘗試讀取標籤: {label_path}")
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    boxes.append(list(map(float, line.strip().split())))

        print(boxes)
        
        return img, pose_data, boxes

    def get_3d_corners(self, box):
        # box: [cls, bx, by, bz, l, w, h, yaw]
        _, bx, by, bz, l, w, h, yaw = box
        dx, dy, dz = l/2, w/2, h/2
        
        # 8個頂點 (Local)
        corners = np.array([
            [ dx,  dy,  dz], [ dx, -dy,  dz], [-dx, -dy,  dz], [-dx,  dy,  dz],
            [ dx,  dy, -dz], [ dx, -dy, -dz], [-dx, -dy, -dz], [-dx,  dy, -dz]
        ])
        
        # 旋轉與平移到世界座標
        rot = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw),  np.cos(yaw), 0],
                        [0, 0, 1]])
        world_corners = (rot @ corners.T).T + np.array([bx, by, bz])
        return world_corners

    def debug_visualize(self, fid):
        img, pose, boxes = self.load_frame_data(fid)
        
        # 計算 World -> Camera 矩陣
        T_world_to_base = np.eye(4)
        T_world_to_base[:3, :3] = R.from_quat(pose['orient']).as_matrix()
        T_world_to_base[:3, 3] = pose['pos']
        
        T_world_to_cam = T_world_to_base @ self.T_base_to_cam
        T_cam_to_world_inv = np.linalg.inv(T_world_to_cam) # 這是 World 到 Camera 的轉換

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # --- 左圖：影像投影 ---
        ax1.imshow(img)
        ax1.set_title(f"Frame {fid}: 3D Projection Debug")

        # --- 右圖：世界座標俯視 (BEV) ---
        ax2.set_aspect('equal')
        ax2.grid(True)
        ax2.set_title("World Coordinates (BEV)")
        # 畫機器人位置
        rx, ry = pose['pos'][0], pose['pos'][1]
        ax2.scatter(rx, ry, c='red', marker='^', s=100, label='Robot')

        edges = [[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4], [0,4], [1,5], [2,6], [3,7]]

        for i, box in enumerate(boxes):
            corners_w = self.get_3d_corners(box)
            
            # 1. 畫在 BEV (右圖)
            ax2.plot(corners_w[:4, 0], corners_w[:4, 1], 'g-')
            ax2.text(box[1], box[2], f"ID:{i}", color='blue')

            # 2. 投影到影像 (左圖)
            pts_w_4d = np.vstack((corners_w.T, np.ones(8)))
            pts_cam = T_cam_to_world_inv @ pts_w_4d
            
            # 過濾相機後方的點
            if np.any(pts_cam[2, :] < 0.1):
                print(f"⚠️ Box {i} 有頂點在相機後方，跳過投影")
                continue

            # 投影公式: u = f*x/z + cx
            z = pts_cam[2, :]
            u = (self.K[0, 0] * pts_cam[0, :] / z) + self.K[0, 2]
            v = (self.K[1, 1] * pts_cam[1, :] / z) + self.K[1, 2]

            for edge in edges:
                ax1.plot([u[edge[0]], u[edge[1]]], [v[edge[0]], v[edge[1]]], color='lime', lw=1.5)
            
        ax2.legend()
        plt.tight_layout()
        save_path = f"debug_frame_{fid}.png"
        plt.savefig(save_path)
        print(f"🎨 Debug 圖檔已儲存至: {save_path}")

if __name__ == "__main__":
    debugger = ProjectionDebugger(BASE_DIR)
    debugger.debug_visualize(FRAME_ID)