import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import yaml
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Wedge
from PIL import Image
from scipy.spatial.transform import Rotation as R
import glob
import cv2
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import subprocess
import shutil
import open3d as o3d

class HeadlessVisualizer:
    def __init__(self, root_path):
        self.root = root_path
        self.cmap = matplotlib.colormaps.get_cmap('tab10')

        self.scene_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(self.root))))
        self.condition = os.path.basename(os.path.dirname(os.path.dirname(self.root)))
        self.robot_name = os.path.basename(self.root)

        yaml_path = os.path.join(self.root, "camera_info.yaml")
        with open(yaml_path, 'r') as f:
            cam_info = yaml.safe_load(f)
        self.K = np.array(cam_info["camera_matrix"]["data"]).reshape(3, 3)
        self.T_base_to_cam = np.array([
            [ 0.0,  0.0,  1.0,  0.25],
            [-1.0,  0.0,  0.0,  0.0],
            [ 0.0, -1.0,  0.0,  0.25],
            [ 0.0,  0.0,  0.0,  1.0]
        ])

        lidar_yaml_path = os.path.join(self.root, "lidar_info.yaml")
        with open(lidar_yaml_path, 'r') as f:
            self.lidar_info = yaml.safe_load(f)

        # Map
        map_png_path = os.path.join(self.root, "/home/uuugaga/Downloads/map/refined_warehouse.png")
        map_yaml_path = os.path.join(self.root, "/home/uuugaga/Downloads/map/refined_warehouse.yaml")
        
        if os.path.exists(map_png_path) and os.path.exists(map_yaml_path):
            with open(map_yaml_path, 'r') as f:
                m_cfg = yaml.safe_load(f)
            
            res, org = m_cfg['resolution'], m_cfg['origin']
            tmp_img = cv2.imread(map_png_path)
            self.map_img = cv2.flip(tmp_img, 0)
            h, w = self.map_img.shape[:2]
            
            self.map_extent = [org[0], org[0] + w * res, org[1], org[1] + h * res]
            self.has_map = True
        else:
            print("Warning: Map files not found. Visualization will use empty background.")
            return 0

    def load_npy(self, sub_path):
        path = os.path.join(self.root, sub_path)
        return np.load(path, allow_pickle=True) if os.path.exists(path) else None
    
    def load_pcd(self, sub_path):
        path = os.path.join(self.root, sub_path)
        if not os.path.exists(path): return None
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points)

    def load_yolo(self, sub_path):
        path = os.path.join(self.root, sub_path)
        if not os.path.exists(path): return []
        with open(path, 'r') as f:
            return [list(map(float, line.strip().split())) for line in f.readlines()]
        
    def load_yolo_3d(self, sub_path):
        path = os.path.join(self.root, sub_path)
        if not os.path.exists(path): return []
        
        yolo_3d_data = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:
                    parsed_line = [int(float(parts[0]))] + list(map(float, parts[1:8]))
                    yolo_3d_data.append(parsed_line)
        return yolo_3d_data

    def render_frame(self, fid, temp_dir=None):
        rgb = Image.open(os.path.join(self.root, f"data/rgb/{fid}.png"))
        depth = self.load_npy(f"label/depth/{fid}.npy")
        lidar = self.load_pcd(f"data/lidar/{fid}.pcd")
        occ_2d = self.load_npy(f"label/occupancy/2d_bev/{fid}.npy")
        occ_3d = self.load_npy(f"label/occupancy/3d/{fid}.npy")
        yolo_tight = self.load_yolo(f"label/detection/2d_tight/{fid}.txt")
        yolo_loose = self.load_yolo(f"label/detection/2d_loose/{fid}.txt")
        yolo_3d = self.load_yolo_3d(f"label/detection/3d/{fid}.txt")
        pose_raw = self.load_npy(f"data/pose/{fid}.npy")
        pose_data = pose_raw.item() if pose_raw is not None else None
        imu_raw = self.load_npy(f"data/imu/{fid}.npy")
        instance_raw = self.load_npy(f"label/segmentation/instance/{fid}.npy")

        img_w, img_h = rgb.size

        fig = plt.figure(figsize=(32, 10), constrained_layout=True, facecolor='#fdfdfd')

        fid_str = str(fid)
        formatted_fid = f"{fid_str[:-3]}.{fid_str[-3:]}"
        plt.suptitle(f"{self.scene_name} | {self.condition} | {self.robot_name} | Frame: {formatted_fid} (s)", fontsize=20, fontweight='bold', color='#333333')

        # --- 1. Original RGB Reference ---
        ax1 = fig.add_subplot(2, 6, 1)
        ax1.imshow(rgb)
        ax1.axis('off')
        ax1.set_title("RGB Image", fontsize=16)

        # --- 2. 2D Tight Bbox ---
        ax2 = fig.add_subplot(2, 6, 2)
        ax2.imshow(rgb)
        for b in yolo_tight:
            x, y = (b[1]-b[3]/2)*img_w, (b[2]-b[4]/2)*img_h
            ax2.add_patch(patches.Rectangle((x, y), b[3]*img_w, b[4]*img_h, lw=2, ec='red', fc='none'))
        ax2.axis('off')
        ax2.set_title("2D Tight bboxes (Visible Parts)", fontsize=16)

        # --- 3. RGB + 2D Loose Bbox ---
        ax3 = fig.add_subplot(2, 6, 3)
        ax3.imshow(rgb)
        for b in yolo_loose:
            x, y = (b[1]-b[3]/2)*img_w, (b[2]-b[4]/2)*img_h
            ax3.add_patch(patches.Rectangle((x, y), b[3]*img_w, b[4]*img_h, lw=2, ec='blue', fc='none'))
        ax3.set_title("2D Loose bboxes (Full Object)", fontsize=16)
        ax3.axis('off')

        # --- 4. 3D Bounding Boxes 投影在 RGB 圖像上 ---
        ax4 = fig.add_subplot(2, 6, 4)
        ax4.imshow(rgb)

        pos = np.array(pose_data['pos'])
        q_scipy = pose_data['orient'] 
        
        T_world_to_base = np.eye(4)
        T_world_to_base[:3, :3] = R.from_quat(q_scipy).as_matrix()
        T_world_to_base[:3, 3] = pos
        
        T_world_to_cam = T_world_to_base @ self.T_base_to_cam
        T_cam_to_world_cv = np.linalg.inv(T_world_to_cam)
        
        for box in yolo_3d:
            cls_id, bx, by, bz, l, w, h, yaw = box
            
            dx, dy, dz = l / 2.0, w / 2.0, h / 2.0
            x_c = [dx, dx, -dx, -dx, dx, dx, -dx, -dx]
            y_c = [dy, -dy, -dy, dy, dy, -dy, -dy, dy]
            z_c = [dz, dz, dz, dz, -dz, -dz, -dz, -dz]
            
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            x_w = [bx + x*cos_y - y*sin_y for x, y in zip(x_c, y_c)]
            y_w = [by + x*sin_y + y*cos_y for x, y in zip(x_c, y_c)]
            z_w = [bz + z for z in z_c]
            
            pts_world = np.vstack((x_w, y_w, z_w, np.ones(8)))
            
            pts_cam = T_cam_to_world_cv @ pts_world
            
            x_cv = pts_cam[0, :]
            y_cv = pts_cam[1, :]
            z_cv = pts_cam[2, :]
            
            if np.mean(z_cv) < 0.1: 
                continue
            z_cv = np.maximum(z_cv, 1e-5)
            
            u = (self.K[0, 0] * x_cv / z_cv) + self.K[0, 2]
            v = (self.K[1, 1] * y_cv / z_cv) + self.K[1, 2]
            
            lines = [[0,1], [1,2], [2,3], [3,0], 
                        [4,5], [5,6], [6,7], [7,4], 
                        [0,4], [1,5], [2,6], [3,7]] 
            
            for edge in lines:
                ax4.plot([u[edge[0]], u[edge[1]]], [v[edge[0]], v[edge[1]]], color='#39FF14', linewidth=1.5)

        ax4.set_xlim(0, rgb.size[0])
        ax4.set_ylim(rgb.size[1], 0) 
        ax4.axis('off')
        ax4.set_title("3D Bounding Boxes (Projected)", fontsize=16)

        # --- 5. Instance Segmentation ---
        ax5 = fig.add_subplot(2, 6, 5)
        inst_data = instance_raw.item()['data'] if isinstance(instance_raw, np.ndarray) and instance_raw.dtype == object else instance_raw
        if inst_data is not None: 
            ax5.imshow(inst_data.astype(np.float32), cmap='nipy_spectral')
        ax5.axis('off')
        ax5.set_title("Instance Segmentation", fontsize=16)

        # --- 6. Depth Map ---
        ax6 = fig.add_subplot(2, 6, 6)
        if depth is not None:
            ax6.imshow(np.clip(depth, 2, 30), cmap='viridis')
        ax6.axis('off')
        ax6.set_title("Depth Map (2~30m)", fontsize=16)

        # --- 7. LiDAR Top-down ---
        ax7 = fig.add_subplot(2, 6, 7)
        if lidar is not None:
            pts = lidar['data'] if isinstance(lidar, dict) else lidar
            ax7.scatter(pts[:,0], pts[:,1], s=1, c=pts[:,2], cmap='viridis', alpha=0.5)
            ax7.set_xlim(-10,10); ax7.set_ylim(-10,10); ax7.set_aspect('equal')
            ax7.grid(True, alpha=0.3)
        ax7.set_title("LiDAR Point Cloud (m)", fontsize=16)

        # --- 8. LiDAR Point Cloud ---
        ax8 = fig.add_subplot(2, 6, 8)
        ax8.imshow(rgb, alpha=0) 
        ax8.set_facecolor('black')
                
        T_lidar_to_base = np.array(self.lidar_info["lidar_to_base"]["data"]).reshape(4, 4)
        
        pc = lidar['data'] if isinstance(lidar, dict) else lidar
        
        pc = pc[:, :3] 
        
        pts_lidar = np.hstack((pc, np.ones((pc.shape[0], 1))))
        
        T_cam_to_base = np.linalg.inv(self.T_base_to_cam)
        T_cam_to_lidar = T_cam_to_base @ T_lidar_to_base
        
        pts_cam = (T_cam_to_lidar @ pts_lidar.T).T
        x_cv, y_cv, z_cv = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
        
        valid = z_cv > 0.2
        x_cv, y_cv, z_cv = x_cv[valid], y_cv[valid], z_cv[valid]
        
        u = (self.K[0, 0] * x_cv / z_cv) + self.K[0, 2]
        v = (self.K[1, 1] * y_cv / z_cv) + self.K[1, 2]
        
        img_valid = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
        u, v, depth_c = u[img_valid], v[img_valid], z_cv[img_valid]
        
        ax8.scatter(u, v, c=depth_c, cmap='jet', s=1.0, alpha=1.0, vmin=2, vmax=30)

        ax8.set_xlim(0, img_w)
        ax8.set_ylim(img_h, 0) 
        ax8.margins(0)
        ax8.axis('off')
        ax8.set_title("LiDAR Points Projected on RGB (2~30m)", fontsize=16)

        # --- 9. 2D BEV Occupancy ---
        ax9 = fig.add_subplot(2, 6, 9)
        if occ_2d is not None: 
            ax9.imshow(occ_2d.T, origin='lower', cmap='binary', extent=[-5, 5, -5, 5])
        ax9.grid(True, alpha=0.3)
        ax9.set_title("2D BEV Occupancy (m)", fontsize=16)

        # --- 10. 3D Voxel Occupancy ---
        ax10 = fig.add_subplot(2, 6, 10, projection='3d')
        if occ_3d is not None:
            filled = (occ_3d == 1)

            v_size = 0.2
            half = 5.0
            
            x_idx, y_idx, z_idx = np.indices(np.array(occ_3d.shape) + 1)
            x_m = (x_idx * v_size) - half
            y_m = (y_idx * v_size) - half
            z_m = (z_idx * v_size) - 0.5
            
            ax10.voxels(x_m, y_m, z_m, filled, facecolors='#1f77b480', edgecolor='k', lw=0.1)
            
            ax10.set_xlim(-half, half)
            ax10.set_ylim(-half, half)
            ax10.set_zlim(-0.5, 4.5)
            ax10.view_init(elev=30, azim=-75)

        ax10.set_title("3D Voxel Occupancy (m)", fontsize=16)

        # --- 11. IMU Data ---
        ax11 = fig.add_subplot(2, 6, 11)
        imu_obj = imu_raw.item() if hasattr(imu_raw, 'item') else imu_raw
        
        if isinstance(imu_obj, dict):
            accel = imu_obj.get("lin_acc", [0.0, 0.0, 0.0])
            gyro = imu_obj.get("ang_vel", [0.0, 0.0, 0.0])
            
            raw_vals = [*accel, *gyro]
            vals = [0.0 if abs(v) < 1e-4 else v for v in raw_vals]
            
            colors = ['#ff9999']*3 + ['#66b3ff']*3
            bars = ax11.bar(['Ax','Ay','Az','Gx','Gy','Gz'], vals, color=colors, edgecolor='black', linewidth=0.5)
            
            ax11.set_ylim(-10, 10)  
            ax11.spines['bottom'].set_position('zero')
            
            for bar, val in zip(bars, vals):
                yval = bar.get_height()
                offset = 0.5 if val >= 0 else -0.5
                va = 'bottom' if val >= 0 else 'top'
                
                ax11.text(bar.get_x() + bar.get_width()/2.0, yval + offset, 
                         f"{val:.1f}", ha='center', va=va, 
                         fontsize=10, fontweight='bold', color='#333333')
                             
        ax11.set_title("IMU Data", fontsize=16, pad=20)
        ax11.grid(axis='y', linestyle='--', alpha=0.4)
        
        ax11.spines['top'].set_visible(False)
        ax11.spines['right'].set_visible(False)

        # --- 12. World Pose & Sector FOV ---
        ax12 = fig.add_subplot(2, 6, 12)
        ax12.imshow(self.map_img, cmap='gray', extent=self.map_extent, origin='lower', zorder=0)

        if pose_data:
            t_robot = np.array(pose_data['pos'])
            q_robot = pose_data['orient']
            
            r_robot = R.from_quat(q_robot)
            robot_yaw_deg = r_robot.as_euler('zyx', degrees=True)[0]
            
            fov_half = 89.0 / 2.0
            fov_wedge = Wedge((t_robot[0], t_robot[1]), r=2.0, 
                            theta1=robot_yaw_deg - fov_half, theta2=robot_yaw_deg + fov_half,
                            fc='#ffaa00', ec='#cc8800', alpha=0.4, zorder=10, label='Camera FOV')
            ax12.add_patch(fov_wedge)
            
            ax12.scatter(t_robot[0], t_robot[1], c='darkred', s=30, edgecolors='white', zorder=13, label='Jackal Robot')

        ax12.set_aspect('equal')
        ax12.grid(True, which='both', color='gray', linestyle='--', alpha=0.3)
        ax12.legend(loc='lower right', fontsize=8)
        ax12.set_title("Top-down Pose (m) & Camera FOV", fontsize=16)

        # Save the figure
        out_name = f"{fid}.jpg"
        save_path = os.path.join(temp_dir, out_name) if temp_dir else f"frame_{fid}.jpg"
        plt.savefig(save_path, dpi=80)
        plt.close(fig)
        return save_path

def make_video(image_folder, output_path, fps=5):
    """
    使用 FFmpeg glob 模式合成影片，解決編號不連續或起始值不為 0 的問題。
    """
    # 檢查是否有圖片
    images = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    if not images:
        print("Error: No images found in temp folder.")
        return

    print(f"Directly rendering video via FFmpeg glob to {output_path}...")
    
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-pattern_type', 'glob', 
        '-i', os.path.join(image_folder, '*.jpg'),
        '-vcodec', 'libx264',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        output_path
    ]

    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Video saved successfully: {output_path}")
    
    shutil.rmtree(image_folder)
    print(f"Cleaned up temp folder: {image_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=str, default=None, help="Specific Frame ID (e.g. 0003166). If None, enters Video Mode.")
    parser.add_argument("--base_dir", type=str, default="./Dataset")
    parser.add_argument("--scene", type=str, default="full_warehouse_v1")
    parser.add_argument("--condition", type=str, default="single_optimal")
    parser.add_argument("--robot", type=str, default="Jackal_Single")
    parser.add_argument("--workers", type=int, default=mp.cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    target_path = os.path.expanduser(os.path.join(args.base_dir, args.scene, args.condition, args.robot))
    vis = HeadlessVisualizer(target_path)

    if args.frame:
        print(f"Single Frame Mode: {args.frame}")
        vis.render_frame(args.frame)
    else:
        print("Video Mode: Processing all frames...")
        rgb_files = sorted(glob.glob(os.path.join(target_path, "data/rgb/*.png")))
        fids = [os.path.splitext(os.path.basename(f))[0] for f in rgb_files]
        
        temp_dir = "temp_render"
        os.makedirs(temp_dir, exist_ok=True)

        with mp.Pool(args.workers) as pool:
            func = partial(vis.render_frame, temp_dir=temp_dir)
            list(tqdm(pool.imap(func, fids), total=len(fids), desc="Rendering Frames", ncols=80, leave=False))

        video_name = f"{vis.condition}_{vis.robot_name}.mp4"
        make_video(temp_dir, video_name)