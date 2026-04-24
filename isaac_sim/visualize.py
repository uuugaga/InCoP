import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import yaml
import csv
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
import subprocess
import shutil
from frame_utils import load_camera_optical_to_base, load_lidar_sensor_to_base

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import open3d as o3d
except ImportError:
    o3d = None

class HeadlessVisualizer:
    DEFAULT_PANELS = [
        "rgb",
        "bbox_tight",
        "bbox_loose",
        "bbox_3d",
        "semantic",
        "depth",
        "lidar_bev",
        "lidar_projection",
        "imu",
        "pose",
    ]

    PANEL_TITLES = {
        "rgb": "RGB Image",
        "bbox_tight": "2D Tight Bboxes",
        "bbox_loose": "2D Loose Bboxes",
        "bbox_3d": "3D Bounding Boxes",
        "semantic": "Semantic Segmentation",
        "depth": "Depth Map (2-30m)",
        "lidar_bev": "LiDAR Top-down",
        "lidar_projection": "LiDAR Projection on RGB",
        "imu": "IMU Data",
        "pose": "Top-down Pose & Camera FOV",
    }

    def __init__(self, root_path, panels=None, show_text=True):
        self.root = os.path.normpath(os.path.expanduser(root_path))
        self.cmap = matplotlib.colormaps.get_cmap('tab10')
        self.panels = list(panels or self.DEFAULT_PANELS)
        self.show_text = show_text
        base_colors = (plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:, :3] * 255).astype(np.uint8)
        self.semantic_palette = np.zeros((256, 3), dtype=np.uint8)
        for idx in range(1, len(self.semantic_palette)):
            self.semantic_palette[idx] = base_colors[(idx - 1) % len(base_colors)]

        self.scene_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(self.root))))
        self.condition = os.path.basename(os.path.dirname(os.path.dirname(self.root)))
        self.robot_name = os.path.basename(self.root)

        yaml_path = os.path.join(self.root, "camera_info.yaml")
        with open(yaml_path, 'r') as f:
            cam_info = yaml.safe_load(f)
        self.K = np.array(cam_info["camera_matrix"]["data"]).reshape(3, 3)
        self.T_camera_to_base = load_camera_optical_to_base(cam_info, yaml_path)
        self.T_base_to_camera = np.linalg.inv(self.T_camera_to_base)

        lidar_yaml_path = os.path.join(self.root, "lidar_info.yaml")
        with open(lidar_yaml_path, 'r') as f:
            self.lidar_info = yaml.safe_load(f)

        map_key = self.scene_name
        for suffix in ("_v1", "_v2"):
            if map_key.endswith(suffix):
                map_key = map_key[:-len(suffix)]

        map_dir = "/home/uuugaga/_my_code/map/cropped_maps"
        map_png_path = os.path.join(map_dir, f"{map_key}.png")
        map_yaml_path = os.path.join(map_dir, f"{map_key}.yaml")
        
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
            self.has_map = False
            self.map_img = None
            self.map_extent = None
            if "pose" in self.panels:
                self.panels.remove("pose")
                print("Map files not found. Skipping pose map panel.")

    def load_npy(self, sub_path):
        path = os.path.join(self.root, sub_path)
        return np.load(path, allow_pickle=True) if os.path.exists(path) else None

    @staticmethod
    def _unwrap_npy(value):
        if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
            value = value.item()
        if isinstance(value, dict):
            value = value.get("data")
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return arr

    def render_semantic_rgb(self, semantic):
        if semantic is None:
            return None
        seg = np.asarray(semantic)
        if seg.ndim != 2:
            return None
        seg = np.where(seg >= 0, seg, 0).astype(np.int64)
        return self.semantic_palette[seg % len(self.semantic_palette)]
    
    def load_pcd(self, sub_path):
        path = os.path.join(self.root, sub_path)
        if not os.path.exists(path): return None
        if o3d is not None:
            pcd = o3d.io.read_point_cloud(path)
            return np.asarray(pcd.points)
        return self.load_binary_pcd(path)

    @staticmethod
    def load_binary_pcd(path):
        fields = []
        points = 0
        data_start = 0

        with open(path, "rb") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                data_start = f.tell()
                text = line.decode("ascii", errors="ignore").strip()
                if text.startswith("FIELDS"):
                    fields = text.split()[1:]
                elif text.startswith("POINTS"):
                    points = int(text.split()[1])
                elif text.startswith("DATA"):
                    if text.split()[1].lower() != "binary":
                        return None
                    break

            if points <= 0 or not fields:
                return None
            raw = f.read()

        arr = np.frombuffer(raw, dtype=np.float32)
        width = len(fields)
        if width <= 0 or arr.size < points * width:
            return None
        return arr[:points * width].reshape(points, width)[:, :3]

    def load_yolo(self, sub_path):
        path = os.path.join(self.root, sub_path)
        if not os.path.exists(path): return []
        with open(path, 'r') as f:
            boxes = []
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if not self._is_number(parts[0]):
                    parts = parts[1:]
                if len(parts) >= 5:
                    boxes.append(list(map(float, parts[:5])))
            return boxes
        
    def load_yolo_3d(self, sub_path):
        path = os.path.join(self.root, sub_path)
        if not os.path.exists(path): return []
        
        yolo_3d_data = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if not self._is_number(parts[0]):
                    parts = parts[1:]
                if len(parts) >= 8:
                    parsed_line = [int(float(parts[0]))] + list(map(float, parts[1:8]))
                    yolo_3d_data.append(parsed_line)
        return yolo_3d_data

    @staticmethod
    def _is_number(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def load_yolo_first(self, sub_paths):
        for sub_path in sub_paths:
            labels = self.load_yolo(sub_path)
            if labels:
                return labels
        return []

    def load_yolo_3d_first(self, sub_paths):
        for sub_path in sub_paths:
            labels = self.load_yolo_3d(sub_path)
            if labels:
                return labels
        return []

    def load_csv_record(self, csv_name, fid):
        path = os.path.join(self.root, "data", csv_name)
        if not os.path.exists(path): return None

        target_ts = str(int(fid))
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("timestamp_ms") == target_ts:
                    return row
        return None

    def load_pose_record(self, fid):
        row = self.load_csv_record("pose.csv", fid)
        if row is None: return None

        return {
            "pos": np.array([float(row["x"]), float(row["y"]), float(row["z"])]),
            "orient": np.array([float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])])
        }

    def load_imu_record(self, fid):
        row = self.load_csv_record("imu.csv", fid)
        if row is None: return None

        return {
            "lin_acc": [
                float(row["linear_acceleration_x"]),
                float(row["linear_acceleration_y"]),
                float(row["linear_acceleration_z"])
            ],
            "ang_vel": [
                float(row["angular_velocity_x"]),
                float(row["angular_velocity_y"]),
                float(row["angular_velocity_z"])
            ],
            "orientation": [
                float(row["orientation_x"]),
                float(row["orientation_y"]),
                float(row["orientation_z"]),
                float(row["orientation_w"])
            ]
        }

    @staticmethod
    def _auto_grid(num_panels):
        cols = int(np.ceil(np.sqrt(num_panels)))
        rows = int(np.ceil(num_panels / cols))
        return rows, cols

    @staticmethod
    def _pose_to_world_to_camera(pose_data, T_camera_to_base):
        T_world_to_base = np.eye(4)
        T_world_to_base[:3, :3] = R.from_quat(pose_data["orient"]).as_matrix()
        T_world_to_base[:3, 3] = pose_data["pos"]
        return np.linalg.inv(T_world_to_base @ T_camera_to_base)

    @staticmethod
    def _set_image_axis(ax, img_w, img_h):
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)
        ax.margins(0)
        ax.axis("off")

    @staticmethod
    def _strip_axis_text(ax):
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(
            labelbottom=False,
            labelleft=False,
            labelright=False,
            labeltop=False,
            bottom=False,
            left=False,
            right=False,
            top=False,
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    def render_frame(self, fid, temp_dir=None, output_path=None):
        rgb = Image.open(os.path.join(self.root, f"data/rgb/{fid}.png"))
        depth_path = os.path.join(self.root, f"data/depth/{fid}.png")
        depth = np.array(Image.open(depth_path), dtype=np.float32) / 1000.0 if os.path.exists(depth_path) else None
        lidar = self.load_pcd(f"data/lidar/{fid}.pcd")
        yolo_tight = self.load_yolo_first([
            f"label/detection/2d_tight/{fid}.txt",
            f"label/detection_raw/2d_tight_raw/{fid}.txt",
        ])
        yolo_loose = self.load_yolo_first([
            f"label/detection/2d_loose/{fid}.txt",
            f"label/detection_raw/2d_loose_raw/{fid}.txt",
        ])
        yolo_3d = self.load_yolo_3d_first([
            f"label/detection/3d/{fid}.txt",
            f"label/detection_raw/3d_raw/{fid}.txt",
        ])
        pose_data = self.load_pose_record(fid)
        imu_obj = self.load_imu_record(fid)
        semantic = self._unwrap_npy(self.load_npy(f"label/segmentation/semantic/{fid}.npy"))

        img_w, img_h = rgb.size
        rows, cols = self._auto_grid(len(self.panels))
        fig = plt.figure(
            figsize=(cols * 5.4, rows * 3.8),
            constrained_layout=self.show_text,
            facecolor='#fdfdfd' if self.show_text else 'black',
        )

        if self.show_text:
            fid_str = str(fid)
            formatted_fid = f"{fid_str[:-3]}.{fid_str[-3:]}"
            fig.suptitle(
                f"{self.scene_name} | {self.condition} | {self.robot_name} | Frame: {formatted_fid} (s)",
                fontsize=18,
                fontweight='bold',
                color='#333333',
            )

        def draw_rgb(ax):
            ax.imshow(rgb)
            ax.axis('off')

        def draw_2d_boxes(ax, boxes, color):
            ax.imshow(rgb)
            for b in boxes:
                x, y = (b[1] - b[3] / 2) * img_w, (b[2] - b[4] / 2) * img_h
                ax.add_patch(patches.Rectangle((x, y), b[3] * img_w, b[4] * img_h, lw=2, ec=color, fc='none'))
            ax.axis('off')

        def draw_3d_boxes(ax):
            ax.imshow(rgb)
            if pose_data is None:
                self._set_image_axis(ax, img_w, img_h)
                return

            T_world_to_camera = self._pose_to_world_to_camera(pose_data, self.T_camera_to_base)
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7],
            ]

            for box in yolo_3d:
                _, bx, by, bz, l, w, h, yaw = box
                dx, dy, dz = l / 2.0, w / 2.0, h / 2.0
                corners = np.array([
                    [dx, dy, dz], [dx, -dy, dz], [-dx, -dy, dz], [-dx, dy, dz],
                    [dx, dy, -dz], [dx, -dy, -dz], [-dx, -dy, -dz], [-dx, dy, -dz],
                ])
                rot = np.array([
                    [np.cos(yaw), -np.sin(yaw), 0.0],
                    [np.sin(yaw), np.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0],
                ])
                corners_world = (rot @ corners.T).T + np.array([bx, by, bz])
                pts_cam = T_world_to_camera @ np.hstack((corners_world, np.ones((8, 1)))).T
                z = pts_cam[2, :]
                if np.mean(z) < 0.1:
                    continue
                z = np.maximum(z, 1e-5)
                u = (self.K[0, 0] * pts_cam[0, :] / z) + self.K[0, 2]
                v = (self.K[1, 1] * pts_cam[1, :] / z) + self.K[1, 2]
                for edge in edges:
                    ax.plot([u[edge[0]], u[edge[1]]], [v[edge[0]], v[edge[1]]], color='#39FF14', linewidth=1.5)
            self._set_image_axis(ax, img_w, img_h)

        def draw_depth(ax):
            if depth is not None:
                ax.imshow(np.clip(depth, 2, 30), cmap='viridis')
            ax.axis('off')

        def draw_semantic(ax):
            semantic_rgb = self.render_semantic_rgb(semantic)
            if semantic_rgb is not None:
                ax.imshow(semantic_rgb)
            ax.axis('off')

        def draw_lidar_bev(ax):
            if lidar is not None:
                pts = lidar['data'] if isinstance(lidar, dict) else lidar
                ax.scatter(pts[:, 0], pts[:, 1], s=1, c=pts[:, 2], cmap='viridis', alpha=0.5)
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        def draw_lidar_projection(ax):
            ax.imshow(rgb, alpha=0)
            ax.set_facecolor('black')
            if lidar is not None:
                lidar_yaml_path = os.path.join(self.root, "lidar_info.yaml")
                T_lidar_to_base = load_lidar_sensor_to_base(self.lidar_info, lidar_yaml_path)
                pc = lidar['data'] if isinstance(lidar, dict) else lidar
                pc = pc[:, :3]
                pts_lidar = np.hstack((pc, np.ones((pc.shape[0], 1))))
                pts_cam = (self.T_base_to_camera @ T_lidar_to_base @ pts_lidar.T).T
                x_cv, y_cv, z_cv = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
                valid = z_cv > 0.2
                x_cv, y_cv, z_cv = x_cv[valid], y_cv[valid], z_cv[valid]
                if z_cv.size:
                    u = (self.K[0, 0] * x_cv / z_cv) + self.K[0, 2]
                    v = (self.K[1, 1] * y_cv / z_cv) + self.K[1, 2]
                    img_valid = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
                    ax.scatter(u[img_valid], v[img_valid], c=z_cv[img_valid], cmap='jet', s=1.0, alpha=1.0, vmin=2, vmax=30)
            self._set_image_axis(ax, img_w, img_h)

        def draw_imu(ax):
            if isinstance(imu_obj, dict):
                accel = imu_obj.get("lin_acc", [0.0, 0.0, 0.0])
                gyro = imu_obj.get("ang_vel", [0.0, 0.0, 0.0])
                vals = [0.0 if abs(v) < 1e-4 else v for v in [*accel, *gyro]]
                colors = ['#ff9999'] * 3 + ['#66b3ff'] * 3
                bars = ax.bar(['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz'], vals, color=colors, edgecolor='black', linewidth=0.5)
                ax.set_ylim(-10, 10)
                ax.spines['bottom'].set_position('zero')
                if self.show_text:
                    for bar, val in zip(bars, vals):
                        yval = bar.get_height()
                        offset = 0.5 if val >= 0 else -0.5
                        va = 'bottom' if val >= 0 else 'top'
                        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + offset, f"{val:.1f}", ha='center', va=va, fontsize=9, fontweight='bold', color='#333333')
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        def draw_pose(ax):
            if self.has_map:
                ax.imshow(self.map_img, cmap='gray', extent=self.map_extent, origin='lower', zorder=0)
            if pose_data:
                t_robot = np.array(pose_data['pos'])
                robot_yaw_deg = R.from_quat(pose_data['orient']).as_euler('zyx', degrees=True)[0]
                fov_half = 89.0 / 2.0
                fov_wedge = Wedge(
                    (t_robot[0], t_robot[1]),
                    r=2.0,
                    theta1=robot_yaw_deg - fov_half,
                    theta2=robot_yaw_deg + fov_half,
                    fc='#ffaa00',
                    ec='#cc8800',
                    alpha=0.4,
                    zorder=10,
                    label='Camera FOV',
                )
                ax.add_patch(fov_wedge)
                ax.scatter(t_robot[0], t_robot[1], c='darkred', s=30, edgecolors='white', zorder=13, label='Jackal Robot')
                if self.show_text:
                    ax.legend(loc='lower right', fontsize=8)
            ax.set_aspect('equal')
            ax.grid(True, which='both', color='gray', linestyle='--', alpha=0.3)

        draw_panel = {
            "rgb": draw_rgb,
            "bbox_tight": lambda ax: draw_2d_boxes(ax, yolo_tight, 'red'),
            "bbox_loose": lambda ax: draw_2d_boxes(ax, yolo_loose, 'blue'),
            "bbox_3d": draw_3d_boxes,
            "semantic": draw_semantic,
            "depth": draw_depth,
            "lidar_bev": draw_lidar_bev,
            "lidar_projection": draw_lidar_projection,
            "imu": draw_imu,
            "pose": draw_pose,
        }

        for idx, panel in enumerate(self.panels, start=1):
            ax = fig.add_subplot(rows, cols, idx)
            draw_panel[panel](ax)
            if self.show_text:
                ax.set_title(self.PANEL_TITLES[panel], fontsize=13, pad=8)
            else:
                self._strip_axis_text(ax)

        if not self.show_text:
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        # Save the figure
        out_name = f"{fid}.jpg"
        if output_path:
            save_path = output_path
        elif temp_dir:
            save_path = os.path.join(temp_dir, out_name)
        else:
            save_path = out_name
        plt.savefig(save_path, dpi=200, bbox_inches=None, pad_inches=0 if not self.show_text else 0.1)
        plt.close(fig)
        return save_path

def make_video(image_folder, output_path, fps=10):
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
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-vcodec', 'libx264',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            print(exc.stderr)
        raise
    print(f"Video saved successfully: {output_path}")
    
    shutil.rmtree(image_folder)
    print(f"Cleaned up temp folder: {image_folder}")

def parse_panels(raw_panels):
    if not raw_panels:
        return None

    panels = [item.strip() for item in raw_panels.split(",") if item.strip()]
    valid = set(HeadlessVisualizer.PANEL_TITLES)
    unknown = [panel for panel in panels if panel not in valid]
    if unknown:
        raise ValueError(
            f"Unknown panel(s): {', '.join(unknown)}. "
            f"Valid panels: {', '.join(HeadlessVisualizer.PANEL_TITLES)}"
        )
    if not panels:
        raise ValueError("Panel list cannot be empty.")
    return panels

def find_robot_roots(root_path):
    root = os.path.expanduser(root_path)
    if os.path.exists(os.path.join(root, "camera_info.yaml")):
        return [root]

    robot_roots = []
    for child in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(child) and os.path.exists(os.path.join(child, "camera_info.yaml")):
            robot_roots.append(child)

    if robot_roots:
        return robot_roots

    raise FileNotFoundError(
        "Could not find camera_info.yaml. Pass a robot folder like "
        "'.../case_0/Jackal_R1', or a case folder containing robot subfolders."
    )

def default_video_name(target_path):
    parts = os.path.normpath(target_path).split(os.sep)
    if len(parts) >= 4 and parts[-1].startswith("Jackal_"):
        scene, condition, case_name, robot_name = parts[-4], parts[-3], parts[-2], parts[-1]
        return f"{scene}_{condition}_{case_name}_{robot_name}.mp4"
    return f"{os.path.basename(os.path.normpath(target_path))}.mp4"

def default_frame_name(target_path, fid):
    parts = os.path.normpath(target_path).split(os.sep)
    if len(parts) >= 4 and parts[-1].startswith("Jackal_"):
        scene, condition, case_name, robot_name = parts[-4], parts[-3], parts[-2], parts[-1]
        return f"{scene}_{condition}_{case_name}_{robot_name}_{fid}.jpg"
    return f"{os.path.basename(os.path.normpath(target_path))}_{fid}.jpg"

def output_path(output_folder, filename):
    os.makedirs(output_folder, exist_ok=True)
    return os.path.join(output_folder, filename)

def main(argv=None, force_show_text=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=str, default=None, help="Specific Frame ID (e.g. 0003166). If None, enters Video Mode.")
    parser.add_argument("--root", type=str, required=True, help="Robot dataset folder, e.g. Dataset/hospital_v1/dual_case_shadow/case_0/Jackal_R1")
    parser.add_argument(
        "--panels",
        type=str,
        default=None,
        help="Comma-separated panel keys. Default: rgb,bbox_tight,bbox_loose,bbox_3d,depth,lidar_bev,lidar_projection,imu,pose",
    )
    parser.add_argument(
        "--no_text",
        "--no-text",
        action="store_false",
        dest="show_text",
        help="Render without titles, legends, tick labels, or numeric annotations.",
    )
    parser.add_argument(
        "--output_folder",
        "--output-folder",
        type=str,
        default="visualization_outputs",
        help="Folder for rendered jpg/mp4 outputs.",
    )
    parser.set_defaults(show_text=True)
    args = parser.parse_args(argv)
    if force_show_text is not None:
        args.show_text = force_show_text
    panels = parse_panels(args.panels)

    target_path = os.path.expanduser(args.root)
    output_folder = os.path.normpath(os.path.expanduser(args.output_folder))
    os.makedirs(output_folder, exist_ok=True)
    robot_roots = find_robot_roots(target_path)
    for robot_root in robot_roots:
        vis = HeadlessVisualizer(robot_root, panels=panels, show_text=args.show_text)

        if args.frame:
            print(f"Single Frame Mode: {args.frame} | {vis.robot_name}")
            out_path = output_path(output_folder, default_frame_name(robot_root, args.frame))
            vis.render_frame(args.frame, output_path=out_path)
            print(f"Saved frame: {out_path}")
            continue

        print(f"Video Mode: Processing all frames for {vis.robot_name}...")
        rgb_files = sorted(glob.glob(os.path.join(robot_root, "data/rgb/*.png")))
        fids = [os.path.splitext(os.path.basename(f))[0] for f in rgb_files]
        if not fids:
            print(f"Error: No RGB frames found in {robot_root}")
            continue
        
        temp_dir = os.path.join(output_folder, f"temp_render_{vis.robot_name}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        with mp.Pool(mp.cpu_count()) as pool:
            func = partial(vis.render_frame, temp_dir=temp_dir)
            list(tqdm(pool.imap(func, fids), total=len(fids), desc=f"Rendering {vis.robot_name}", ncols=80, leave=False))

        video_name = output_path(output_folder, default_video_name(robot_root))
        make_video(temp_dir, video_name)


if __name__ == "__main__":
    main()
