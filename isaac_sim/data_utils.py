import os
import csv
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
import config

class YOLOFormatter:
    OCCLUSION_THRESHOLD = 0.9
    DEBUG_CLASS_ENV = "DATA_UTILS_DEBUG_CLASS"
    DEBUG_ONLY_ENV = "DATA_UTILS_DEBUG_ONLY_CLASS"

    @staticmethod
    def _box_value(box, key, default=None):
        if isinstance(box, dict):
            return box.get(key, default)
        try:
            return box[key]
        except (KeyError, IndexError, TypeError, ValueError):
            return default

    @staticmethod
    def _is_bad_occlusion(box):
        try:
            occ = float(YOLOFormatter._box_value(box, "occlusionRatio", 1.0))
        except (TypeError, ValueError):
            return True
        return (not np.isfinite(occ)) or occ < 0.0 or occ >= YOLOFormatter.OCCLUSION_THRESHOLD

    @staticmethod
    def _class_name(bbox_data, semantic_id):
        labels = bbox_data["info"]["idToLabels"]
        label = labels.get(str(semantic_id), labels.get(int(semantic_id)))
        return label["class"]

    @staticmethod
    def _class_names(class_name):
        if isinstance(class_name, (list, tuple, set)):
            values = class_name
        else:
            values = str(class_name).split(",")
        return [name.strip() for name in values if str(name).strip()]

    @staticmethod
    def _has_multiple_class_names(class_name):
        return len(YOLOFormatter._class_names(class_name)) > 1

    @staticmethod
    def _debug_class():
        return os.environ.get(YOLOFormatter.DEBUG_CLASS_ENV, "").strip()

    @staticmethod
    def _debug_only_class():
        return os.environ.get(YOLOFormatter.DEBUG_ONLY_ENV, "").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _class_matches(class_name, target):
        if not target:
            return False
        names = YOLOFormatter._class_names(class_name)
        return target in names or str(class_name) == target

    @staticmethod
    def _keep_classes():
        return set(getattr(config, "KEEP_CLASSES", []) or [])

    @staticmethod
    def _should_keep_class(class_name):
        keep_classes = YOLOFormatter._keep_classes()
        if not keep_classes:
            return True
        names = YOLOFormatter._class_names(class_name)
        return len(names) == 1 and names[0] in keep_classes

    @staticmethod
    def _debug_metadata_presence(bbox_data, data, source):
        target = YOLOFormatter._debug_class()
        if not target or not isinstance(bbox_data, dict):
            return

        labels = bbox_data.get("info", {}).get("idToLabels", {})
        matching_ids = []
        for semantic_id, label in labels.items():
            class_name = label.get("class") if isinstance(label, dict) else None
            if YOLOFormatter._class_matches(class_name, target):
                matching_ids.append(str(semantic_id))

        if not matching_ids:
            print(f"[data_utils debug] {source}: class '{target}' is not in idToLabels")
            return

        counts = {semantic_id: 0 for semantic_id in matching_ids}
        if data is not None:
            for box in data:
                semantic_id = YOLOFormatter._box_value(box, "semanticId")
                if semantic_id is not None and str(int(semantic_id)) in counts:
                    counts[str(int(semantic_id))] += 1

        print(f"[data_utils debug] {source}: class '{target}' semantic_ids={matching_ids} raw_box_counts={counts}")

    @staticmethod
    def _debug_box(source, class_name, class_id, box, status, reason, extra=""):
        target = YOLOFormatter._debug_class()
        if not YOLOFormatter._class_matches(class_name, target):
            return

        occ = YOLOFormatter._box_value(box, "occlusionRatio", None)
        occ_text = "None" if occ is None else f"{float(occ):.6f}"
        suffix = f" {extra}" if extra else ""
        print(
            f"[data_utils debug] {source}: class={class_name} semanticId={class_id} "
            f"occlusionRatio={occ_text} status={status} reason={reason}{suffix}"
        )

    @staticmethod
    def to_yolo_2d(bbox_data, img_w, img_h):
        lines = []
        data = bbox_data["data"] if isinstance(bbox_data, dict) else bbox_data
        if data is None or data.size == 0: return lines
        YOLOFormatter._debug_metadata_presence(bbox_data, data, "2d")

        for box in data:
            class_id = int(YOLOFormatter._box_value(box, "semanticId"))
            class_name = YOLOFormatter._class_name(bbox_data, class_id)
            if YOLOFormatter._debug_only_class() and not YOLOFormatter._class_matches(class_name, YOLOFormatter._debug_class()):
                continue

            if not YOLOFormatter._should_keep_class(class_name):
                YOLOFormatter._debug_box("2d", class_name, class_id, box, "skip", "not_in_keep_classes")
                continue

            if YOLOFormatter._has_multiple_class_names(class_name):
                YOLOFormatter._debug_box("2d", class_name, class_id, box, "skip", "multiple_class_names")
                continue

            if YOLOFormatter._is_bad_occlusion(box):
                YOLOFormatter._debug_box("2d", class_name, class_id, box, "skip", "bad_occlusion")
                continue
            
            x_min = float(YOLOFormatter._box_value(box, "x_min"))
            y_min = float(YOLOFormatter._box_value(box, "y_min"))
            x_max = float(YOLOFormatter._box_value(box, "x_max"))
            y_max = float(YOLOFormatter._box_value(box, "y_max"))
            
            w = (x_max - x_min) / img_w
            h = (y_max - y_min) / img_h
            cx = (x_min / img_w) + (w / 2.0)
            cy = (y_min / img_h) + (h / 2.0)
            
            cx, cy, w, h = np.clip([cx, cy, w, h], 0.0, 1.0)
            if w == 0.0 or h == 0.0:
                YOLOFormatter._debug_box(
                    "2d",
                    class_name,
                    class_id,
                    box,
                    "skip",
                    "zero_size_after_clip",
                    f"bbox_px=({x_min:.1f},{y_min:.1f},{x_max:.1f},{y_max:.1f})",
                )
                continue
            YOLOFormatter._debug_box(
                "2d",
                class_name,
                class_id,
                box,
                "keep",
                "ok",
                f"bbox_norm=({cx:.6f},{cy:.6f},{w:.6f},{h:.6f})",
            )
            lines.append(f"{class_name} {class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        return lines

    @staticmethod
    def to_yolo_3d(bbox_3d_data):
        lines = []
        data = bbox_3d_data["data"] if isinstance(bbox_3d_data, dict) else bbox_3d_data
        
        if data is None or data.size == 0: 
            return lines
        YOLOFormatter._debug_metadata_presence(bbox_3d_data, data, "3d")

        for box in data:
            class_id = int(YOLOFormatter._box_value(box, "semanticId"))
            class_name = YOLOFormatter._class_name(bbox_3d_data, class_id)
            if YOLOFormatter._debug_only_class() and not YOLOFormatter._class_matches(class_name, YOLOFormatter._debug_class()):
                continue

            if not YOLOFormatter._should_keep_class(class_name):
                YOLOFormatter._debug_box("3d", class_name, class_id, box, "skip", "not_in_keep_classes")
                continue

            if YOLOFormatter._has_multiple_class_names(class_name):
                YOLOFormatter._debug_box("3d", class_name, class_id, box, "skip", "multiple_class_names")
                continue

            if YOLOFormatter._is_bad_occlusion(box):
                YOLOFormatter._debug_box("3d", class_name, class_id, box, "skip", "bad_occlusion")
                continue

            transform_matrix = box['transform']
            
            scale_x = np.linalg.norm(transform_matrix[0, :3])
            scale_y = np.linalg.norm(transform_matrix[1, :3])
            scale_z = np.linalg.norm(transform_matrix[2, :3])

            l = float(box['x_max'] - box['x_min']) * scale_x
            w = float(box['y_max'] - box['y_min']) * scale_y
            h = float(box['z_max'] - box['z_min']) * scale_z

            local_center = np.array([
                (box['x_max'] + box['x_min']) / 2.0,
                (box['y_max'] + box['y_min']) / 2.0,
                (box['z_max'] + box['z_min']) / 2.0,
                1.0
            ])
            
            world_center = local_center @ transform_matrix
            cx, cy, cz = world_center[:3]

            rot_matrix = transform_matrix[:3, :3].copy()

            norms = np.linalg.norm(rot_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0 
            rot_matrix = rot_matrix / norms 
            
            if np.linalg.det(rot_matrix) < 0:
                rot_matrix[:, 2] *= -1
            
            r = R.from_matrix(rot_matrix)
            yaw = r.as_euler('xyz')[2]

            line = f"{class_name} {class_id} {cx:.6f} {cy:.6f} {cz:.6f} {l:.6f} {w:.6f} {h:.6f} {yaw:.6f}"
            YOLOFormatter._debug_box(
                "3d",
                class_name,
                class_id,
                box,
                "keep",
                "ok",
                f"center=({cx:.3f},{cy:.3f},{cz:.3f}) size=({l:.3f},{w:.3f},{h:.3f}) yaw={yaw:.3f}",
            )
            lines.append(line)
            
        return lines

class DataProcessor:
    @staticmethod
    def _extract_id_to_labels(annotation):
        if not isinstance(annotation, dict):
            return {}
        info = annotation.get("info", {})
        id_to_labels = info.get("idToLabels", {})
        return id_to_labels if isinstance(id_to_labels, dict) else {}

    @staticmethod
    def _allowed_label_ids(annotation):
        id_to_labels = DataProcessor._extract_id_to_labels(annotation)
        allowed = set()
        for raw_id, label in id_to_labels.items():
            class_name = label.get("class") if isinstance(label, dict) else None
            if YOLOFormatter._should_keep_class(class_name):
                try:
                    allowed.add(int(raw_id))
                except (TypeError, ValueError):
                    continue
        return allowed

    @staticmethod
    def _filtered_info(annotation, allowed_ids):
        if not isinstance(annotation, dict):
            return annotation
        info = annotation.get("info")
        if not isinstance(info, dict):
            return annotation
        id_to_labels = info.get("idToLabels")
        if not isinstance(id_to_labels, dict):
            return annotation
        filtered = dict(annotation)
        filtered_info = dict(info)
        filtered_info["idToLabels"] = {
            key: value
            for key, value in id_to_labels.items()
            if str(key) in {str(v) for v in allowed_ids}
        }
        filtered["info"] = filtered_info
        return filtered

    @staticmethod
    def _filter_box_annotation(annotation):
        if not isinstance(annotation, dict):
            return annotation
        data = annotation.get("data")
        if data is None:
            return annotation
        allowed_ids = DataProcessor._allowed_label_ids(annotation)
        if not allowed_ids:
            filtered = dict(annotation)
            filtered["data"] = data[:0] if hasattr(data, "__getitem__") else []
            return DataProcessor._filtered_info(filtered, allowed_ids)

        if isinstance(data, np.ndarray):
            mask = np.array([
                int(YOLOFormatter._box_value(box, "semanticId", -1)) in allowed_ids
                for box in data
            ], dtype=bool)
            filtered_data = data[mask]
        else:
            filtered_data = [
                box for box in data
                if int(YOLOFormatter._box_value(box, "semanticId", -1)) in allowed_ids
            ]

        filtered = dict(annotation)
        filtered["data"] = filtered_data
        return DataProcessor._filtered_info(filtered, allowed_ids)

    @staticmethod
    def _filter_segmentation_annotation(annotation):
        if not isinstance(annotation, dict):
            return annotation
        data = annotation.get("data")
        if data is None:
            return annotation
        allowed_ids = DataProcessor._allowed_label_ids(annotation)
        arr = np.array(data, copy=True)
        keep_ids = np.array(sorted(allowed_ids | {0}), dtype=arr.dtype if hasattr(arr, "dtype") else np.int64)
        filtered_arr = np.where(np.isin(arr, keep_ids), arr, 0)
        filtered = dict(annotation)
        filtered["data"] = filtered_arr
        return DataProcessor._filtered_info(filtered, allowed_ids)

    @staticmethod
    def filter_annotations(data):
        filtered = dict(data)
        filtered["tight_bbox"] = DataProcessor._filter_box_annotation(data.get("tight_bbox"))
        filtered["loose_bbox"] = DataProcessor._filter_box_annotation(data.get("loose_bbox"))
        filtered["bbox_3d"] = DataProcessor._filter_box_annotation(data.get("bbox_3d"))
        filtered["semantic"] = DataProcessor._filter_segmentation_annotation(data.get("semantic"))
        return filtered

    @staticmethod
    def _annotation_array(annotation):
        if isinstance(annotation, dict):
            return annotation.get("data")
        return annotation

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
            pc_np = pc_np[:, :4]
        else:
            header += "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
            pc_np = pc_np[:, :3]
            
        header += f"WIDTH {num_points}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        header += f"POINTS {num_points}\nDATA binary\n"
        
        os.makedirs(os.path.dirname(pcd_path), exist_ok=True)
        with open(pcd_path, 'wb') as f:
            f.write(header.encode('ascii'))
            f.write(pc_np.tobytes())

    @staticmethod
    def save_depth_png(depth_m, depth_path):
        depth_m = np.asarray(depth_m, dtype=np.float32)
        valid = np.isfinite(depth_m) & (depth_m > 0.0)

        depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
        depth_mm[valid] = np.clip(np.rint(depth_m[valid] * 1000.0), 0, 65535).astype(np.uint16)

        os.makedirs(os.path.dirname(depth_path), exist_ok=True)
        Image.fromarray(depth_mm).save(depth_path)

    @staticmethod
    def _append_csv_row(csv_path, header, row):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)

    @staticmethod
    def _remove_csv_timestamp(csv_path, timestamp_ms):
        if not os.path.exists(csv_path):
            return 0

        timestamp_ms = str(int(timestamp_ms))
        tmp_path = csv_path + ".tmp"
        removed = 0

        with open(csv_path, newline="") as src, open(tmp_path, "w", newline="") as dst:
            reader = csv.reader(src)
            writer = csv.writer(dst)
            header = next(reader, None)
            if header is not None:
                writer.writerow(header)
            for row in reader:
                if row and row[0] == timestamp_ms:
                    removed += 1
                    continue
                writer.writerow(row)

        os.replace(tmp_path, csv_path)
        return removed

    @staticmethod
    def delete_frame(root, f_idx):
        timestamp_ms = int(f_idx)
        removed = 0
        file_paths = [
            f"data/rgb/{f_idx}.png",
            f"data/lidar/{f_idx}.pcd",
            f"data/lidar/{f_idx}.npy",
            f"data/depth/{f_idx}.png",
            f"label/detection/2d_tight/{f_idx}.txt",
            f"label/detection/2d_loose/{f_idx}.txt",
            f"label/detection/3d/{f_idx}.txt",
            f"label/segmentation/semantic/{f_idx}.npy",
            f"label/occupancy/3d/{f_idx}.npy",
            f"label/occupancy/2d_bev/{f_idx}.npy",
        ]

        for rel_path in file_paths:
            path = os.path.join(root, rel_path)
            if os.path.exists(path):
                os.remove(path)
                removed += 1

        removed += DataProcessor._remove_csv_timestamp(os.path.join(root, "data/pose.csv"), timestamp_ms)
        removed += DataProcessor._remove_csv_timestamp(os.path.join(root, "data/imu.csv"), timestamp_ms)
        return removed

    @staticmethod
    def _vec3(data, key):
        value = data.get(key) if isinstance(data, dict) else None
        if value is None:
            return [0.0, 0.0, 0.0]
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        out = np.zeros(3, dtype=np.float64)
        out[:min(3, arr.size)] = arr[:3]
        return out.tolist()

    @staticmethod
    def _quat_xyzw(data, key):
        value = data.get(key) if isinstance(data, dict) else None
        if value is None:
            return [0.0, 0.0, 0.0, 1.0]
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size < 4:
            return [0.0, 0.0, 0.0, 1.0]
        return arr[:4].tolist()

    @staticmethod
    def save_pose_csv(pose, root, timestamp_ms):
        pos = np.asarray(pose[0], dtype=np.float64).reshape(-1)
        quat = np.asarray(pose[1], dtype=np.float64).reshape(-1)
        row = [timestamp_ms, *pos[:3].tolist(), *quat[:4].tolist()]

        DataProcessor._append_csv_row(
            os.path.join(root, "data/pose.csv"),
            ["timestamp_ms", "x", "y", "z", "qx", "qy", "qz", "qw"],
            row
        )

    @staticmethod
    def save_imu_csv(imu, root, timestamp_ms):
        imu_data = imu if isinstance(imu, dict) else {}
        accel = DataProcessor._vec3(imu_data, "lin_acc")
        gyro = DataProcessor._vec3(imu_data, "ang_vel")
        quat = DataProcessor._quat_xyzw(imu_data, "orientation")

        DataProcessor._append_csv_row(
            os.path.join(root, "data/imu.csv"),
            [
                "timestamp_ms",
                "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
                "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
                "orientation_x", "orientation_y", "orientation_z", "orientation_w"
            ],
            [timestamp_ms, *accel, *gyro, *quat]
        )

    @staticmethod
    def save_frame(data, root, f_idx):
        data = DataProcessor.filter_annotations(data)
        img_h, img_w = data["rgb"].shape[:2]
        timestamp_ms = int(f_idx)
        
        Image.fromarray(data["rgb"][:, :, :3]).save(os.path.join(root, f"data/rgb/{f_idx}.png"))
        
        lidar_data = data["lidar"]["data"] if isinstance(data["lidar"], dict) else data["lidar"]
        DataProcessor.save_pcd(lidar_data, os.path.join(root, f"data/lidar/{f_idx}.pcd"))
        
        DataProcessor.save_pose_csv(data["pose"], root, timestamp_ms)
        DataProcessor.save_imu_csv(data["imu"], root, timestamp_ms)

        labels = {
            "label/detection/2d_tight": YOLOFormatter.to_yolo_2d(data["tight_bbox"], img_w, img_h),
            "label/detection/2d_loose": YOLOFormatter.to_yolo_2d(data["loose_bbox"], img_w, img_h),
            "label/detection/3d": YOLOFormatter.to_yolo_3d(data["bbox_3d"])
        }
        for path, lines in labels.items():
            txt_path = os.path.join(root, f"{path}/{f_idx}.txt")
            with open(txt_path, "w") as f:
                f.write("\n".join(lines))

        DataProcessor.save_depth_png(data["depth"], os.path.join(root, f"data/depth/{f_idx}.png"))
        np.save(
            os.path.join(root, f"label/segmentation/semantic/{f_idx}.npy"),
            DataProcessor._annotation_array(data["semantic"]),
        )

        print(f"Saved frame {f_idx} to {root}")
