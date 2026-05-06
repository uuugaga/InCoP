#!/usr/bin/env python3
"""Convert Isaac Sim Jackal exports to an OpenCOOD OPV2V-like layout.

The source dataset is expected to look like:

    scene/condition/case/Jackal_R1/
      camera_info.yaml
      lidar_info.yaml
      data/lidar/<timestamp>.pcd
      data/rgb/<timestamp>.png
      data/pose.csv or data/pose/<timestamp>.npy
      data/depth/<timestamp>.png or label/depth/<timestamp>.npy
      label/detection/3d/<timestamp>.txt or label/detection_raw/3d_raw/<timestamp>.txt

The output layout is:

    output/<split>/<scenario>/<cav_id>/<timestamp>.yaml
    output/<split>/<scenario>/<cav_id>/<timestamp>.pcd
    output/<split>/<scenario>/<cav_id>/<timestamp>_camera0.png
    ...

Depth images are written to the OpenCOOD-compatible mirror path where the first
"OPV2V" in the output path is replaced by "OPV2V_Hetero".
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - user environment check
    raise SystemExit("NumPy is required. Run this with the same env as OpenCOOD.") from exc

try:
    import yaml
except ImportError as exc:  # pragma: no cover - user environment check
    raise SystemExit("PyYAML is required. Run this with the same env as OpenCOOD.") from exc

try:
    from PIL import Image, ImageDraw
except ImportError as exc:  # pragma: no cover - user environment check
    raise SystemExit("Pillow is required for RGB/depth conversion.") from exc


DEFAULT_CAMERA_COUNT = 1
DEFAULT_IMAGE_SIZE = (1280, 800)  # width, height of the current IsaacSim camera export
VISIBILITY_SIZE = 256
VISIBILITY_RESOLUTION = 0.39
DEFAULT_VISIBILITY_RANGE = (0.0, -25.0, 25.0, 25.0)  # x_min, y_min, x_max, y_max in lidar frame
DEFAULT_LIDAR_FILTER_RANGE = (0.0, -22.4, -3.0, 22.4, 22.4, 1.0)
DEFAULT_LIDAR_FILTER_AUG_ROTATION_RANGE = (-0.78539816, 0.78539816)
DEFAULT_LIDAR_FILTER_AUG_ROTATION_SAMPLES = 17
DEFAULT_LIDAR_FILTER_AUG_SCALES = (0.95, 1.0, 1.05)
DEFAULT_OBJECT_LIST = Path("isaac_sim/map/object_list.json")
DEFAULT_DETECTION_CLASS = "object"
DEDUP_CENTER_DISTANCE = 0.35
DEDUP_DIM_REL_DIFF = 0.35
DEDUP_YAW_DIFF_DEG = 20.0
SOURCE_SPLIT_FILES = {
    "dual_case_distance": "path_case_distance_split.json",
    "dual_case_shadow": "path_case_shadow_split.json",
}
DEFAULT_KEEP_CLASSES = (
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
)

# Isaac Sim's lidar export frame for this Jackal setup is not the OPV2V lidar
# frame. In the raw lidar frame, x is right, y is forward, z is up.
# OpenCOOD/OPV2V BEV here follows the ROS/base convention: x forward, y left,
# z up, matching isaac_sim/README.md.
OPENCOOD_LIDAR_FROM_ISAAC_LIDAR = np.asarray(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)
BASE_FROM_OPENCOOD_LIDAR_ROT = np.eye(3, dtype=float)


class NoAliasSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


@dataclass(frozen=True)
class RobotExport:
    name: str
    path: Path


@dataclass(frozen=True)
class ScenarioExport:
    name: str
    path: Path
    robots: Tuple[RobotExport, ...]


def has_robot_export(path: Path) -> bool:
    return (
        (path / "camera_info.yaml").is_file()
        and (path / "lidar_info.yaml").is_file()
        and (path / "data" / "lidar").is_dir()
        and ((path / "data" / "pose.csv").is_file() or (path / "data" / "pose").is_dir())
    )


def natural_key(value: str) -> List[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def scenario_name_from_path(source: Path, scenario_path: Path) -> str:
    rel = scenario_path.relative_to(source)
    return "__".join(rel.parts).replace(" ", "_")


def scenario_name_from_source_split(scene_name: str, condition: str, case_file: str) -> str:
    return f"{scene_name}__{condition}__{Path(case_file).stem}"


def load_source_split(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def source_split_files_available(source: Path) -> bool:
    for scene_dir in source.iterdir():
        if not scene_dir.is_dir():
            continue
        for split_file in SOURCE_SPLIT_FILES.values():
            if (scene_dir / split_file).is_file():
                return True
    return False


def source_split_scenario_names(source: Path, split_key: str, require_all: bool) -> Tuple[set[str], Dict[str, Dict[str, int]]]:
    selected_names: set[str] = set()
    groups: Dict[str, Dict[str, int]] = {}
    found_any = False

    for scene_dir in sorted((p for p in source.iterdir() if p.is_dir()), key=lambda p: natural_key(p.name)):
        for condition, split_file in SOURCE_SPLIT_FILES.items():
            split_path = scene_dir / split_file
            if not split_path.is_file():
                if require_all:
                    raise SystemExit(f"Missing source split JSON: {split_path}")
                continue

            found_any = True
            source_split = load_source_split(split_path)
            group = f"{scene_dir.name}__{condition}"
            groups[group] = {
                key: len(source_split.get(key, []))
                for key in ("train", "validate", "test")
            }
            selected_names.update(
                scenario_name_from_source_split(scene_dir.name, condition, case_file)
                for case_file in source_split.get(split_key, [])
            )

    if require_all and not found_any:
        raise SystemExit(f"No source split JSON files found under {source}")
    return selected_names, groups


def discover_scenarios(source: Path) -> List[ScenarioExport]:
    scenarios: List[ScenarioExport] = []
    for root, dirnames, _ in os.walk(source):
        root_path = Path(root)
        robot_dirs = [root_path / d for d in dirnames if has_robot_export(root_path / d)]
        if not robot_dirs:
            continue
        robots = tuple(
            RobotExport(path=p, name=p.name)
            for p in sorted(robot_dirs, key=lambda p: natural_key(p.name))
        )
        scenarios.append(
            ScenarioExport(
                name=scenario_name_from_path(source, root_path),
                path=root_path,
                robots=robots,
            )
        )
        dirnames[:] = []
    return sorted(scenarios, key=lambda s: natural_key(s.name))


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_pose_npy(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True).item()
    return np.asarray(data["pos"], dtype=float), np.asarray(data["orient"], dtype=float)


def load_pose_csv(path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if not path.is_file():
        return poses
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            timestamp = str(row["timestamp_ms"]).strip()
            pos = np.asarray([float(row["x"]), float(row["y"]), float(row["z"])], dtype=float)
            quat = np.asarray([float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])], dtype=float)
            poses[timestamp] = (pos, quat)
            if timestamp.isdigit():
                poses[timestamp.zfill(7)] = (pos, quat)
    return poses


def robot_pose_timestamps(robot: RobotExport) -> set:
    pose_csv = robot.path / "data" / "pose.csv"
    if pose_csv.is_file():
        return set(load_pose_csv(pose_csv).keys())
    return {p.stem for p in (robot.path / "data" / "pose").glob("*.npy")}


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ],
        dtype=float,
    )


def transform_from_pose(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    tfm = np.eye(4, dtype=float)
    tfm[:3, :3] = quat_xyzw_to_matrix(quat_xyzw)
    tfm[:3, 3] = pos
    return tfm


def matrix_from_yaml_entry(entry: dict) -> np.ndarray:
    return np.asarray(entry["data"], dtype=float).reshape(entry["rows"], entry["cols"])


def matrix_from_yaml_aliases(config: dict, *keys: str) -> np.ndarray:
    for key in keys:
        if key in config:
            return matrix_from_yaml_entry(config[key])
    raise KeyError(f"None of the calibration keys exist: {keys}")


def opencood_lidar_from_base_lidar(t_base_lidar_raw: np.ndarray) -> np.ndarray:
    t_base_lidar = np.eye(4, dtype=float)
    t_base_lidar[:3, :3] = BASE_FROM_OPENCOOD_LIDAR_ROT
    t_base_lidar[:3, 3] = t_base_lidar_raw[:3, 3]
    return t_base_lidar


def opencood_camera_from_base_camera_optical(t_base_camera_optical: np.ndarray) -> np.ndarray:
    # The Isaac Sim source export already stores the camera in optical frame:
    # x right, y down, z forward. OpenCOOD's Lift-Splat-Shoot expects the same
    # optical frame for the camera extrinsic, so only the translation/pose chain
    # should change here.
    return t_base_camera_optical.copy()


def tfm_to_opencood_pose(tfm: np.ndarray) -> List[float]:
    yaw = math.degrees(math.atan2(tfm[1, 0], tfm[0, 0]))
    roll = math.degrees(math.atan2(-tfm[2, 1], tfm[2, 2]))
    pitch = math.degrees(math.atan2(tfm[2, 0], math.sqrt(tfm[2, 1] ** 2 + tfm[2, 2] ** 2)))
    x, y, z = tfm[:3, 3]
    return [float(x), float(y), float(z), float(roll), float(yaw), float(pitch)]


def x1_to_x2_from_tfm(t_world_x1: np.ndarray, t_world_x2: np.ndarray) -> np.ndarray:
    return np.linalg.solve(t_world_x2, t_world_x1)


def common_timestamps(robots: Sequence[RobotExport]) -> List[str]:
    timestamp_sets = []
    for robot in robots:
        lidar_ts = {p.stem for p in (robot.path / "data" / "lidar").glob("*.pcd")}
        pose_ts = robot_pose_timestamps(robot)
        rgb_ts = {p.stem for p in (robot.path / "data" / "rgb").glob("*.png")}
        timestamp_sets.append(lidar_ts & pose_ts & rgb_ts)
    if not timestamp_sets:
        return []
    return sorted(set.intersection(*timestamp_sets), key=natural_key)


def copy_or_link(src: Path, dst: Path, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        rel_src = os.path.relpath(src.resolve(), dst.parent.resolve())
        os.symlink(rel_src, dst)


def read_pcd_header(path: Path) -> Tuple[List[str], int, str, int]:
    header_lines: List[str] = []
    fields: List[str] = []
    points = 0
    data_type = ""
    with path.open("rb") as f:
        while True:
            raw = f.readline()
            if not raw:
                break
            line = raw.decode("ascii", errors="ignore").strip()
            header_lines.append(line)
            if line.startswith("FIELDS"):
                fields = line.split()[1:]
            elif line.startswith("POINTS"):
                points = int(line.split()[1])
            elif line.startswith("DATA"):
                data_type = line.split()[1]
                break
        offset = f.tell()
    return fields, points, data_type, offset


def load_xyz_from_pcd(path: Path) -> np.ndarray:
    fields, points, data_type, offset = read_pcd_header(path)
    if fields[:3] != ["x", "y", "z"]:
        raise ValueError(f"Unsupported PCD fields in {path}: {fields}")

    if data_type == "binary":
        raw = np.fromfile(path, dtype=np.float32, offset=offset)
        if points <= 0:
            points = raw.size // len(fields)
        arr = raw.reshape(points, len(fields))
    elif data_type == "ascii":
        with path.open("rb") as f:
            f.seek(offset)
            arr = np.loadtxt(f)
        arr = np.atleast_2d(arr)
    else:
        raise ValueError(f"Unsupported PCD DATA type in {path}: {data_type}")

    return arr[:, :3].astype(np.float32, copy=False)


def write_xyzrgb_pcd(
    src: Path,
    dst: Path,
    copy_files: bool,
    xyz_rotation: Optional[np.ndarray] = None,
) -> None:
    fields, _, _, _ = read_pcd_header(src)
    if "rgb" in fields and xyz_rotation is None:
        copy_or_link(src, dst, copy_files)
        return

    xyz = load_xyz_from_pcd(src)
    if xyz_rotation is not None:
        xyz = xyz @ xyz_rotation.T
    dst.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.full((xyz.shape[0], 1), 1.0, dtype=np.float32)
    xyzi = np.hstack([xyz, rgb])
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z rgb\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {xyz.shape[0]}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {xyz.shape[0]}\n"
        "DATA binary\n"
    )
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    with dst.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(xyzi.astype(np.float32, copy=False).tobytes())


def save_rgb_camera_set(src: Path, dst_prefix: Path, copy_files: bool, image_size: Tuple[int, int], camera_count: int) -> None:
    img = Image.open(src).convert("RGB")
    if img.size == image_size and not copy_files:
        for idx in range(camera_count):
            copy_or_link(src, dst_prefix.with_name(f"{dst_prefix.name}_camera{idx}.png"), copy_files=False)
        return

    if img.size != image_size:
        img = img.resize(image_size, Image.BILINEAR)
    dst_prefix.parent.mkdir(parents=True, exist_ok=True)
    for idx in range(camera_count):
        img.save(dst_prefix.with_name(f"{dst_prefix.name}_camera{idx}.png"))


def depth_mirror_path(output_root: Path) -> Path:
    text = str(output_root)
    if "OPV2V" in text:
        return Path(text.replace("OPV2V", "OPV2V_Hetero", 1))
    return output_root.with_name(output_root.name + "_Hetero")


def resolve_depth_path(robot_path: Path, timestamp: str) -> Optional[Path]:
    candidates = [
        robot_path / "data" / "depth" / f"{timestamp}.png",
        robot_path / "label" / "depth" / f"{timestamp}.npy",
    ]
    return next((path for path in candidates if path.is_file()), None)


def load_depth_meters(src: Path) -> np.ndarray:
    if src.suffix.lower() == ".npy":
        depth = np.load(src)
        return np.squeeze(depth).astype(np.float32)

    img = Image.open(src)
    depth = np.asarray(img).astype(np.float32)
    # Current Isaac Sim depth PNGs are 16-bit millimeters. Older 8-bit files
    # already encode meter values in the 0..255 range expected by this HEAL fork.
    if depth.dtype != np.uint8 and float(np.nanmax(depth)) > 255.0:
        depth /= 1000.0
    return np.squeeze(depth).astype(np.float32)


def save_depth_camera_set(src: Optional[Path], dst_prefix: Path, image_size: Tuple[int, int], camera_count: int) -> bool:
    if src is None or not src.is_file():
        return False
    depth = load_depth_meters(src)
    depth = np.nan_to_num(depth, nan=255.0, posinf=255.0, neginf=0.0)
    depth_u8 = np.clip(depth, 0.0, 255.0).astype(np.uint8)
    img = Image.fromarray(depth_u8, mode="L")
    if img.size != image_size:
        img = img.resize(image_size, Image.NEAREST)
    dst_prefix.parent.mkdir(parents=True, exist_ok=True)
    for idx in range(camera_count):
        img.save(dst_prefix.with_name(f"{dst_prefix.name}_depth{idx}.png"))
    return True


def world_to_lidar(t_world_lidar: np.ndarray, xyz: Sequence[float]) -> np.ndarray:
    point = np.asarray([xyz[0], xyz[1], xyz[2], 1.0], dtype=float)
    return np.linalg.solve(t_world_lidar, point)[:3]


def object_center_lidar(t_world_lidar: np.ndarray, object_content: dict) -> np.ndarray:
    location = np.asarray(object_content["location"], dtype=float)
    center = np.asarray(object_content.get("center", [0.0, 0.0, 0.0]), dtype=float)
    return world_to_lidar(t_world_lidar, location + center)


def in_visibility_range(point_lidar: np.ndarray, visibility_range: Tuple[float, float, float, float]) -> bool:
    x_min, y_min, x_max, y_max = visibility_range
    return x_min <= point_lidar[0] <= x_max and y_min <= point_lidar[1] <= y_max


def object_footprint_lidar(t_world_lidar: np.ndarray, object_content: dict) -> List[Tuple[int, int]]:
    location = np.asarray(object_content["location"], dtype=float)
    center = np.asarray(object_content.get("center", [0.0, 0.0, 0.0]), dtype=float)
    extent = np.asarray(object_content["extent"], dtype=float)
    yaw = math.radians(float(object_content["angle"][1]))
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rot = np.asarray([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=float)
    local = np.asarray(
        [
            [extent[0], extent[1]],
            [extent[0], -extent[1]],
            [-extent[0], -extent[1]],
            [-extent[0], extent[1]],
        ],
        dtype=float,
    )
    world_xy = local @ rot.T + (location + center)[:2]
    pixels: List[Tuple[int, int]] = []
    for x, y in world_xy:
        lidar_xyz = world_to_lidar(t_world_lidar, [x, y, location[2] + center[2]])
        py = 127 - int(lidar_xyz[0] / VISIBILITY_RESOLUTION)
        # OpenCOOD lidar y is positive to the left, while image x grows right.
        px = 127 - int(lidar_xyz[1] / VISIBILITY_RESOLUTION)
        pixels.append((px, py))
    return pixels


def save_bev_visibility(
    dst: Path,
    vehicles: Dict[str, dict],
    t_world_lidar: np.ndarray,
    visibility_range: Tuple[float, float, float, float],
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("L", (VISIBILITY_SIZE, VISIBILITY_SIZE), color=0)
    draw = ImageDraw.Draw(img)
    for object_content in vehicles.values():
        if not in_visibility_range(object_center_lidar(t_world_lidar, object_content), visibility_range):
            continue
        footprint = object_footprint_lidar(t_world_lidar, object_content)
        in_bounds = [
            (max(0, min(VISIBILITY_SIZE - 1, px)), max(0, min(VISIBILITY_SIZE - 1, py)))
            for px, py in footprint
        ]
        draw.polygon(in_bounds, fill=255)
        cx = int(sum(px for px, _ in in_bounds) / len(in_bounds))
        cy = int(sum(py for _, py in in_bounds) / len(in_bounds))
        draw.rectangle((cx - 1, cy - 1, cx + 1, cy + 1), fill=255)
    img.save(dst)


def yaw_diff_deg(a: float, b: float) -> float:
    diff = abs((a - b + 180.0) % 360.0 - 180.0)
    return min(diff, abs(diff - 180.0))


def dimension_rel_diff(a: Sequence[float], b: Sequence[float]) -> float:
    a_np = np.asarray(a, dtype=float)
    b_np = np.asarray(b, dtype=float)
    denom = np.maximum(np.maximum(np.abs(a_np), np.abs(b_np)), 1e-6)
    return float(np.max(np.abs(a_np - b_np) / denom))


def is_duplicate_object(candidate: dict, kept: dict) -> bool:
    if candidate["class_name"] != kept["class_name"]:
        return False
    center_dist = np.linalg.norm(np.asarray(candidate["location"]) - np.asarray(kept["location"]))
    if center_dist > DEDUP_CENTER_DISTANCE:
        return False
    if dimension_rel_diff(candidate["dimensions"], kept["dimensions"]) > DEDUP_DIM_REL_DIFF:
        return False
    return yaw_diff_deg(candidate["yaw_deg"], kept["yaw_deg"]) <= DEDUP_YAW_DIFF_DEG


def deduplicate_objects(candidates: List[dict]) -> List[dict]:
    candidates = sorted(
        candidates,
        key=lambda item: (
            item["class_name"],
            -item["volume"],
            item["source_index"],
        ),
    )
    kept: List[dict] = []
    for candidate in candidates:
        if any(is_duplicate_object(candidate, existing) for existing in kept):
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda item: item["source_index"])


def resolve_label_path(robot_path: Path, timestamp: str) -> Path:
    candidates = [
        robot_path / "label" / "detection" / "3d" / f"{timestamp}.txt",
        robot_path / "label" / "detection_raw" / "3d_raw" / f"{timestamp}.txt",
    ]
    return next((path for path in candidates if path.is_file()), candidates[0])


def load_objects(
    label_path: Path,
    classes: Optional[set],
    min_box_size: float,
    detection_class: str,
    class_to_id: Dict[str, int],
    asset_to_class: Dict[str, str],
) -> Dict[str, dict]:
    candidates: List[dict] = []
    if not label_path.is_file():
        return {}
    with label_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            raw_class = parts[0]
            if "," in raw_class:
                continue
            class_name = normalize_class_name(raw_class, asset_to_class)
            if classes is not None and raw_class not in classes and class_name not in classes:
                continue
            _, raw_class_id, x, y, z, length, width, height, yaw = parts
            l, w, h = float(length), float(width), float(height)
            if min(l, w, h) < min_box_size:
                continue
            yaw_deg = math.degrees(float(yaw))
            location = [float(x), float(y), float(z)]
            dimensions = [l, w, h]
            try:
                class_id = int(raw_class_id)
            except ValueError:
                class_id = class_to_id.get(class_name, -1)
            candidates.append({
                "source_index": idx,
                "class_name": class_name,
                "class_id": class_to_id.get(class_name, class_id),
                "raw_class": raw_class,
                "location": location,
                "dimensions": dimensions,
                "extent": [l / 2.0, w / 2.0, h / 2.0],
                "yaw_deg": yaw_deg,
                "volume": l * w * h,
            })

    vehicles: Dict[str, dict] = {}
    for candidate in deduplicate_objects(candidates):
        vehicles[f"obj_{candidate['source_index']:06d}"] = {
            "angle": [0.0, candidate["yaw_deg"], 0.0],
            "center": [0.0, 0.0, 0.0],
            "extent": candidate["extent"],
            "location": candidate["location"],
            "class": detection_class,
            "obj_type": detection_class,
            "class_name": candidate["class_name"],
            "class_id": candidate["class_id"],
            "raw_class": candidate["raw_class"],
        }
    return vehicles


def make_frame_yaml(
    robot: RobotExport,
    timestamp: str,
    lidar_pose: List[float],
    lidar_tfm: np.ndarray,
    camera_tfm: np.ndarray,
    camera_info: dict,
    vehicles: Dict[str, dict],
    camera_count: int,
    image_size: Tuple[int, int],
) -> dict:
    intrinsic = np.asarray(camera_info["camera_matrix"]["data"], dtype=float).reshape(3, 3)
    source_width = float(camera_info.get("image_width", image_size[0]))
    source_height = float(camera_info.get("image_height", image_size[1]))
    intrinsic[0, 0] *= image_size[0] / source_width
    intrinsic[0, 2] *= image_size[0] / source_width
    intrinsic[1, 1] *= image_size[1] / source_height
    intrinsic[1, 2] *= image_size[1] / source_height
    intrinsic = intrinsic.tolist()
    camera_pose = tfm_to_opencood_pose(camera_tfm)
    camera_to_lidar = x1_to_x2_from_tfm(camera_tfm, lidar_tfm).tolist()
    camera_entry = {
        "cords": camera_pose,
        "extrinsic": camera_to_lidar,
        "intrinsic": intrinsic,
    }
    frame = {
        f"camera{i}": {
            "cords": list(camera_entry["cords"]),
            "extrinsic": camera_entry["extrinsic"],
            "intrinsic": camera_entry["intrinsic"],
        }
        for i in range(camera_count)
    }
    frame.update({
        "ego_speed": 0.0,
        "lidar_pose": lidar_pose,
        "true_ego_pos": lidar_pose,
        "predicted_ego_pos": lidar_pose,
        "vehicles": vehicles,
    })
    return frame


def convert_scenario(
    scenario: ScenarioExport,
    output_split: Path,
    depth_split: Path,
    copy_files: bool,
    image_size: Tuple[int, int],
    limit_frames: Optional[int],
    skip_initial_frames: int,
    classes: Optional[set],
    min_box_size: float,
    detection_class: str,
    class_to_id: Dict[str, int],
    asset_to_class: Dict[str, str],
    camera_count: int,
    visibility_range: Tuple[float, float, float, float],
    min_lidar_points_in_range: int,
    lidar_filter_range: Tuple[float, float, float, float, float, float],
    lidar_filter_aug_rotation_range: Tuple[float, float],
    lidar_filter_aug_rotation_samples: int,
    lidar_filter_aug_scales: Sequence[float],
    lidar_filter_aug_flip_x: bool,
) -> Tuple[int, Dict[str, str], int]:
    timestamps = common_timestamps(scenario.robots)
    if skip_initial_frames:
        timestamps = timestamps[skip_initial_frames:]
    if limit_frames is not None:
        timestamps = timestamps[:limit_frames]
    timestamps, skipped_lidar_sparse = filter_timestamps_by_lidar_points(
        scenario,
        timestamps,
        min_lidar_points_in_range,
        lidar_filter_range,
        lidar_filter_aug_rotation_range,
        lidar_filter_aug_rotation_samples,
        lidar_filter_aug_scales,
        lidar_filter_aug_flip_x,
    )

    assignment: Dict[str, str] = {}
    cav_ids = {robot.name: str(idx) for idx, robot in enumerate(scenario.robots)}

    for robot in scenario.robots:
        cav_id = cav_ids[robot.name]
        assignment[cav_id] = "m1"
        pose_by_timestamp = load_pose_csv(robot.path / "data" / "pose.csv")
        camera_info = load_yaml(robot.path / "camera_info.yaml")
        lidar_info = load_yaml(robot.path / "lidar_info.yaml")
        t_base_camera_optical = matrix_from_yaml_aliases(
            camera_info,
            "camera_optical_to_base",
            "camera_to_base",
        )
        t_base_lidar_raw = matrix_from_yaml_aliases(
            lidar_info,
            "lidar_sensor_to_base",
            "lidar_to_base",
        )
        t_base_camera = opencood_camera_from_base_camera_optical(t_base_camera_optical)
        t_base_lidar = opencood_lidar_from_base_lidar(t_base_lidar_raw)

        out_cav = output_split / scenario.name / cav_id
        depth_cav = depth_split / scenario.name / cav_id
        out_cav.mkdir(parents=True, exist_ok=True)

        for timestamp in timestamps:
            if pose_by_timestamp:
                pos, quat = pose_by_timestamp[timestamp]
            else:
                pos, quat = load_pose_npy(robot.path / "data" / "pose" / f"{timestamp}.npy")
            t_world_base = transform_from_pose(pos, quat)
            t_world_lidar = t_world_base @ t_base_lidar
            t_world_camera = t_world_base @ t_base_camera
            lidar_pose = tfm_to_opencood_pose(t_world_lidar)
            vehicles = load_objects(
                resolve_label_path(robot.path, timestamp),
                classes,
                min_box_size,
                detection_class,
                class_to_id,
                asset_to_class,
            )

            write_xyzrgb_pcd(
                robot.path / "data" / "lidar" / f"{timestamp}.pcd",
                out_cav / f"{timestamp}.pcd",
                copy_files,
                xyz_rotation=OPENCOOD_LIDAR_FROM_ISAAC_LIDAR,
            )
            save_rgb_camera_set(robot.path / "data" / "rgb" / f"{timestamp}.png", out_cav / timestamp, copy_files, image_size, camera_count)
            save_depth_camera_set(resolve_depth_path(robot.path, timestamp), depth_cav / timestamp, image_size, camera_count)
            save_bev_visibility(out_cav / f"{timestamp}_bev_visibility.png", vehicles, t_world_lidar, visibility_range)

            frame_yaml = make_frame_yaml(
                robot=robot,
                timestamp=timestamp,
                lidar_pose=lidar_pose,
                lidar_tfm=t_world_lidar,
                camera_tfm=t_world_camera,
                camera_info=camera_info,
                vehicles=vehicles,
                camera_count=camera_count,
                image_size=image_size,
            )
            with (out_cav / f"{timestamp}.yaml").open("w", encoding="utf-8") as f:
                yaml.dump(frame_yaml, f, Dumper=NoAliasSafeDumper, sort_keys=False)

    return len(timestamps), assignment, skipped_lidar_sparse


def parse_classes(raw: str) -> Optional[set]:
    if raw.lower() in {"", "all", "*"}:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def default_keep_classes_arg() -> str:
    return "all"


def parse_csv(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_image_size(raw: str) -> Tuple[int, int]:
    width, height = raw.lower().split("x", 1)
    return int(width), int(height)


def parse_visibility_range(raw: str) -> Tuple[float, float, float, float]:
    values = [float(item.strip()) for item in raw.split(",")]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("--visibility-range must be x_min,y_min,x_max,y_max")
    if values[0] >= values[2] or values[1] >= values[3]:
        raise argparse.ArgumentTypeError("--visibility-range min values must be smaller than max values")
    return values[0], values[1], values[2], values[3]


def parse_lidar_filter_range(raw: str) -> Tuple[float, float, float, float, float, float]:
    values = [float(item.strip()) for item in raw.split(",")]
    if len(values) != 6:
        raise argparse.ArgumentTypeError("--lidar-filter-range must be x_min,y_min,z_min,x_max,y_max,z_max")
    if values[0] >= values[3] or values[1] >= values[4] or values[2] >= values[5]:
        raise argparse.ArgumentTypeError("--lidar-filter-range min values must be smaller than max values")
    return values[0], values[1], values[2], values[3], values[4], values[5]


def parse_float_pair(raw: str, arg_name: str) -> Tuple[float, float]:
    values = [float(item.strip()) for item in raw.split(",")]
    if len(values) != 2:
        raise argparse.ArgumentTypeError(f"{arg_name} must have two comma-separated values")
    if values[0] > values[1]:
        raise argparse.ArgumentTypeError(f"{arg_name} min value must be <= max value")
    return values[0], values[1]


def parse_lidar_filter_aug_rotation_range(raw: str) -> Tuple[float, float]:
    return parse_float_pair(raw, "--lidar-filter-aug-rotation-range")


def parse_float_list(raw: str) -> List[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float")
    return values


def count_points_in_range(points: np.ndarray, lidar_range: Tuple[float, float, float, float, float, float]) -> int:
    x_min, y_min, z_min, x_max, y_max, z_max = lidar_range
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        & (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    return int(mask.sum())


def rotation_angles(rotation_range: Tuple[float, float], sample_count: int) -> np.ndarray:
    if sample_count <= 0:
        return np.asarray([0.0], dtype=float)
    if sample_count == 1:
        return np.asarray([0.5 * (rotation_range[0] + rotation_range[1])], dtype=float)
    return np.linspace(rotation_range[0], rotation_range[1], sample_count)


def transformed_min_points_in_range(
    points: np.ndarray,
    lidar_range: Tuple[float, float, float, float, float, float],
    aug_rotation_range: Tuple[float, float],
    aug_rotation_samples: int,
    aug_scales: Sequence[float],
    aug_flip_x: bool,
) -> int:
    angles = rotation_angles(aug_rotation_range, aug_rotation_samples)
    flips = (False, True) if aug_flip_x else (False,)
    min_count = None

    for flip_x in flips:
        base = points.copy()
        if flip_x:
            base[:, 1] *= -1.0
        for scale in aug_scales:
            scaled = base * scale
            for angle in angles:
                transformed = scaled.copy()
                c, s = np.cos(angle), np.sin(angle)
                x = scaled[:, 0] * c - scaled[:, 1] * s
                y = scaled[:, 0] * s + scaled[:, 1] * c
                transformed[:, 0] = x
                transformed[:, 1] = y
                count = count_points_in_range(transformed, lidar_range)
                min_count = count if min_count is None else min(min_count, count)
                if min_count == 0:
                    return 0

    return 0 if min_count is None else min_count


def filter_timestamps_by_lidar_points(
    scenario: ScenarioExport,
    timestamps: Sequence[str],
    min_points: int,
    lidar_range: Tuple[float, float, float, float, float, float],
    aug_rotation_range: Tuple[float, float],
    aug_rotation_samples: int,
    aug_scales: Sequence[float],
    aug_flip_x: bool,
) -> Tuple[List[str], int]:
    if min_points <= 0:
        return list(timestamps), 0

    kept: List[str] = []
    skipped = 0
    for timestamp in timestamps:
        keep_timestamp = True
        for robot in scenario.robots:
            points = load_xyz_from_pcd(robot.path / "data" / "lidar" / f"{timestamp}.pcd")
            points = points @ OPENCOOD_LIDAR_FROM_ISAAC_LIDAR.T
            if transformed_min_points_in_range(
                points,
                lidar_range,
                aug_rotation_range,
                aug_rotation_samples,
                aug_scales,
                aug_flip_x,
            ) < min_points:
                keep_timestamp = False
                break
        if keep_timestamp:
            kept.append(timestamp)
        else:
            skipped += 1
    return kept, skipped


def load_class_map(object_list_path: Optional[Path]) -> Tuple[Dict[str, int], Dict[str, str], List[str]]:
    if object_list_path is None:
        return {}, {}, []
    if not object_list_path.is_file():
        raise SystemExit(f"Object list JSON not found: {object_list_path}")

    with object_list_path.open("r", encoding="utf-8") as f:
        object_list = json.load(f)

    class_names = list(object_list.keys())
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    asset_to_class: Dict[str, str] = {}
    for class_name, asset_names in object_list.items():
        asset_to_class[class_name] = class_name
        for asset_name in asset_names:
            asset_to_class[asset_name] = class_name

    aliases = {
        "plant": "potted_plant",
        "cone": "traffic_cone",
        "sign": "wet_floor_sign",
    }
    for alias, class_name in aliases.items():
        if class_name in class_to_id:
            asset_to_class[alias] = class_name

    return class_to_id, asset_to_class, class_names


def normalize_class_name(raw_class: str, asset_to_class: Dict[str, str]) -> str:
    for token in [part.strip() for part in raw_class.split(",") if part.strip()]:
        if token in asset_to_class:
            return asset_to_class[token]
    return asset_to_class.get(raw_class, raw_class)


def normalize_source_root(source: Path) -> Path:
    if discover_scenarios(source):
        return source
    nested = source / "Dataset"
    if nested.is_dir() and discover_scenarios(nested):
        return nested
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("HEAL/dataset/IsaacSimDataset/Dataset"))
    parser.add_argument("--output", type=Path, default=Path("HEAL/dataset/IsaacSimOPV2V"))
    parser.add_argument("--split", choices=["train", "validate", "test"], default="train")
    parser.add_argument("--limit-scenarios", type=int, default=None)
    parser.add_argument("--scenario-names", default=None, help="Comma-separated scenario names to convert after discovery, e.g. case_1,case_7.")
    parser.add_argument("--include-substrings", default=None, help="Comma-separated substrings; keep scenarios whose generated name contains any of them.")
    parser.add_argument(
        "--source-splits",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Read path_case_distance_split.json/path_case_shadow_split.json from "
            "the IsaacSim source and use --split to select scenarios. 'auto' uses "
            "them when present; 'on' requires them; 'off' ignores them."
        ),
    )
    parser.add_argument("--split-manifest", type=Path, default=None, help="JSON manifest with train/validate/test scenario name lists.")
    parser.add_argument("--split-key", default=None, help="Split name to read from --split-manifest. Defaults to --split.")
    parser.add_argument("--shuffle-scenarios", action="store_true", help="Shuffle discovered scenarios before applying --limit-scenarios.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --shuffle-scenarios.")
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--skip-initial-frames", type=int, default=0)
    parser.add_argument("--copy-files", action="store_true", help="Copy/resize RGB images instead of symlinking originals.")
    parser.add_argument("--classes", default=default_keep_classes_arg(), help="Comma-separated class names to keep from 3D labels, or 'all'. Defaults to all because the current label/detection/3d files are already filtered.")
    parser.add_argument("--object-list", type=Path, default=DEFAULT_OBJECT_LIST, help="JSON class map used to store class_id/class_name metadata for future multi-class training.")
    parser.add_argument("--detection-class", default=DEFAULT_DETECTION_CLASS, help="Single class name used by the current class-agnostic training target.")
    parser.add_argument("--image-size", default="1280x800", help="Output camera/depth PNG size as WIDTHxHEIGHT.")
    parser.add_argument("--min-box-size", type=float, default=0.05, help="Drop boxes with any dimension smaller than this many meters.")
    parser.add_argument("--camera-count", type=int, default=DEFAULT_CAMERA_COUNT, help="Number of camera/depth views to export. IsaacSim currently has only camera0.")
    parser.add_argument(
        "--visibility-range",
        type=parse_visibility_range,
        default=DEFAULT_VISIBILITY_RANGE,
        help="Rectangular visible BEV area in lidar frame: x_min,y_min,x_max,y_max. Default keeps front 25m and +/-25m left/right.",
    )
    parser.add_argument(
        "--min-lidar-points-in-range",
        type=int,
        default=5,
        help="Skip a timestamp if any CAV has fewer than this many lidar points inside --lidar-filter-range. Use 0 to disable.",
    )
    parser.add_argument(
        "--lidar-filter-range",
        type=parse_lidar_filter_range,
        default=DEFAULT_LIDAR_FILTER_RANGE,
        help="Range used by --min-lidar-points-in-range: x_min,y_min,z_min,x_max,y_max,z_max.",
    )
    parser.add_argument(
        "--lidar-filter-aug-rotation-range",
        type=parse_lidar_filter_aug_rotation_range,
        default=DEFAULT_LIDAR_FILTER_AUG_ROTATION_RANGE,
        help="Rotation range, in radians, to stress-test sparse lidar filtering. Default mirrors IsaacSim PointPillar training aug.",
    )
    parser.add_argument(
        "--lidar-filter-aug-rotation-samples",
        type=int,
        default=DEFAULT_LIDAR_FILTER_AUG_ROTATION_SAMPLES,
        help="Number of rotations sampled inside --lidar-filter-aug-rotation-range while filtering sparse lidar.",
    )
    parser.add_argument(
        "--lidar-filter-aug-scales",
        type=parse_float_list,
        default=DEFAULT_LIDAR_FILTER_AUG_SCALES,
        help="Comma-separated scales to stress-test sparse lidar filtering. Default mirrors IsaacSim PointPillar training aug.",
    )
    parser.add_argument(
        "--disable-lidar-filter-aug-flip-x",
        action="store_true",
        help="Disable x-axis flip stress test in sparse lidar filtering.",
    )
    args = parser.parse_args()

    source = normalize_source_root(args.source.resolve())
    output = args.output
    output_split = output / args.split
    depth_output = depth_mirror_path(output)
    depth_split = depth_output / args.split
    image_size = parse_image_size(args.image_size)
    classes = parse_classes(args.classes)
    object_list_path = args.object_list.resolve() if args.object_list is not None else None
    class_to_id, asset_to_class, class_names = load_class_map(object_list_path)

    scenarios = discover_scenarios(source)
    include_substrings = parse_csv(args.include_substrings)
    if include_substrings:
        scenarios = [
            scenario for scenario in scenarios
            if any(token in scenario.name for token in include_substrings)
        ]
    if args.source_splits != "off" and args.split_manifest is not None:
        raise SystemExit("--source-splits and --split-manifest are mutually exclusive. Use --source-splits off to use a manifest.")
    if args.source_splits != "off":
        require_source_splits = args.source_splits == "on"
        if require_source_splits or source_split_files_available(source):
            selected_names, split_groups = source_split_scenario_names(
                source,
                args.split,
                require_all=require_source_splits,
            )
            if not selected_names:
                raise SystemExit(f"No scenarios listed for split '{args.split}' in source split JSON files under {source}")
            scenarios = [scenario for scenario in scenarios if scenario.name in selected_names]
            print(
                f"Using source split JSON for split '{args.split}': "
                f"{len(selected_names)} scenario name(s), groups={split_groups}",
                flush=True,
            )
    if args.split_manifest is not None:
        split_key = args.split_key or args.split
        with args.split_manifest.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        selected_names = set(manifest.get(split_key, []))
        if not selected_names:
            raise SystemExit(f"No scenarios listed for split '{split_key}' in {args.split_manifest}")
        scenarios = [scenario for scenario in scenarios if scenario.name in selected_names]
    if args.scenario_names:
        selected_names = {item.strip() for item in args.scenario_names.split(",") if item.strip()}
        scenarios = [scenario for scenario in scenarios if scenario.name in selected_names]
    if args.shuffle_scenarios:
        rng = random.Random(args.seed)
        rng.shuffle(scenarios)
    if args.limit_scenarios is not None:
        scenarios = scenarios[: args.limit_scenarios]
    if not scenarios:
        raise SystemExit(f"No Isaac Sim scenarios found under {source}")

    assignment_path = output / "heter_modality_assign.json"
    if assignment_path.is_file():
        with assignment_path.open("r", encoding="utf-8") as f:
            assignment_json: Dict[str, Dict[str, str]] = json.load(f)
    else:
        assignment_json = {}
    frame_count = 0
    for scenario in scenarios:
        nframes, assignment, skipped_lidar_sparse = convert_scenario(
            scenario,
            output_split,
            depth_split,
            copy_files=args.copy_files,
            image_size=image_size,
            limit_frames=args.limit_frames,
            skip_initial_frames=args.skip_initial_frames,
            classes=classes,
            min_box_size=args.min_box_size,
            detection_class=args.detection_class,
            class_to_id=class_to_id,
            asset_to_class=asset_to_class,
            camera_count=args.camera_count,
            visibility_range=args.visibility_range,
            min_lidar_points_in_range=args.min_lidar_points_in_range,
            lidar_filter_range=args.lidar_filter_range,
            lidar_filter_aug_rotation_range=args.lidar_filter_aug_rotation_range,
            lidar_filter_aug_rotation_samples=args.lidar_filter_aug_rotation_samples,
            lidar_filter_aug_scales=args.lidar_filter_aug_scales,
            lidar_filter_aug_flip_x=not args.disable_lidar_filter_aug_flip_x,
        )
        assignment_json[scenario.name] = assignment
        frame_count += nframes * len(scenario.robots)
        skip_msg = f", skipped {skipped_lidar_sparse} sparse lidar frame(s)" if skipped_lidar_sparse else ""
        print(f"{scenario.name}: {len(scenario.robots)} CAV(s), {nframes} shared frame(s){skip_msg}", flush=True)

    # A split conversion can be run after, or in parallel with, another split.
    # Reload right before writing so heter_modality_assign.json accumulates all
    # converted scenarios instead of keeping only the last finished split.
    if assignment_path.is_file():
        with assignment_path.open("r", encoding="utf-8") as f:
            latest_assignment_json: Dict[str, Dict[str, str]] = json.load(f)
        latest_assignment_json.update(assignment_json)
        assignment_json = latest_assignment_json

    output.mkdir(parents=True, exist_ok=True)
    with assignment_path.open("w", encoding="utf-8") as f:
        json.dump(assignment_json, f, indent=2, sort_keys=True)
    if class_names:
        class_map_path = output / "isaacsim_class_map.json"
        with class_map_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "training_mode": "class_agnostic",
                    "detection_class": args.detection_class,
                    "class_names": class_names,
                    "class_to_id": class_to_id,
                    "object_list": str(object_list_path),
                },
                f,
                indent=2,
                sort_keys=True,
            )

    print(f"\nWrote OPV2V-like data: {output_split}", flush=True)
    print(f"Wrote depth mirror:    {depth_split}", flush=True)
    print(f"Wrote assignment JSON: {assignment_path}", flush=True)
    if class_names:
        print(f"Wrote class metadata: {class_map_path}", flush=True)
    print(f"Total CAV frames:      {frame_count}", flush=True)


if __name__ == "__main__":
    main()
