#!/usr/bin/env python3
"""Convert Isaac Sim Jackal exports to an OpenCOOD OPV2V-like layout.

The source dataset is expected to look like:

    scene/condition/case/Jackal_R1/
      camera_info.yaml
      lidar_info.yaml
      data/lidar/<timestamp>.pcd
      data/rgb/<timestamp>.png
      data/pose/<timestamp>.npy
      label/depth/<timestamp>.npy
      label/detection_raw/3d_raw/<timestamp>.txt

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
DEFAULT_IMAGE_SIZE = (800, 600)  # width, height used by current OPV2V LSS hypes
VISIBILITY_SIZE = 256
VISIBILITY_RESOLUTION = 0.39
DEFAULT_VISIBILITY_RANGE = (0.0, -25.0, 25.0, 25.0)  # x_min, y_min, x_max, y_max in lidar frame


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
        and (path / "data" / "pose").is_dir()
    )


def natural_key(value: str) -> List[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def scenario_name_from_path(source: Path, scenario_path: Path) -> str:
    rel = scenario_path.relative_to(source)
    return "__".join(rel.parts).replace(" ", "_")


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


def load_pose(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True).item()
    return np.asarray(data["pos"], dtype=float), np.asarray(data["orient"], dtype=float)


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
        pose_ts = {p.stem for p in (robot.path / "data" / "pose").glob("*.npy")}
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


def write_xyzrgb_pcd(src: Path, dst: Path, copy_files: bool) -> None:
    fields, _, _, _ = read_pcd_header(src)
    if "rgb" in fields:
        copy_or_link(src, dst, copy_files)
        return

    xyz = load_xyz_from_pcd(src)
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
    if copy_files:
        img = Image.open(src).convert("RGB")
        if img.size != image_size:
            img = img.resize(image_size, Image.BILINEAR)
        for idx in range(camera_count):
            img.save(dst_prefix.with_name(f"{dst_prefix.name}_camera{idx}.png"))
    else:
        for idx in range(camera_count):
            copy_or_link(src, dst_prefix.with_name(f"{dst_prefix.name}_camera{idx}.png"), copy_files=False)


def depth_mirror_path(output_root: Path) -> Path:
    text = str(output_root)
    if "OPV2V" in text:
        return Path(text.replace("OPV2V", "OPV2V_Hetero", 1))
    return output_root.with_name(output_root.name + "_Hetero")


def save_depth_camera_set(src: Path, dst_prefix: Path, image_size: Tuple[int, int], camera_count: int) -> bool:
    if not src.is_file():
        return False
    depth = np.load(src)
    depth = np.squeeze(depth).astype(np.float32)
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
        px = 127 + int(lidar_xyz[1] / VISIBILITY_RESOLUTION)
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


def load_objects(label_path: Path, classes: Optional[set], min_box_size: float) -> Dict[str, dict]:
    vehicles: Dict[str, dict] = {}
    if not label_path.is_file():
        return vehicles
    with label_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            class_name = parts[0]
            if classes is not None and class_name not in classes:
                continue
            _, _, x, y, z, length, width, height, yaw = parts
            l, w, h = float(length), float(width), float(height)
            if min(l, w, h) < min_box_size:
                continue
            vehicles[f"obj_{idx:06d}"] = {
                "angle": [0.0, math.degrees(float(yaw)), 0.0],
                "center": [0.0, 0.0, 0.0],
                "extent": [l / 2.0, w / 2.0, h / 2.0],
                "location": [float(x), float(y), float(z)],
                "class": class_name,
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
) -> dict:
    intrinsic = np.asarray(camera_info["camera_matrix"]["data"], dtype=float).reshape(3, 3).tolist()
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
    camera_count: int,
    visibility_range: Tuple[float, float, float, float],
) -> Tuple[int, Dict[str, str]]:
    timestamps = common_timestamps(scenario.robots)
    if skip_initial_frames:
        timestamps = timestamps[skip_initial_frames:]
    if limit_frames is not None:
        timestamps = timestamps[:limit_frames]

    assignment: Dict[str, str] = {}
    cav_ids = {robot.name: str(idx) for idx, robot in enumerate(scenario.robots)}

    for robot in scenario.robots:
        cav_id = cav_ids[robot.name]
        assignment[cav_id] = "m1"
        camera_info = load_yaml(robot.path / "camera_info.yaml")
        lidar_info = load_yaml(robot.path / "lidar_info.yaml")
        t_base_camera = matrix_from_yaml_entry(camera_info["camera_to_base"])
        t_base_lidar = matrix_from_yaml_entry(lidar_info["lidar_to_base"])

        out_cav = output_split / scenario.name / cav_id
        depth_cav = depth_split / scenario.name / cav_id
        out_cav.mkdir(parents=True, exist_ok=True)

        for timestamp in timestamps:
            pos, quat = load_pose(robot.path / "data" / "pose" / f"{timestamp}.npy")
            t_world_base = transform_from_pose(pos, quat)
            t_world_lidar = t_world_base @ t_base_lidar
            t_world_camera = t_world_base @ t_base_camera
            lidar_pose = tfm_to_opencood_pose(t_world_lidar)
            vehicles = load_objects(
                robot.path / "label" / "detection_raw" / "3d_raw" / f"{timestamp}.txt",
                classes,
                min_box_size,
            )

            write_xyzrgb_pcd(robot.path / "data" / "lidar" / f"{timestamp}.pcd", out_cav / f"{timestamp}.pcd", copy_files)
            save_rgb_camera_set(robot.path / "data" / "rgb" / f"{timestamp}.png", out_cav / timestamp, copy_files, image_size, camera_count)
            save_depth_camera_set(robot.path / "label" / "depth" / f"{timestamp}.npy", depth_cav / timestamp, image_size, camera_count)
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
            )
            with (out_cav / f"{timestamp}.yaml").open("w", encoding="utf-8") as f:
                yaml.dump(frame_yaml, f, Dumper=NoAliasSafeDumper, sort_keys=False)

    return len(timestamps), assignment


def parse_classes(raw: str) -> Optional[set]:
    if raw.lower() in {"", "all", "*"}:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("HEAL/dataset/IsaacSimDataset"))
    parser.add_argument("--output", type=Path, default=Path("HEAL/dataset/IsaacSimOPV2V"))
    parser.add_argument("--split", choices=["train", "validate", "test"], default="train")
    parser.add_argument("--limit-scenarios", type=int, default=None)
    parser.add_argument("--scenario-names", default=None, help="Comma-separated scenario names to convert after discovery, e.g. case_1,case_7.")
    parser.add_argument("--shuffle-scenarios", action="store_true", help="Shuffle discovered scenarios before applying --limit-scenarios.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --shuffle-scenarios.")
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--skip-initial-frames", type=int, default=0)
    parser.add_argument("--copy-files", action="store_true", help="Copy/resize RGB images instead of symlinking originals.")
    parser.add_argument("--classes", default="all", help="Comma-separated class names to keep from 3D labels, or 'all'.")
    parser.add_argument("--image-size", default="800x600", help="Output camera/depth PNG size as WIDTHxHEIGHT.")
    parser.add_argument("--min-box-size", type=float, default=0.05, help="Drop boxes with any dimension smaller than this many meters.")
    parser.add_argument("--camera-count", type=int, default=DEFAULT_CAMERA_COUNT, help="Number of camera/depth views to export. IsaacSim currently has only camera0.")
    parser.add_argument(
        "--visibility-range",
        type=parse_visibility_range,
        default=DEFAULT_VISIBILITY_RANGE,
        help="Rectangular visible BEV area in lidar frame: x_min,y_min,x_max,y_max. Default keeps front 25m and +/-25m left/right.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output
    output_split = output / args.split
    depth_output = depth_mirror_path(output)
    depth_split = depth_output / args.split
    image_size = parse_image_size(args.image_size)
    classes = parse_classes(args.classes)

    scenarios = discover_scenarios(source)
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
        nframes, assignment = convert_scenario(
            scenario,
            output_split,
            depth_split,
            copy_files=args.copy_files,
            image_size=image_size,
            limit_frames=args.limit_frames,
            skip_initial_frames=args.skip_initial_frames,
            classes=classes,
            min_box_size=args.min_box_size,
            camera_count=args.camera_count,
            visibility_range=args.visibility_range,
        )
        assignment_json[scenario.name] = assignment
        frame_count += nframes * len(scenario.robots)
        print(f"{scenario.name}: {len(scenario.robots)} CAV(s), {nframes} shared frame(s)")

    output.mkdir(parents=True, exist_ok=True)
    with assignment_path.open("w", encoding="utf-8") as f:
        json.dump(assignment_json, f, indent=2, sort_keys=True)

    print(f"\nWrote OPV2V-like data: {output_split}")
    print(f"Wrote depth mirror:    {depth_split}")
    print(f"Wrote assignment JSON: {assignment_path}")
    print(f"Total CAV frames:      {frame_count}")


if __name__ == "__main__":
    main()
