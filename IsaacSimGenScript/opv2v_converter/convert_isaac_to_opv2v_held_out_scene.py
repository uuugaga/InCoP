#!/usr/bin/env python3
"""Build RGB/LiDAR IsaacSim OPV2V leave-one-scene-out datasets.

For each held-out scene, source train and test from the other scenes become
output train, source validate from the other scenes becomes output validate,
and every source split from the held-out scene becomes output test.  The
complementary (distance/occlusion) and high-overlap case sets are converted
independently so they cannot be mixed accidentally.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import convert_isaac_to_opv2v as converter


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_PARENT = PROJECT_ROOT.parent / "dataset"
HELD_OUT_SCENES = {
    "warehouse": "full_warehouse_v1",
    "hospital": "hospital_v1",
    "office": "office_v1",
}
OUTPUT_SPLITS = ("train", "validate", "test")
SOURCE_SPLITS = ("train", "validate", "test")
CASE_SET_SPLIT_FILES = {
    "complementary": dict(converter.SOURCE_SPLIT_FILES),
    "highoverlap": {
        "dual_case_highoverlap": "path_case_highoverlap_split.json",
    },
}
CASE_SET_OUTPUT_SUFFIX = {
    "complementary": "",
    "highoverlap": "_highoverlap",
}
DEFAULT_KEEP_CLASSES = (
    "potted_plant",
    "chair",
    "medical_bag",
    "traffic_cone",
    "wet_floor_sign",
    "fire_extinguisher",
    "trash_can",
)


def source_case_names(
    source: Path,
    source_split_files: Dict[str, str],
) -> Dict[str, Dict[str, set[str]]]:
    result: Dict[str, Dict[str, set[str]]] = {}
    for scene_name in HELD_OUT_SCENES.values():
        scene_dir = source / scene_name
        if not scene_dir.is_dir():
            raise SystemExit(f"Required source scene not found: {scene_dir}")

        split_names = {split: set() for split in SOURCE_SPLITS}
        for condition, split_file in source_split_files.items():
            split_path = scene_dir / split_file
            if not split_path.is_file():
                raise SystemExit(f"Missing source split JSON: {split_path}")
            source_split = converter.load_source_split(split_path)
            for split in SOURCE_SPLITS:
                split_names[split].update(
                    converter.scenario_name_from_source_split(
                        scene_name, condition, case_file
                    )
                    for case_file in source_split.get(split, [])
                )
        result[scene_name] = split_names
    return result


def build_split_plans(
    source: Path,
    scenarios: Sequence[converter.ScenarioExport],
    source_split_files: Dict[str, str],
) -> Dict[str, Dict[str, set[str]]]:
    source_names = source_case_names(source, source_split_files)
    discovered_names = {scenario.name for scenario in scenarios}
    plans: Dict[str, Dict[str, set[str]]] = {}

    for held_out_alias, held_out_scene in HELD_OUT_SCENES.items():
        plan = {split: set() for split in OUTPUT_SPLITS}
        for scene_name, split_names in source_names.items():
            if scene_name == held_out_scene:
                for source_split in SOURCE_SPLITS:
                    plan["test"].update(split_names[source_split])
            else:
                plan["train"].update(split_names["train"])
                plan["train"].update(split_names["test"])
                plan["validate"].update(split_names["validate"])

        missing = set().union(*plan.values()) - discovered_names
        if missing:
            preview = ", ".join(sorted(missing, key=converter.natural_key)[:10])
            raise SystemExit(
                f"{held_out_alias}: {len(missing)} planned scenario(s) were not "
                f"discovered under {source}: {preview}"
            )
        if (
            plan["train"] & plan["validate"]
            or plan["train"] & plan["test"]
            or plan["validate"] & plan["test"]
        ):
            raise SystemExit(
                f"{held_out_alias}: output splits overlap; "
                "check the source split JSON files"
            )
        plans[held_out_alias] = plan
    return plans


def convert_scenario(
    scenario: converter.ScenarioExport,
    output_split: Path,
    image_size: Tuple[int, int],
    limit_frames: Optional[int],
    skip_initial_frames: int,
    classes: Optional[Sequence[str]],
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
    object_id_location_decimals: int,
    object_id_size_decimals: int,
    object_id_registry: Dict[str, str],
) -> Tuple[int, Dict[str, str], int]:
    timestamps = converter.common_timestamps(scenario.robots)
    if skip_initial_frames:
        timestamps = timestamps[skip_initial_frames:]
    if limit_frames is not None:
        timestamps = timestamps[:limit_frames]
    timestamps, skipped_lidar_sparse = converter.filter_timestamps_by_lidar_points(
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
    object_id_scene_key = converter.scene_key_from_scenario_name(scenario.name)

    for robot in scenario.robots:
        cav_id = cav_ids[robot.name]
        assignment[cav_id] = "m1"
        pose_by_timestamp = converter.load_pose_csv(robot.path / "data" / "pose.csv")
        camera_info = converter.load_yaml(robot.path / "camera_info.yaml")
        lidar_info = converter.load_yaml(robot.path / "lidar_info.yaml")
        t_base_camera = converter.opencood_camera_from_base_camera_optical(
            converter.matrix_from_yaml_aliases(
                camera_info, "camera_optical_to_base", "camera_to_base"
            )
        )
        t_base_lidar = converter.opencood_lidar_from_base_lidar(
            converter.matrix_from_yaml_aliases(
                lidar_info, "lidar_sensor_to_base", "lidar_to_base"
            )
        )
        out_cav = output_split / scenario.name / cav_id
        out_cav.mkdir(parents=True, exist_ok=True)

        for timestamp in timestamps:
            if pose_by_timestamp:
                pos, quat = pose_by_timestamp[timestamp]
            else:
                pos, quat = converter.load_pose_npy(
                    robot.path / "data" / "pose" / f"{timestamp}.npy"
                )
            t_world_base = converter.transform_from_pose(pos, quat)
            t_world_lidar = t_world_base @ t_base_lidar
            t_world_camera = t_world_base @ t_base_camera
            lidar_pose = converter.tfm_to_opencood_pose(t_world_lidar)
            vehicles = converter.load_objects(
                converter.resolve_label_path(robot.path, timestamp),
                object_id_scene_key,
                classes,
                min_box_size,
                detection_class,
                class_to_id,
                asset_to_class,
                object_id_location_decimals,
                object_id_size_decimals,
                object_id_registry,
            )
            converter.write_xyzrgb_pcd(
                robot.path / "data" / "lidar" / f"{timestamp}.pcd",
                out_cav / f"{timestamp}.pcd",
                True,
                xyz_rotation=converter.OPENCOOD_LIDAR_FROM_ISAAC_LIDAR,
            )
            converter.save_rgb_camera_set(
                robot.path / "data" / "rgb" / f"{timestamp}.png",
                out_cav / timestamp,
                True,
                image_size,
                camera_count,
            )
            converter.save_bev_visibility(
                out_cav / f"{timestamp}_bev_visibility.png",
                vehicles,
                t_world_lidar,
                visibility_range,
            )
            frame_yaml = converter.make_frame_yaml(
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
            with (out_cav / f"{timestamp}.yaml").open(
                "w", encoding="utf-8"
            ) as handle:
                converter.yaml.dump(
                    frame_yaml,
                    handle,
                    Dumper=converter.NoAliasSafeDumper,
                    sort_keys=False,
                )
    return len(timestamps), assignment, skipped_lidar_sparse


def convert_scenario_task(
    scenario: converter.ScenarioExport,
    convert_kwargs: dict,
) -> Tuple[str, int, Dict[str, str], int]:
    nframes, assignment, skipped = convert_scenario(scenario, **convert_kwargs)
    return scenario.name, nframes, assignment, skipped


def convert_output_split(
    scenarios: Sequence[converter.ScenarioExport],
    output_split: Path,
    worker_count: int,
    convert_kwargs: dict,
) -> Tuple[Dict[str, Dict[str, str]], int]:
    assignments: Dict[str, Dict[str, str]] = {}
    frame_count = 0
    robot_counts = {scenario.name: len(scenario.robots) for scenario in scenarios}
    split_kwargs = dict(convert_kwargs)
    split_kwargs["output_split"] = output_split

    def record(result: Tuple[str, int, Dict[str, str], int]) -> None:
        nonlocal frame_count
        scenario_name, nframes, assignment, skipped = result
        assignments[scenario_name] = assignment
        frame_count += nframes * robot_counts[scenario_name]
        skip_msg = f", skipped {skipped} sparse lidar frame(s)" if skipped else ""
        print(
            f"{scenario_name}: {robot_counts[scenario_name]} CAV(s), "
            f"{nframes} shared frame(s){skip_msg}",
            flush=True,
        )

    if worker_count == 1:
        for scenario in scenarios:
            record(convert_scenario_task(scenario, split_kwargs))
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(convert_scenario_task, scenario, split_kwargs)
                for scenario in scenarios
            ]
            for future in as_completed(futures):
                record(future.result())
    return assignments, frame_count


def write_metadata(
    output: Path,
    assignments: Dict[str, Dict[str, str]],
    object_id_entries: Sequence[dict],
    args: argparse.Namespace,
    class_names: Sequence[str],
    class_to_id: Dict[str, int],
    object_list_path: Optional[Path],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "heter_modality_assign.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(assignments, handle, indent=2, sort_keys=True)
    with (output / "isaacsim_global_object_ids.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "id_format": "seven_digit_numeric_string",
                "id_source": "scene_location_size",
                "location_decimals": args.object_id_location_decimals,
                "size_decimals": args.object_id_size_decimals,
                "objects": object_id_entries,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    with (output / "isaacsim_class_map.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "training_mode": "class_agnostic",
                "detection_class": args.detection_class,
                "class_names": list(class_names),
                "class_to_id": class_to_id,
                "object_list": str(object_list_path),
            },
            handle,
            indent=2,
            sort_keys=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Generated Isaac Sim Dataset directory. This input is required.",
    )
    parser.add_argument(
        "--output-parent",
        type=Path,
        default=DATASET_PARENT,
        help=(
            "Parent directory for the three fixed IsaacSimOPV2V_* outputs. "
            "Defaults to ../dataset."
        ),
    )
    parser.add_argument(
        "--case-set",
        choices=tuple(CASE_SET_SPLIT_FILES),
        default="complementary",
        help=(
            "Convert distance/occlusion cases to IsaacSimOPV2V_{scene} "
            "(complementary, default), or convert only high-overlap cases to "
            "IsaacSimOPV2V_{scene}_highoverlap."
        ),
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--skip-initial-frames", type=int, default=0)
    parser.add_argument("--classes", default=",".join(DEFAULT_KEEP_CLASSES))
    parser.add_argument(
        "--object-list", type=Path, default=converter.DEFAULT_OBJECT_LIST
    )
    parser.add_argument(
        "--detection-class", default=converter.DEFAULT_DETECTION_CLASS
    )
    parser.add_argument(
        "--object-id-location-decimals",
        type=int,
        default=converter.DEFAULT_OBJECT_ID_LOCATION_DECIMALS,
    )
    parser.add_argument(
        "--object-id-size-decimals",
        type=int,
        default=converter.DEFAULT_OBJECT_ID_SIZE_DECIMALS,
    )
    parser.add_argument("--image-size", default="1280x800")
    parser.add_argument("--min-box-size", type=float, default=0.05)
    parser.add_argument(
        "--camera-count", type=int, default=converter.DEFAULT_CAMERA_COUNT
    )
    parser.add_argument(
        "--visibility-range",
        type=converter.parse_visibility_range,
        default=converter.DEFAULT_VISIBILITY_RANGE,
    )
    parser.add_argument("--min-lidar-points-in-range", type=int, default=5)
    parser.add_argument(
        "--lidar-filter-range",
        type=converter.parse_lidar_filter_range,
        default=converter.DEFAULT_LIDAR_FILTER_RANGE,
    )
    parser.add_argument(
        "--lidar-filter-aug-rotation-range",
        type=converter.parse_lidar_filter_aug_rotation_range,
        default=converter.DEFAULT_LIDAR_FILTER_AUG_ROTATION_RANGE,
    )
    parser.add_argument(
        "--lidar-filter-aug-rotation-samples",
        type=int,
        default=converter.DEFAULT_LIDAR_FILTER_AUG_ROTATION_SAMPLES,
    )
    parser.add_argument(
        "--lidar-filter-aug-scales",
        type=converter.parse_float_list,
        default=converter.DEFAULT_LIDAR_FILTER_AUG_SCALES,
    )
    parser.add_argument(
        "--disable-lidar-filter-aug-flip-x", action="store_true"
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Replace the exact non-empty held-out output roots before conversion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers is not None and args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.limit_frames is not None and args.limit_frames < 1:
        raise SystemExit("--limit-frames must be >= 1")
    if args.skip_initial_frames < 0:
        raise SystemExit("--skip-initial-frames must be >= 0")
    if args.camera_count < 1:
        raise SystemExit("--camera-count must be >= 1")

    source = converter.require_source_root(args.source)
    output_parent = args.output_parent.expanduser().resolve()
    all_scenarios = converter.discover_scenarios(source)
    if not all_scenarios:
        raise SystemExit(f"No Isaac Sim scenarios found under {source}")
    source_split_files = CASE_SET_SPLIT_FILES[args.case_set]
    plans = build_split_plans(source, all_scenarios, source_split_files)
    for held_out_alias, plan in plans.items():
        for output_split, scenario_names in plan.items():
            if not scenario_names:
                raise SystemExit(
                    f"{held_out_alias}/{output_split}: no scenarios selected"
                )

    output_suffix = CASE_SET_OUTPUT_SUFFIX[args.case_set]
    outputs = {
        held_out_alias: (
            output_parent
            / f"IsaacSimOPV2V_{held_out_alias}{output_suffix}"
        )
        for held_out_alias in plans
    }
    planned_names = set().union(
        *(names for plan in plans.values() for names in plan.values())
    )
    planned_scenarios = [
        scenario for scenario in all_scenarios if scenario.name in planned_names
    ]

    classes = converter.parse_classes(args.classes)
    object_list_path = (
        args.object_list.resolve() if args.object_list is not None else None
    )
    class_to_id, asset_to_class, class_names = converter.load_class_map(
        object_list_path
    )
    classes = converter.normalize_keep_classes(classes, asset_to_class)
    class_to_id, class_names = converter.filter_export_class_map(
        class_names, classes
    )
    object_id_registry, object_id_entries = (
        converter.build_global_object_id_registry(
            planned_scenarios,
            classes,
            args.min_box_size,
            class_to_id,
            asset_to_class,
            args.object_id_location_decimals,
            args.object_id_size_decimals,
        )
    )
    print(
        f"Built global static-object ID registry: "
        f"{len(object_id_registry)} object(s)",
        flush=True,
    )

    worker_count = (
        args.workers
        if args.workers is not None
        else converter.default_worker_count()
    )
    scenario_by_name = {
        scenario.name: scenario for scenario in planned_scenarios
    }
    convert_kwargs = {
        "image_size": converter.parse_image_size(args.image_size),
        "limit_frames": args.limit_frames,
        "skip_initial_frames": args.skip_initial_frames,
        "classes": classes,
        "min_box_size": args.min_box_size,
        "detection_class": args.detection_class,
        "class_to_id": class_to_id,
        "asset_to_class": asset_to_class,
        "camera_count": args.camera_count,
        "visibility_range": args.visibility_range,
        "min_lidar_points_in_range": args.min_lidar_points_in_range,
        "lidar_filter_range": args.lidar_filter_range,
        "lidar_filter_aug_rotation_range": args.lidar_filter_aug_rotation_range,
        "lidar_filter_aug_rotation_samples":
            args.lidar_filter_aug_rotation_samples,
        "lidar_filter_aug_scales": args.lidar_filter_aug_scales,
        "lidar_filter_aug_flip_x":
            not args.disable_lidar_filter_aug_flip_x,
        "object_id_location_decimals": args.object_id_location_decimals,
        "object_id_size_decimals": args.object_id_size_decimals,
        "object_id_registry": object_id_registry,
    }

    converter.prepare_output_roots(
        tuple(outputs.values()),
        source,
        args.overwrite_output,
    )

    for held_out_alias, plan in plans.items():
        output = outputs[held_out_alias]
        assignments: Dict[str, Dict[str, str]] = {}
        total_frame_count = 0
        print(f"\n=== Held out: {held_out_alias} -> {output} ===", flush=True)

        for output_split in OUTPUT_SPLITS:
            split_scenarios = sorted(
                (scenario_by_name[name] for name in plan[output_split]),
                key=lambda scenario: converter.natural_key(scenario.name),
            )
            if not split_scenarios:
                raise SystemExit(
                    f"{held_out_alias}/{output_split}: no scenarios selected"
                )
            split_workers = min(worker_count, len(split_scenarios))
            print(
                f"\n[{held_out_alias}/{output_split}] "
                f"{len(split_scenarios)} scenario(s), "
                f"{split_workers} worker(s)",
                flush=True,
            )
            split_assignments, split_frames = convert_output_split(
                split_scenarios,
                output / output_split,
                split_workers,
                convert_kwargs,
            )
            assignments.update(split_assignments)
            total_frame_count += split_frames

        write_metadata(
            output,
            assignments,
            object_id_entries,
            args,
            class_names,
            class_to_id,
            object_list_path,
        )
        print(f"\nWrote dataset:          {output}", flush=True)
        print(
            f"Wrote assignment JSON: "
            f"{output / 'heter_modality_assign.json'}",
            flush=True,
        )
        print(f"Total CAV frames:      {total_frame_count}", flush=True)


if __name__ == "__main__":
    main()
