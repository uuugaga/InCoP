# Isaac Sim to OPV2V Converter

This folder contains a converter for the dataset exported by `isaac_sim/main.py`.
It does not modify the original Isaac Sim data or OpenCOOD code.

## Coordinate Design

The source Isaac Sim export and the converted OPV2V/OpenCOOD sample now follow
a strict frame split:

### Source Isaac Sim frames

- `world`
  - Used by raw robot pose and raw 3D labels.
- `base`
  - Robot body frame used by `data/pose/*.npy`.
- `camera_optical`
  - Used by RGB images, depth images, and pinhole intrinsics.
  - Axes: `x right`, `y down`, `z forward`.
- `lidar_sensor`
  - Native Isaac Sim lidar frame used by the raw point cloud export.
  - Axes: `x right`, `y forward`, `z up`.

### Source calibration files

`camera_info.yaml` stores:

- `camera_optical_to_base`
  - Preferred explicit field name.
- `camera_to_base`
  - Backward-compatible alias with the same matrix.

`lidar_info.yaml` stores:

- `lidar_sensor_to_base`
  - Preferred explicit field name.
- `lidar_to_base`
  - Backward-compatible alias with the same matrix.

### Converted OPV2V/OpenCOOD frames

- `lidar`
  - OpenCOOD BEV/detection frame.
  - Axes: `x forward`, `y lateral`, `z up`.
- `camera0.extrinsic`
  - `T_lidar_camera_optical`, where the camera remains in optical frame.
- `camera0.intrinsic`
  - Pinhole intrinsic for the same optical frame.

### Important rule

The converter does not reinterpret the optical camera as a lidar-style frame.
It only:

- keeps the source camera in `camera_optical`
- converts the raw Isaac Sim lidar frame into the OpenCOOD lidar frame
- projects world labels into the converted lidar frame

This keeps `RGB + depth + intrinsic + camera extrinsic` geometrically
consistent for Lift-Splat-Shoot style models.

Typical small-case conversion from the repo root:

```bash
python isaac_sim/opv2v_converter/convert_isaac_to_opv2v.py \
  --source HEAL/dataset/IsaacSimDataset/Dataset \
  --output HEAL/dataset/IsaacSimOPV2V \
  --split train \
  --source-splits on \
  --limit-scenarios 2 \
  --limit-frames 8 \
  --copy-files
```

By default the converter runs with `--source-splits auto`. If the IsaacSim
source scene folders contain `path_case_distance_split.json` and
`path_case_shadow_split.json`, the converter reads those files directly and
uses the requested `--split` (`train`, `validate`, or `test`) to select
scenarios. Use `--source-splits on` for full dataset conversion so missing split
JSON files fail fast; use `--source-splits off` only for ad-hoc debugging where
all discovered scenarios should be eligible for the selected output split.

OpenCOOD camera-depth loading replaces `OPV2V` with `OPV2V_Hetero` in depth
paths. For an output named `IsaacSimOPV2V`, this converter also writes depths to
`IsaacSimOPV2V_Hetero`.

For the current single-camera Jackal export, only `camera0` and `depth0` are
written. Use the `isaacsim` dataset adapter and set camera hypes to
`cams: ['camera0']` and `Ncams: 1`.

The current converter keeps the IsaacSim camera export at its native
`1280x800` size by default and scales `camera0.intrinsic` only when
`--image-size` requests another size. The matching IsaacSim LSS hypes therefore
use `H: 800` and `W: 1280`.

`bev_visibility.png` is generated as a 256x256 black image with white polygons
rasterized from each frame's 3D bbox footprints. By default it keeps only the
front rectangular indoor region in lidar coordinates:

```text
x: [0, 25] meters
y: [-25, 25] meters
```

Override it with `--visibility-range x_min,y_min,x_max,y_max`. Use
`--skip-initial-frames 1` when the first Isaac Sim frame has incomplete labels.

By default, conversion keeps all rows in the current filtered
`label/detection/3d` files. It does not run an extra FOV projection check
because Isaac Sim's current 3D labels are already generated from camera-visible
annotations. Pass a comma-separated `--classes` list to apply an additional
class filter.

Raw 3D labels whose first field contains multiple labels, such as
`sign,wet_floor_sign`, are discarded as duplicate Isaac Sim annotations.
Single-label boxes are also deduplicated within each normalized class by
comparing center distance, dimensions, and yaw.

The converted labels are multi-class-ready but class-agnostic for the current
OpenCOOD training setup. Each object is written with:

```yaml
class: object        # current single-class training target
obj_type: object     # explicit class-agnostic detection type
class_name: table    # normalized IsaacSim class for future multi-class use
class_id: 12         # stable id from isaac_sim/map/object_list.json
raw_class: table     # raw first field from the IsaacSim 3D label txt
```

The converter also writes `isaacsim_class_map.json` beside
`heter_modality_assign.json`. Use `--object-list` to point at a different class
map and `--detection-class` to rename the current class-agnostic target.

To train on the converted sample, copy your OpenCOOD hypes file and change only:

```yaml
root_dir: "dataset/IsaacSimOPV2V/train"
validate_dir: "dataset/IsaacSimOPV2V/train"
test_dir: "dataset/IsaacSimOPV2V/train"
fusion:
  dataset: "isaacsim"
heter:
  assignment_path: "dataset/IsaacSimOPV2V/heter_modality_assign.json"
```

Keep the original OPV2V hypes unchanged until this small sample behaves as
expected.

IsaacSim-specific hypes are available under:

```text
HEAL/opencood/hypes_yaml/isaacsim/Single/
```
