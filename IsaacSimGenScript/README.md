# Isaac Sim Multi-Robot Dataset Generator

This project generates synchronized multi-robot perception datasets in Isaac
Sim. It plans paired Jackal trajectories, runs the simulation, records sensor
data and annotations, and provides tools for inspecting the generated data.

## Repository scope

Git is intended to track source code, configuration, documentation, maps,
roadmaps, case PKLs, and split manifests. Generated datasets, compressed dataset
archives, logs, Python caches, and rendered videos are intentionally ignored.

The current main workflow is:

```text
Scene USD
    |
Isaac Sim Occupancy Map Generator
    |
map/raw_maps/*.png + *.yaml
    |
    +--> map/crop.py --> map/cropped_maps/ --> map/planning/debug.py
    |
map/planning/planner.py      build a scene roadmap
    |
map/planning/dual.py         generate paired cases and split manifests
    |
main.sh                      start the configured dataset batches
    |
Isaac Sim python.sh          provide the Isaac Sim Python environment
    |
main.py                      run each dataset-generation job
    |
Dataset/
    |
    +--> opv2v_converter/    convert generated data to OPV2V-like datasets
    |
visualize.py                 inspect cases and render images/videos
```

## Environment

Dataset generation and planning/visualization use different Python runtimes.

### System and simulator

| Component | Verified version |
| --- | --- |
| Operating system | Ubuntu 24.04.4 LTS |
| GPU | NVIDIA GeForce RTX 4080 SUPER (16 GB) |
| NVIDIA driver | 580.173.02 |
| Isaac Sim | 5.1.0-rc.19 |
| Isaac Sim Python | 3.11.13 |

The simulator runtime currently lives at:

```bash
export ISAAC_SIM_ROOT="$HOME/isaacsim"
export ISAAC_PYTHON="$ISAAC_SIM_ROOT/_build/linux-x86_64/release/python.sh"
```

Do not run `main.py` with the system Python or a regular Conda Python. It imports
`isaacsim`, `omni`, and `pxr` modules supplied by the Isaac Sim runtime.

### Planning and visualization environment

Create the non-simulator environment from `requirements.txt`. Choose any
environment name and replace `{env_name}` in the commands below:

```bash
conda create -n {env_name} python=3.11 -y
conda activate {env_name}
python -m pip install -r requirements.txt
```

Use this environment for occupancy-map cropping, roadmap and case generation,
visualization, and OPV2V conversion. Dataset recording itself must still use
the Isaac Sim `python.sh` described above.

## Required external assets

The GitHub package does not contain the Isaac Sim installation or the local USD
assets. Only `assets/README.md` is tracked. After cloning or unpacking the
project, copy the scene and object USD files into this local structure:

```text
assets/
  map/
    hospital_v1.usd
    office_v1.usd
    full_warehouse_v1.usd
  map_objects/
    *.usd
```

Keep `map/` and `map_objects/` as siblings: scene payloads use paths such
as `../map_objects/SM_ChairOffice.usd`. The base NVIDIA environments and
some props are still referenced through Isaac Sim 5.1 HTTPS asset URLs, so this
local copy is relocatable but not fully offline. See `assets/README.md` for the
expected filenames and setup notes.

The command-line `--usd_path` may remain repository-relative. `main.py`
resolves the root scene path to an absolute layer identifier before opening it,
which keeps the scene's relative object payloads anchored to `assets/map/`.

The default Isaac Sim runtime remains:

```text
~/isaacsim/_build/linux-x86_64/release/python.sh
```

Set `ISAAC_PYTHON` to override that location.

## Generate a Dataset

Run commands from the repository root. `--roadmap` accepts either one
`case_*.pkl` file or a directory containing multiple cases.

```bash
"$ISAAC_PYTHON" main.py \
  --condition dual_case_shadow \
  --roadmap map/planning/trajectory/hospital/dual/path_case_shadow/ \
  --usd_path assets/map/hospital_v1.usd
```

The default output root is the repository-relative `Dataset/` directory.
`main.sh` resolves all project paths from its own location, so it can be
launched from any working directory. It contains the existing 12
scene/condition batch commands and runs them sequentially.

### Override the Dataset output root

Set `DATASET_OUTPUT_ROOT` for the command that starts the generator. When the
value is relative, it is resolved from the repository root rather than the
current working directory.

Use a repository-relative output directory:

```bash
DATASET_OUTPUT_ROOT=outputs/Dataset_experiment_01 ./main.sh
```

This writes to:

```text
<repository-root>/outputs/Dataset_experiment_01/
```

Use an absolute output directory, such as a separate data disk:

```bash
DATASET_OUTPUT_ROOT=/mnt/datasets/isaac_dataset_v1 ./main.sh
```

The same override works when running one scene directly:

```bash
DATASET_OUTPUT_ROOT=/mnt/datasets/hospital_test \
  "$ISAAC_PYTHON" main.py \
  --condition dual_case_shadow \
  --roadmap map/planning/trajectory/hospital/dual/path_case_shadow/ \
  --usd_path assets/map/hospital_v1.usd
```

The variable only changes where new output is written. It does not move,
rename, merge, or delete an existing `Dataset/` directory. If the variable is
not set, output continues to use `<repository-root>/Dataset/`.

## Convert generated data to OPV2V

Run the converters only after `main.sh` or the required `main.py` jobs have
finished writing the Isaac Sim Dataset. These scripts use the Conda environment
created from `requirements.txt`; they do not require the Isaac Sim Python
runtime.

Run all commands below from the repository root. `--source` has no default and
is always required, so the converter cannot silently read an unrelated Dataset.
`--output`, `--output-parent`, and `--object-list` remain overridable.

The repository-relative defaults are:

```text
standard output:       <repository-root>/../dataset/IsaacSimOPV2V/
held-out output parent: <repository-root>/../dataset/
object list:            <repository-root>/map/object_list.json
```

Set portable input and output locations. `ISAAC_DATASET_ROOT` must match the
`DATASET_OUTPUT_ROOT` used during generation. The example keeps converted data
in a sibling directory so large generated files are not placed in this source
repository:

```bash
conda activate {env_name}
export ISAAC_DATASET_ROOT="${DATASET_OUTPUT_ROOT:-Dataset}"
export OPV2V_OUTPUT_PARENT="../dataset"
```

`main.py` copies the required `path_case_*_split.json` files into each generated
`Dataset/<scene>/` directory. The converter requires the distance and shadow
split JSON files for every source scene and stops immediately if any are
missing. There is no automatic no-manifest fallback. A custom JSON containing
`train`, `validate`, and `test` scenario lists can be supplied explicitly with
`--split-manifest` instead.

### Standard OPV2V-like conversion

The standard converter processes the distance and shadow case manifests into
one train/validate/test dataset:

```bash
python opv2v_converter/convert_isaac_to_opv2v.py \
  --source "$ISAAC_DATASET_ROOT" \
  --output "$OPV2V_OUTPUT_PARENT/IsaacSimOPV2V" \
  --object-list map/object_list.json \
  --split all
```

RGB, lidar, frame YAML, visibility images, and metadata are written below
`IsaacSimOPV2V/`. Depth images are written to the automatically derived sibling
directory `IsaacSimOPV2V_Hetero/`.

### Leave-one-scene-out conversion

This converter requires generated data and split manifests for all three scene
names: `full_warehouse_v1`, `hospital_v1`, and `office_v1`. The complementary
mode uses the distance and shadow cases and creates one held-out dataset for
each scene:

```bash
python opv2v_converter/convert_isaac_to_opv2v_held_out_scene.py \
  --source "$ISAAC_DATASET_ROOT" \
  --output-parent "$OPV2V_OUTPUT_PARENT" \
  --object-list map/object_list.json \
  --case-set complementary
```

To convert only the high-overlap cases, run the same command with:

```bash
--case-set highoverlap
```

The converters refuse to write into a non-empty output root by default. To
intentionally replace an earlier conversion, append `--overwrite-output`. This
removes only the exact OPV2V output roots selected by the command before
rebuilding them; protected project, home, source Dataset, and symlink paths are
rejected. The converters default to half of the available CPU cores; pass
`--workers N` when a smaller worker limit is needed.

## Prepare occupancy maps

Roadmap preparation requires one 2D occupancy map for each Isaac Sim scene.
Generate these maps from the corresponding USD stages in Isaac Sim 5.1 before
running `crop.py` or the planning scripts.

1. Open the scene USD in Isaac Sim.
2. In **Window > Extensions**, search for and enable
   `isaacsim.asset.gen.omap` if the Occupancy Map tool is not already
   available.
3. Open **Tools > Robotics > Occupancy Map**.
4. Set the origin in unoccupied space, then set the lower bound, upper bound,
   and cell size. Geometry that should appear in the map must have collision
   geometry enabled.
5. Select **Calculate**, then **Visualize Image**.
6. In the visualization window, select **ROS Occupancy Map Parameters File**
   as the coordinate type, choose the scene filename, and save both the PNG
   image and YAML metadata.

For screenshots, parameter descriptions, and the complete Isaac Sim procedure,
see the official
[Isaac Sim 5.1 Mapping and Occupancy Map Generator guide](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/digital_twin/ext_isaacsim_asset_generator_occupancy_map.html).

Place the exported file pairs at these exact paths:

```text
map/raw_maps/
  hospital.png
  hospital.yaml
  office.png
  office.yaml
  warehouse.png
  warehouse.yaml
```

Each YAML file must contain at least `image`, `resolution`, and `origin`. The
Isaac Sim ROS YAML export also includes `negate`, `occupied_thresh`, and
`free_thresh`.

Run the crop script from the `map/` directory because its input and output
paths are currently relative to the working directory:

```bash
conda activate {env_name}
cd map
python crop.py
cd ..
```

The processed files are written to:

```text
map/cropped_maps/
  hospital.png
  hospital.yaml
  office.png
  office.yaml
  warehouse.png
  warehouse.yaml
```

Do not remove the files in `map/raw_maps/` after cropping. In the current code,
`map/planning/planner.py` and `map/planning/dual.py` still read `raw_maps/`,
while `map/planning/debug.py` reads `cropped_maps/`.

## Build roadmaps and paired cases

Use the Conda environment created from `requirements.txt`:

```bash
conda activate {env_name}
python map/planning/planner.py
python map/planning/dual.py
```

These scripts currently process the hospital, warehouse, and office scenes from
their `if __name__ == "__main__"` blocks. Planning outputs are written below
`map/planning/trajectory/`.

## Dataset format and coordinate frames

The remainder of this document describes the on-disk schema and frame
conventions used by the current generator.

This repo exports Isaac Sim data in the current local dataset format under folders like:

```text
Dataset/<scene>/<condition>/<case>/<robot>/
```

The calibration files explicitly preserve the source sensor frames used by the
generator and visualization tools.

## Current file format

For each robot folder:

```text
camera_info.yaml
lidar_info.yaml
tf_static.yaml
data/
  pose.csv
  imu.csv
  rgb/<timestamp>.png
  depth/<timestamp>.png
  lidar/<timestamp>.pcd
label/
  detection/3d/<timestamp>.txt
  detection/2d_tight/<timestamp>.txt
  detection/2d_loose/<timestamp>.txt
  segmentation/semantic/<timestamp>.npy
  ...
```

`pose.csv` is the current pose source. The repo does not use `pose/*.npy`.

## Frame split

The intended frame design is:

- `world`
  - Used by robot poses from `data/pose.csv`.
  - Used by filtered 3D labels in `label/detection/3d`.
- `base`
  - Robot body frame.
  - This is the local frame used by the static camera/lidar extrinsics.
- `camera_optical`
  - Used by RGB, depth, and the pinhole intrinsics.
  - Axis convention: `x right`, `y down`, `z forward`.
- `lidar_sensor`
  - Native Isaac Sim lidar frame stored by the raw point cloud export.
  - Axis convention: `x right`, `y forward`, `z up`.

## Exact transform meaning

The saved matrices are local-to-parent transforms:

- `camera_optical_to_base`: maps a point from camera optical coordinates into
  base coordinates.
- `lidar_sensor_to_base`: maps a point from lidar sensor coordinates into base
  coordinates.

Using homogeneous coordinates:

```text
p_base = T_camera_optical_to_base * p_camera_optical
p_base = T_lidar_sensor_to_base * p_lidar_sensor
```

If the robot pose from `pose.csv` is written as the base pose in world:

```text
p_world = T_world_to_base * p_base
```

then the world-to-camera projection used by the repo is:

```text
T_world_to_camera_optical = inv(T_world_to_base * T_camera_optical_to_base)
p_camera_optical = T_world_to_camera_optical * p_world
```

This is the form used by the current `visualize.py` projection code.

## Current camera transform

When the saved calibration is missing, the repo falls back to the original
optical camera transform:

```text
T_camera_optical_to_base =
[ 0,  0,  1, 0.25]
[-1,  0,  0, 0.00]
[ 0, -1,  0, 0.25]
[ 0,  0,  0, 1.00]
```

This means:

- camera `z forward` aligns with base `x forward`
- camera `x right` aligns with base `-y`
- camera `y down` aligns with base `-z`

So RGB, depth, intrinsics, and 3D projection stay in the optical frame instead
of being reinterpreted as a lidar-style frame.

## Current lidar transform

The lidar extrinsic is exported from the real Isaac Sim sensor pose and kept in
the native sensor frame:

```text
p_base = T_lidar_sensor_to_base * p_lidar_sensor
```

The current `tf_static.yaml` writes the same transform under:

- `base_link -> camera_color_optical_frame`
- `base_link -> lidar_link`

so the YAML files and TF export stay consistent.

## Raw 3D label convention

`label/detection/3d/<timestamp>.txt` is treated as world-frame boxes.
The current scripts assume each valid row is:

```text
class_name class_id bx by bz l w h yaw
```

where:

- `bx by bz` is the box center in `world`
- `l w h` are box dimensions
- `yaw` is a world-frame yaw around `z`

The debug and visualization helpers were updated to tolerate the current repo
format where the first field may be a class string.

## If you convert to OPV2V / OpenCOOD later

The correct rule is:

- keep the camera in `camera_optical`
- keep `camera0.intrinsic` in the same optical frame
- store `camera0.extrinsic` as `T_lidar_camera_optical`
- convert only the lidar cloud and world labels into the target OpenCOOD lidar
  frame

For the commonly used Isaac lidar frame:

```text
x right, y forward, z up
```

and a BEV lidar frame:

```text
x forward, y left, z up
```

the axis remap is:

```text
x_bev =  y_lidar_sensor
y_bev = -x_lidar_sensor
z_bev =  z_lidar_sensor
```

or in matrix form:

```text
T_lidar_sensor_to_bev =
[ 0, 1, 0, 0]
[-1, 0, 0, 0]
[ 0, 0, 1, 0]
[ 0, 0, 0, 1]
```

Then:

```text
p_bev = T_lidar_sensor_to_bev * p_lidar_sensor
```

If your downstream code uses a different lateral sign convention, only this
matrix should change. The camera optical frame should not be redefined.
