# Isaac Sim Dataset Frame Notes

This repo exports Isaac Sim data in the current local dataset format under folders like:

```text
Dataset/<scene>/<condition>/<case>/<robot>/
```

The code has been updated so the calibration files explicitly preserve the source
sensor frames while staying backward-compatible with the existing keys already
used by the repo.

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

## Calibration keys

`camera_info.yaml` now writes both:

- `camera_optical_to_base`
- `camera_to_base`

They store the same `4x4` matrix. `camera_optical_to_base` is the preferred
name. `camera_to_base` is kept as a compatibility alias for older scripts.

`lidar_info.yaml` now writes both:

- `lidar_sensor_to_base`
- `lidar_to_base`

They also store the same `4x4` matrix. `lidar_sensor_to_base` is the preferred
name. `lidar_to_base` is the compatibility alias.

The loader priority in the updated code is:

1. Prefer the explicit new key.
2. Fall back to the old alias.
3. For the camera only, if neither key exists, fall back to the repo's original
   default optical-to-base transform.

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

This is exactly the form used by the updated `visualize.py`,
`visualize_partial.py`, `3d_filter.py`, and `debug_projection.py`.

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

## Updated scripts

The following files now follow the explicit-frame calibration format:

- `robot.py`
- `frame_utils.py`
- `visualize.py`
- `visualize_partial.py`
- `3d_filter.py`
- `debug_projection.py`
