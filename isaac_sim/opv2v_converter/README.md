# Isaac Sim to OPV2V Converter

This folder contains a converter for the dataset exported by `isaac_sim/main.py`.
It does not modify the original Isaac Sim data or OpenCOOD code.

Typical small-case conversion from the repo root:

```bash
python isaac_sim/opv2v_converter/convert_isaac_to_opv2v.py \
  --source HEAL/dataset/IsaacSimDataset \
  --output HEAL/dataset/IsaacSimOPV2V \
  --split train \
  --limit-scenarios 2 \
  --limit-frames 8 \
  --copy-files
```

OpenCOOD camera-depth loading replaces `OPV2V` with `OPV2V_Hetero` in depth
paths. For an output named `IsaacSimOPV2V`, this converter also writes depths to
`IsaacSimOPV2V_Hetero`.

For the current single-camera Jackal export, only `camera0` and `depth0` are
written. Use the `isaacsim` dataset adapter and set camera hypes to
`cams: ['camera0']` and `Ncams: 1`.

`bev_visibility.png` is generated as a 256x256 black image with white polygons
rasterized from each frame's 3D bbox footprints. By default it keeps only the
front rectangular indoor region in lidar coordinates:

```text
x: [0, 25] meters
y: [-25, 25] meters
```

Override it with `--visibility-range x_min,y_min,x_max,y_max`. Use
`--skip-initial-frames 1` when the first Isaac Sim frame has incomplete labels.

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
