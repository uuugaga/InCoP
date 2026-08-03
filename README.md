# InCoP

Supplementary code for **InCoP: A Benchmark for Indoor Collaborative 3D Object Detection with Complementary Ground-Robot Views**.

InCoP studies indoor ground-to-ground collaborative perception with paired RGB and LiDAR observations. The benchmark focuses on complementary robot viewpoints in cluttered indoor environments and supports no-fusion, late-fusion, and intermediate-fusion evaluation. The included implementation contains the Complementary Local Correction (CLC) architecture and its communication-efficient configuration, Complementarity-Guided Residual Fusion (CGRF).

## Data Preparation

Place the released InCoP data under `dataset/` using the following layout:

```text
dataset/
├── IsaacSimOPV2V_hospital/
│   ├── train/
│   ├── validate/
│   ├── test/
│   └── heter_modality_assign.json
└── real_world/
    ├── train/
    ├── validate/
    └── heter_modality_assign.json
```

The IsaacSim hospital data provides separate `train/`, `validate/`, and `test/` splits. The real-world configuration uses its `validate/` split for evaluation. Update `root_dir`, `validate_dir`, `test_dir`, and `heter.assignment_path` in the selected YAML file if the data is stored elsewhere.

## Codebase

This project is built on top of [HEAL](https://github.com/yifanlu0227/HEAL), which is itself based on [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD). The `opencood/` package structure is retained for compatibility with those frameworks.

The main InCoP components are organized as follows:

```text
opencood/
├── data_utils/                 # InCoP/Isaac Sim dataset loaders
├── hypes_yaml/isaacsim/       # Experiment configurations
├── models/                    # Encoders, fusion models, and detection heads
├── loss/                      # Training objectives
├── tools/train_isaac.py       # Training entry point
├── tools/inference_isaac.py   # Quantitative evaluation entry point
```

## Installation

This codebase follows the [HEAL dependency stack](https://github.com/yifanlu0227/HEAL#installation) in a conda environment named `InCoP`. Run the following commands from the repository root.

The `InCoP` environment uses:

| Component | Version |
| --- | --- |
| Python | 3.8.20 |
| PyTorch | 2.4.1+cu121 |

### 1. Create and activate the conda environment

```bash
conda create -n InCoP python=3.8.20 pip=24.2 -y
conda activate InCoP
```

If a compatible `InCoP` environment already exists, skip the `conda create` command and activate it directly.

### 2. Install the Python dependencies

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` pins the PyTorch CUDA 12.1 wheels, torchvision, timm, spconv, and the remaining Python packages. Do not install more than one spconv package in the same environment.

### 3. Install the project in editable mode

```bash
python setup.py develop
```

An `EasyInstallDeprecationWarning` from the inherited HEAL setup is harmless.

### 4. Build the required bounding-box overlap extension

```bash
python opencood/utils/setup.py build_ext --inplace
```

This is a required Cython/C extension, not the optional FPV-RCNN CUDA extension. A working C compiler must be available on the machine.

## Pretrained encoder paths

`isaac_pretrained.path` is resolved relative to the repository root, not
relative to the YAML file. For example, if single-agent training creates:

```text
opencood/logs/isaacsim_m6_BEVFusion_center_head_hospital_2026_08_01_10_31_42/
└── net_epoch_bestval_at15.pth
```

then the pretrained configuration should use the following relative path:

```yaml
isaac_pretrained:
  enabled: true
  path: opencood/logs/isaacsim_m6_BEVFusion_center_head_hospital_2026_08_01_10_31_42
  checkpoint_mode: bestval
  load_prefixes: [encoder_m1]
```

The path may point either to a checkpoint directory or directly to a `.pth`
file. With `checkpoint_mode: bestval`, a directory path first selects its
single `net_epoch_bestval_at*.pth` file.

## Training

Run all commands from the repository root. Choose a configuration from
`opencood/hypes_yaml/isaacsim/` and start training with:

```bash
python opencood/tools/train_isaac.py \
  --hypes_yaml <CONFIG_FILE>
```

To continue an existing run:

```bash
python opencood/tools/train_isaac.py \
  --hypes_yaml <CONFIG_FILE> \
  --model_dir <CHECKPOINT_DIRECTORY>
```

Relevant configuration groups include:

- `BEVFusion_intermediate_ours/ours_*.yaml`: CGRF configurations.
- `BEVFusion_intermediate_ours/ablation_clc_dense_*.yaml`: Dense CLC configurations.
- `BEVFusion_intermediate/`: intermediate-fusion baselines.
- `Single/`: single-agent encoder pretraining.

## Evaluation

`<CHECKPOINT_DIRECTORY>` must point to a training run directory containing
its saved `config.yaml` and the applicable `net_epoch*.pth` checkpoint.
Choose `--fusion_method no` for single-agent or no-fusion evaluation,
`--fusion_method late` for late fusion, and `--fusion_method intermediate`
for intermediate fusion. For example, evaluate an intermediate-fusion
checkpoint on the configured test split with:

```bash
python opencood/tools/inference_isaac.py \
  --model_dir <CHECKPOINT_DIRECTORY> \
  --fusion_method intermediate \
  --eval_split test
```

To create an ego-only versus intermediate-fusion comparison video, only the
model directory and the high-level video flag are required:

```bash
python opencood/tools/inference_isaac.py \
  --model_dir <CHECKPOINT_DIRECTORY> \
  --video_compare_fusion
```

Video export requires the `ffmpeg` executable to be installed and available
on the system `PATH`. It is a system dependency and is not installed by
`requirements.txt`.

## Acknowledgements

We thank the authors of OpenCOOD and HEAL for their excellent open-source frameworks. This work builds upon their contributions to collaborative perception research.
