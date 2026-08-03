# InCoP Dataset

Dataset files are intentionally excluded from version control. Place the
released InCoP data in this directory using the following structure:

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

Update `root_dir`, `validate_dir`, `test_dir`, and `heter.assignment_path` in
the selected experiment YAML file if the dataset is stored elsewhere.
