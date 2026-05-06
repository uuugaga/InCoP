# -*- coding: utf-8 -*-
"""Isaac Sim dataset adapter for OPV2V-style converted data.

The converted Isaac Sim layout intentionally keeps only the real front camera
(`camera0`) instead of fabricating OPV2V's four camera views. This adapter
reuses OPV2V parsing and label generation, but discovers camera/depth files
from the hypes `data_aug_conf.cams` / `Ncams` settings.
"""

import json
import os

import numpy as np
import torch

from opencood.data_utils.datasets.basedataset.opv2v_basedataset import OPV2VBaseDataset
from opencood.utils.transformation_utils import x1_to_x2


class IsaacSimBaseDataset(OPV2VBaseDataset):
    # IsaacSim front camera uses OpenCV optical axes: x right, y down, z forward.
    # Converted lidar/BEV uses OpenCOOD axes: x forward, y left, z up. This
    # maps camera optical points (x right, y down, z forward) into lidar.
    camera_to_lidar_rotation = np.asarray(
        [[0.0, 0.0, 1.0],
         [-1.0, 0.0, 0.0],
         [0.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    camera_to_lidar_transform = np.eye(4, dtype=np.float32)
    camera_to_lidar_transform[:3, :3] = camera_to_lidar_rotation

    default_class_names = (
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
        "mug",
    )

    def __init__(self, params, visualize, train=True):
        super().__init__(params, visualize, train)
        self.isaac_class_names = self._resolve_class_names(params)
        self.isaac_class_to_id = {
            class_name: idx for idx, class_name in enumerate(self.isaac_class_names)
        }
        self._isaac_last_class_ids = None
        self._patch_class_aware_label_collate()

    def _resolve_class_names(self, params):
        configured = params.get("postprocess", {}).get("class_names")
        if configured:
            return list(configured)

        class_map_path = params.get("postprocess", {}).get("class_map_path")
        if class_map_path is None:
            root_dir = params.get("root_dir", "")
            dataset_dir = os.path.dirname(root_dir)
            class_map_path = os.path.join(dataset_dir, "isaacsim_class_map.json")

        try:
            with open(class_map_path, "r", encoding="utf-8") as class_map_file:
                class_map = json.load(class_map_file)
            class_names = class_map.get("class_names", [])
            if class_names:
                return list(class_names)
        except OSError:
            pass

        return list(self.default_class_names)

    def _patch_class_aware_label_collate(self):
        base_generate_label = self.post_processor.generate_label
        base_collate_batch = self.post_processor.collate_batch

        def generate_label_with_class_ids(*args, **kwargs):
            label_dict = base_generate_label(*args, **kwargs)
            if self._isaac_last_class_ids is not None:
                label_dict["object_class_ids"] = self._isaac_last_class_ids.copy()
            return label_dict

        def collate_batch_with_class_ids(label_batch_list):
            label_torch_dict = base_collate_batch(label_batch_list)
            if label_batch_list and all(
                "object_class_ids" in item for item in label_batch_list
            ):
                label_torch_dict["object_class_ids"] = torch.from_numpy(
                    np.asarray(
                        [item["object_class_ids"] for item in label_batch_list],
                        dtype=np.int64,
                    )
                )
            return label_torch_dict

        self.post_processor.generate_label = generate_label_with_class_ids
        self.post_processor.collate_batch = collate_batch_with_class_ids

    def _class_id_for_vehicle(self, vehicle_info):
        class_name = (
            vehicle_info.get("class_name")
            or vehicle_info.get("raw_class")
            or vehicle_info.get("obj_type")
            or vehicle_info.get("class")
        )
        if class_name in self.isaac_class_to_id:
            return self.isaac_class_to_id[class_name]

        raw_class = vehicle_info.get("raw_class")
        if raw_class in self.isaac_class_to_id:
            return self.isaac_class_to_id[raw_class]

        try:
            raw_id = int(vehicle_info.get("class_id", 0))
        except (TypeError, ValueError):
            raw_id = 0
        if 0 <= raw_id < len(self.isaac_class_names):
            return raw_id
        return 0

    def _generate_object_center_with_class_ids(self, cav_contents, reference_lidar_pose):
        object_np, mask, object_ids = self.post_processor.generate_object_center(
            cav_contents, reference_lidar_pose
        )

        vehicles = {}
        for cav_content in cav_contents:
            vehicles.update(cav_content["params"].get("vehicles", {}))

        class_ids = np.zeros(self.post_processor.params["max_num"], dtype=np.int64)
        for idx, object_id in enumerate(object_ids):
            if idx >= class_ids.shape[0]:
                break
            vehicle_info = vehicles.get(object_id, {})
            class_ids[idx] = self._class_id_for_vehicle(vehicle_info)

        self._isaac_last_class_ids = class_ids
        return object_np, mask, object_ids

    def generate_object_center_lidar(self, cav_contents, reference_lidar_pose):
        return self._generate_object_center_with_class_ids(
            cav_contents, reference_lidar_pose
        )

    def generate_object_center_camera(self, cav_contents, reference_lidar_pose):
        return self._generate_object_center_with_class_ids(
            cav_contents, reference_lidar_pose
        )

    def _camera_data_aug_conf(self):
        data_aug_conf = self.params["fusion"]["args"].get("data_aug_conf")
        if not isinstance(data_aug_conf, dict):
            data_aug_conf = getattr(self, "data_aug_conf", {})
        if not isinstance(data_aug_conf, dict):
            for setting in self.params.get("heter", {}).get("modality_setting", {}).values():
                if setting.get("sensor_type") == "camera":
                    data_aug_conf = setting.get("data_aug_conf", {})
                    break
        return data_aug_conf if isinstance(data_aug_conf, dict) else {}

    def _camera_names(self):
        if not self.load_camera_file:
            return ["camera0"]

        data_aug_conf = self._camera_data_aug_conf()
        camera_names = data_aug_conf.get("cams", ["camera0"])
        camera_count = data_aug_conf.get("Ncams", len(camera_names))
        return camera_names[:camera_count]

    def find_camera_files(self, cav_path, timestamp, sensor="camera"):
        camera_files = []
        for camera_name in self._camera_names():
            camera_id = camera_name.replace("camera", "")
            camera_files.append(os.path.join(cav_path,
                                             timestamp + f"_{sensor}{camera_id}.png"))
        return camera_files

    def retrieve_base_data(self, idx):
        data = super().retrieve_base_data(idx)
        target_size = self._target_camera_size()
        for cav_content in data.values():
            for camera_id, _ in enumerate(cav_content.get("camera_data", [])):
                self._normalize_camera_params(cav_content["params"], camera_id)

        if target_size is None:
            return data

        target_w, target_h = target_size
        for cav_content in data.values():
            camera_data = cav_content.get("camera_data", [])
            for camera_id, img in enumerate(camera_data):
                old_w, old_h = img.size
                if (old_w, old_h) == (target_w, target_h):
                    continue
                camera_data[camera_id] = img.resize((target_w, target_h))
                self._scale_camera_intrinsic(
                    cav_content["params"],
                    camera_id,
                    target_w / float(old_w),
                    target_h / float(old_h),
                )
        return data

    def get_ext_int(self, params, camera_id):
        camera_key = "camera%d" % camera_id
        self._normalize_camera_params(params, camera_id)
        camera_to_lidar = np.asarray(params[camera_key]["extrinsic"], dtype=np.float32)
        camera_intrinsic = np.asarray(params[camera_key]["intrinsic"], dtype=np.float32)
        return camera_to_lidar, camera_intrinsic

    def _normalize_camera_params(self, params, camera_id):
        camera_key = "camera%d" % camera_id
        camera_cfg = params[camera_key]
        camera_to_lidar = self.camera_to_lidar_transform.copy()
        camera_to_lidar[:3, 3] = self._camera_translation_in_lidar(params, camera_id)
        camera_cfg["extrinsic"] = camera_to_lidar.tolist()

    def _camera_translation_in_lidar(self, params, camera_id):
        camera_key = "camera%d" % camera_id
        camera_cfg = params.get(camera_key, {})

        # New IsaacSim conversions already store the camera center in the
        # OpenCOOD lidar frame. We only trust the translation part; the
        # rotation in some converted sets is not the OpenCV optical frame
        # expected by Lift-Splat-Shoot.
        if "extrinsic" in camera_cfg:
            stored_extrinsic = np.asarray(camera_cfg["extrinsic"], dtype=np.float32)
            return stored_extrinsic[:3, 3].copy()

        # Fallback for older conversions: keep the raw pose-based translation,
        # but still replace the rotation with the fixed optical-to-lidar map.
        if "cords" in camera_cfg and "lidar_pose_clean" in params:
            camera_coords = np.asarray(camera_cfg["cords"], dtype=np.float32)
            pose_based = x1_to_x2(camera_coords, params["lidar_pose_clean"]).astype(np.float32)
            return pose_based[:3, 3].copy()

        return np.zeros(3, dtype=np.float32)

    def _target_camera_size(self):
        data_aug_conf = self._camera_data_aug_conf()
        if "W" not in data_aug_conf or "H" not in data_aug_conf:
            return None
        return int(data_aug_conf["W"]), int(data_aug_conf["H"])

    @staticmethod
    def _scale_camera_intrinsic(params, camera_id, scale_x, scale_y):
        camera_key = "camera%d" % camera_id
        intrinsic = np.asarray(params[camera_key]["intrinsic"], dtype=np.float32).copy()
        intrinsic[0, 0] *= scale_x
        intrinsic[0, 2] *= scale_x
        intrinsic[1, 1] *= scale_y
        intrinsic[1, 2] *= scale_y
        params[camera_key]["intrinsic"] = intrinsic.tolist()


ISAACSIMBaseDataset = IsaacSimBaseDataset
