# -*- coding: utf-8 -*-
"""Isaac Sim dataset adapter for OPV2V-style converted data.

The converted Isaac Sim layout intentionally keeps only the real front camera
(`camera0`) instead of fabricating OPV2V's four camera views. This adapter
reuses OPV2V parsing and label generation, but discovers camera/depth files
from the hypes `data_aug_conf.cams` / `Ncams` settings.
"""

import os

from opencood.data_utils.datasets.basedataset.opv2v_basedataset import OPV2VBaseDataset


class IsaacSimBaseDataset(OPV2VBaseDataset):
    def _camera_names(self):
        if not self.load_camera_file:
            return ["camera0"]

        data_aug_conf = self.params["fusion"]["args"].get("data_aug_conf")
        if not isinstance(data_aug_conf, dict):
            data_aug_conf = getattr(self, "data_aug_conf", {})
        if not isinstance(data_aug_conf, dict):
            for setting in self.params.get("heter", {}).get("modality_setting", {}).values():
                if setting.get("sensor_type") == "camera":
                    data_aug_conf = setting.get("data_aug_conf", {})
                    break
        if not isinstance(data_aug_conf, dict):
            data_aug_conf = {}

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


ISAACSIMBaseDataset = IsaacSimBaseDataset
