# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

from collections import OrderedDict
import importlib

import torch
import torch.nn as nn
import torchvision

from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.center_head import CenterHead
from opencood.models.sub_modules.downsample_conv import DownsampleConv


class HeterModelLate(nn.Module):
    """Late-heter model shared by anchor-based and CenterHead experiments."""

    def __init__(self, args):
        super(HeterModelLate, self).__init__()
        self.modality_name_list = [
            x for x in args.keys() if x.startswith("m") and x[1:].isdigit()
        ]
        self.cav_range = args["lidar_range"]
        self.sensor_type_dict = OrderedDict()
        self.head_type_dict = OrderedDict()
        self.x_min, self.y_min, _, self.x_max, self.y_max, _ = self.cav_range
        self.min_size = args.get("min_size", 0.05)
        self.max_size = args.get("max_size", 6.0)
        self.export_dense_center_reg_preds = args.get(
            "export_dense_center_reg_preds", False
        )
        default_head_type = args.get("head_type", "anchor_based")

        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting["sensor_type"]
            self.sensor_type_dict[modality_name] = sensor_name
            head_args = model_setting["head_args"]
            head_type = head_args.get("head_type", default_head_type)
            self.head_type_dict[modality_name] = head_type

            encoder_class = self._find_encoder(model_setting["core_method"])
            setattr(
                self,
                f"encoder_{modality_name}",
                encoder_class(model_setting["encoder_args"]),
            )
            setattr(
                self,
                f"depth_supervision_{modality_name}",
                bool(model_setting["encoder_args"].get("depth_supervision", False)),
            )

            setattr(
                self,
                f"backbone_{modality_name}",
                ResNetBEVBackbone(model_setting["backbone_args"]),
            )
            if sensor_name == "camera":
                camera_mask_args = model_setting["camera_mask_args"]
                setattr(
                    self,
                    f"crop_ratio_W_{modality_name}",
                    self.cav_range[3] / camera_mask_args["grid_conf"]["xbound"][1],
                )
                setattr(
                    self,
                    f"crop_ratio_H_{modality_name}",
                    self.cav_range[4] / camera_mask_args["grid_conf"]["ybound"][1],
                )

            setattr(
                self,
                f"layers_{modality_name}",
                ResNetBEVBackbone(model_setting["layers_args"]),
            )
            setattr(
                self,
                f"layers_num_{modality_name}",
                len(model_setting["layers_args"]["num_upsample_filter"]),
            )
            setattr(
                self,
                f"shrink_conv_{modality_name}",
                DownsampleConv(model_setting["shrink_header"]),
            )

            if head_type == "center_head":
                setattr(
                    self,
                    f"center_head_{modality_name}",
                    CenterHead(
                        head_args["in_head"],
                        head_args.get("center_head", {}),
                    ),
                )
            elif head_type == "anchor_based":
                in_head = head_args["in_head"]
                setattr(
                    self,
                    f"cls_head_{modality_name}",
                    nn.Conv2d(in_head, args["anchor_number"], kernel_size=1),
                )
                setattr(
                    self,
                    f"reg_head_{modality_name}",
                    nn.Conv2d(in_head, args["anchor_number"] * 7, kernel_size=1),
                )
                setattr(
                    self,
                    f"dir_head_{modality_name}",
                    nn.Conv2d(
                        in_head,
                        args["anchor_number"] * args["dir_args"]["num_bins"],
                        kernel_size=1,
                    ),
                )
            else:
                raise ValueError(f"Unsupported late head_type: {head_type}")

    @staticmethod
    def _find_encoder(core_method):
        target_model_name = core_method.replace("_", "").lower()
        for module_name in (
            "opencood.models.heter_encoders_dsvt",
            "opencood.models.heter_encoders",
        ):
            encoder_lib = importlib.import_module(module_name)
            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name:
                    return cls
        raise RuntimeError(f"Unknown encoder {core_method}")

    def _decode_center_boxes(self, pred_dict):
        hm = pred_dict["hm"]
        center = pred_dict["center"]
        center_z = pred_dict["center_z"]
        dim = torch.exp(torch.clamp(pred_dict["dim"], min=-4.0, max=2.0))
        dim = torch.clamp(dim, min=self.min_size, max=self.max_size)
        rot = pred_dict["rot"]

        batch_size, _, height, width = hm.shape
        device = hm.device
        dtype = hm.dtype
        stride_x = (self.x_max - self.x_min) / float(width)
        stride_y = (self.y_max - self.y_min) / float(height)

        xs = torch.arange(width, device=device, dtype=dtype)
        ys = torch.arange(height, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        center_x = (grid_x.unsqueeze(0) + center[:, 0]) * stride_x + self.x_min
        center_y = (grid_y.unsqueeze(0) + center[:, 1]) * stride_y + self.y_min
        yaw = torch.atan2(rot[:, 1], rot[:, 0])

        boxes = torch.stack(
            [
                center_x,
                center_y,
                center_z[:, 0],
                dim[:, 0],
                dim[:, 1],
                dim[:, 2],
                yaw,
            ],
            dim=-1,
        )
        return boxes.reshape(batch_size, -1, 7)

    def forward(self, data_dict):
        output_dict = {}
        modality_keys = [x for x in data_dict.keys() if x.startswith("inputs_")]
        assert len(modality_keys) == 1
        modality_name = modality_keys[0][len("inputs_"):]

        feature = getattr(self, f"encoder_{modality_name}")(data_dict, modality_name)
        feature = getattr(self, f"backbone_{modality_name}")(
            {"spatial_features": feature}
        )["spatial_features_2d"]

        if self.sensor_type_dict[modality_name] == "camera":
            _, _, height, width = feature.shape
            feature = torchvision.transforms.CenterCrop(
                (
                    int(height * getattr(self, f"crop_ratio_H_{modality_name}")),
                    int(width * getattr(self, f"crop_ratio_W_{modality_name}")),
                )
            )(feature)

        if getattr(self, f"depth_supervision_{modality_name}") and hasattr(
            getattr(self, f"encoder_{modality_name}"), "depth_items"
        ):
            output_dict.update(
                {
                    f"depth_items_{modality_name}": getattr(
                        self, f"encoder_{modality_name}"
                    ).depth_items
                }
            )

        feature_list = [feature]
        for i in range(1, getattr(self, f"layers_num_{modality_name}")):
            feature = getattr(self, f"layers_{modality_name}").get_layer_i_feature(
                feature, layer_i=i
            )
            feature_list.append(feature)

        feature = getattr(self, f"layers_{modality_name}").decode_multiscale_feature(
            feature_list
        )
        feature = getattr(self, f"shrink_conv_{modality_name}")(feature)

        if self.head_type_dict[modality_name] == "center_head":
            pred_dict = getattr(self, f"center_head_{modality_name}")(feature)
            center_output = {
                "cls_preds": pred_dict["hm"],
                "center_head_preds": pred_dict,
                "center_head_decode_args": {
                    "lidar_range": self.cav_range,
                    "min_size": self.min_size,
                    "max_size": self.max_size,
                },
                "center_preds": pred_dict["hm"],
                "offset_preds": pred_dict["center"],
                "z_preds": pred_dict["center_z"],
                "size_preds": pred_dict["dim"],
                "yaw_preds": pred_dict["rot"],
            }
            if (not self.training) and self.export_dense_center_reg_preds:
                center_output["reg_preds"] = self._decode_center_boxes(pred_dict)
            output_dict.update(center_output)
            return output_dict

        cls_preds = getattr(self, f"cls_head_{modality_name}")(feature)
        reg_preds = getattr(self, f"reg_head_{modality_name}")(feature)
        dir_preds = getattr(self, f"dir_head_{modality_name}")(feature)

        output_dict.update(
            {
                "cls_preds": cls_preds,
                "reg_preds": reg_preds,
                "dir_preds": dir_preds,
            }
        )
        return output_dict
