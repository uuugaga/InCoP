import importlib

import torch.nn as nn

from opencood.models.heter_model_late import HeterModelLate


class HeterModelLateIsaac(HeterModelLate):
    """Late-heter model that can discover Isaac encoders and heads first.

    The base OpenCOOD late model keeps anchor classification class-agnostic
    with one logit per anchor. IsaacSim has semantic class labels, so this
    subclass can opt into a multi-class anchor head using only Isaac configs.
    """

    def __init__(self, args):
        super().__init__(args)
        self.isaac_num_classes = int(
            args.get("num_classes", len(args.get("class_names", [])) or 1)
        )
        self.isaac_multi_class_anchor = bool(
            args.get("multi_class_anchor", False) and self.isaac_num_classes > 1
        )
        if self.isaac_multi_class_anchor:
            self._replace_anchor_heads(args)
    @staticmethod
    def _find_encoder(core_method):
        target_model_name = core_method.replace("_", "").lower()
        for module_name in (
            "opencood.models.heter_encoders_isaac",
            "opencood.models.heter_encoders",
        ):
            encoder_lib = importlib.import_module(module_name)
            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name:
                    return cls
        raise RuntimeError(f"Unknown encoder {core_method}")

    @staticmethod
    def _make_prediction_head(in_channels, out_channels, head_cfg):
        hidden_channels = int(head_cfg.get("hidden_channels", 64))
        num_conv = int(head_cfg.get("num_conv", 1))
        use_batch_norm = bool(head_cfg.get("use_batch_norm", True))
        layers = []
        cur_channels = in_channels
        for _ in range(num_conv):
            layers.append(
                nn.Conv2d(
                    cur_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not use_batch_norm,
                )
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.ReLU(inplace=True))
            cur_channels = hidden_channels
        layers.append(nn.Conv2d(cur_channels, out_channels, kernel_size=1))
        return nn.Sequential(*layers)

    def _replace_anchor_heads(self, args):
        anchor_num = int(args["anchor_number"])
        num_bins = int(args["dir_args"]["num_bins"])
        default_head_cfg = args.get("anchor_head", {})

        for modality_name in self.modality_name_list:
            if self.head_type_dict[modality_name] != "anchor_based":
                continue
            model_setting = args[modality_name]
            head_args = model_setting["head_args"]
            head_cfg = dict(default_head_cfg)
            head_cfg.update(head_args.get("anchor_head", {}))
            in_head = int(head_args["in_head"])

            cls_head = self._make_prediction_head(
                in_head, anchor_num * self.isaac_num_classes, head_cfg
            )
            init_bias = head_cfg.get("init_bias")
            if init_bias is not None:
                cls_head[-1].bias.data.fill_(float(init_bias))

            setattr(self, f"cls_head_{modality_name}", cls_head)
            setattr(
                self,
                f"reg_head_{modality_name}",
                self._make_prediction_head(in_head, anchor_num * 7, head_cfg),
            )
            setattr(
                self,
                f"dir_head_{modality_name}",
                self._make_prediction_head(in_head, anchor_num * num_bins, head_cfg),
            )
