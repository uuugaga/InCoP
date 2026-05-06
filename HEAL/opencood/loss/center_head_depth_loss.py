# -*- coding: utf-8 -*-
"""CenterHead loss with optional LSS depth supervision."""

from opencood.loss.center_head_loss import CenterHeadLoss
from opencood.loss.point_pillar_depth_loss import FocalLoss


class CenterHeadDepthLoss(CenterHeadLoss):
    def __init__(self, args):
        super().__init__(args)
        self.depth = args.get("depth", {"weight": 1.0})
        self.depth_weight = self.depth.get("weight", 1.0)
        self.smooth_target = bool(self.depth.get("smooth_target", False))
        self.use_fg_mask = bool(self.depth.get("use_fg_mask", False))
        self.fg_weight = 3.25
        self.bg_weight = 0.25
        self.depth_loss_func = FocalLoss(
            alpha=0.25,
            gamma=2.0,
            reduction="none",
            smooth_target=self.smooth_target,
        )

    def forward(self, output_dict, target_dict, suffix=""):
        total_loss = super().forward(output_dict, target_dict, suffix)

        all_depth_loss = total_loss.new_tensor(0.0)
        depth_items_list = [
            key for key in output_dict.keys() if key.startswith(f"depth_items{suffix}")
        ]
        for depth_item_name in depth_items_list:
            depth_item = output_dict[depth_item_name]
            depth_logit, depth_gt_indices = depth_item[0], depth_item[1]
            depth_loss = self.depth_loss_func(depth_logit, depth_gt_indices)
            if self.use_fg_mask:
                fg_mask = depth_item[-1]
                weight_mask = (fg_mask > 0) * self.fg_weight + (fg_mask == 0) * self.bg_weight
                depth_loss = depth_loss * weight_mask
            all_depth_loss = all_depth_loss + depth_loss.mean() * self.depth_weight

        total_loss = total_loss + all_depth_loss
        self.loss_dict["depth_loss"] = all_depth_loss.item()
        self.loss_dict["total_loss"] = total_loss.item()
        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer=None, suffix=""):
        loss_text = (
            f"[epoch {epoch}][{batch_id + 1}/{batch_len}] || "
            f"Loss: {self.loss_dict['total_loss']:.4f} || "
            f"HM Loss: {self.loss_dict['hm_loss']:.4f} || "
            f"Loc Loss: {self.loss_dict['loc_loss']:.4f} || "
            f"Depth Loss: {self.loss_dict['depth_loss']:.4f} || "
            f"Pos: {self.loss_dict['num_pos']:.0f}"
        )
        print(loss_text)
        if writer is not None:
            step = epoch * batch_len + batch_id
            for key, value in self.loss_dict.items():
                writer.add_scalar(key + suffix, value, step)
