# -*- coding: utf-8 -*-
"""CenterPoint/CenterHead loss for OpenCOOD target dictionaries."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterHeadLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.lidar_range = args["lidar_range"]
        self.num_classes = args.get("num_classes", 1)
        self.num_max_objs = args.get("num_max_objs", 1024)
        self.gaussian_overlap = args.get("gaussian_overlap", 0.1)
        self.min_radius = args.get("min_radius", 2)
        self.cls_weight = args.get("cls_weight", 1.0)
        self.loc_weight = args.get("loc_weight", 1.0)
        self.code_weights = args.get(
            "code_weights", [1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5]
        )
        self.loss_dict = {}

    @staticmethod
    def _gaussian_radius(height, width, min_overlap=0.1):
        a1 = 1.0
        b1 = height + width
        c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
        sq1 = torch.sqrt(torch.clamp(b1**2 - 4.0 * a1 * c1, min=0.0))
        r1 = (b1 + sq1) / 2.0

        a2 = 4.0
        b2 = 2.0 * (height + width)
        c2 = (1.0 - min_overlap) * width * height
        sq2 = torch.sqrt(torch.clamp(b2**2 - 4.0 * a2 * c2, min=0.0))
        r2 = (b2 + sq2) / 2.0

        a3 = 4.0 * min_overlap
        b3 = -2.0 * min_overlap * (height + width)
        c3 = (min_overlap - 1.0) * width * height
        sq3 = torch.sqrt(torch.clamp(b3**2 - 4.0 * a3 * c3, min=0.0))
        r3 = (b3 + sq3) / 2.0

        return torch.min(torch.min(r1, r2), r3)

    @staticmethod
    def _draw_gaussian_to_heatmap(heatmap, center_x, center_y, radius):
        radius = int(radius)
        diameter = 2 * radius + 1
        sigma = diameter / 6.0
        height, width = heatmap.shape

        x = int(center_x)
        y = int(center_y)
        left = min(x, radius)
        right = min(width - x, radius + 1)
        top = min(y, radius)
        bottom = min(height - y, radius + 1)
        if min(left, right, top, bottom) < 0:
            return

        yy = torch.arange(
            -radius,
            radius + 1,
            device=heatmap.device,
            dtype=heatmap.dtype,
        )
        xx = torch.arange(
            -radius,
            radius + 1,
            device=heatmap.device,
            dtype=heatmap.dtype,
        )
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        gaussian = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))

        masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
        masked_gaussian = gaussian[
            radius - top : radius + bottom,
            radius - left : radius + right,
        ]
        if min(masked_heatmap.shape) > 0 and min(masked_gaussian.shape) > 0:
            torch.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)

    @staticmethod
    def _transpose_and_gather_feat(feat, inds):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        inds = inds.unsqueeze(-1).expand(inds.size(0), inds.size(1), feat.size(2))
        return feat.gather(1, inds)

    def _get_optional_class_ids(self, target_dict, batch_size, max_boxes, device):
        candidate_keys = (
            "object_class_ids",
            "object_bbx_class",
            "object_bbx_classes",
            "class_ids",
        )
        for key in candidate_keys:
            if key in target_dict:
                class_ids = target_dict[key].to(device=device).long()
                if class_ids.dim() == 1:
                    class_ids = class_ids.view(batch_size, max_boxes)
                return torch.clamp(class_ids, min=0, max=self.num_classes - 1)
        return None

    def _build_targets(self, pred_dict, target_dict):
        hm_logits = pred_dict["hm"]
        batch_size, _, height, width = hm_logits.shape
        device = hm_logits.device
        dtype = hm_logits.dtype
        x_min, y_min, _, x_max, y_max, _ = self.lidar_range
        stride_x = (x_max - x_min) / float(width)
        stride_y = (y_max - y_min) / float(height)

        heatmap = torch.zeros(
            (batch_size, self.num_classes, height, width),
            device=device,
            dtype=dtype,
        )
        target_boxes = torch.zeros(
            (batch_size, self.num_max_objs, 8),
            device=device,
            dtype=dtype,
        )
        inds = torch.zeros(
            (batch_size, self.num_max_objs), device=device, dtype=torch.long
        )
        masks = torch.zeros(
            (batch_size, self.num_max_objs), device=device, dtype=torch.bool
        )

        boxes = target_dict["object_bbx_center"].to(device=device, dtype=dtype)
        box_masks = target_dict["object_bbx_mask"].to(device=device).bool()
        class_ids = self._get_optional_class_ids(
            target_dict, batch_size, boxes.shape[1], device
        )

        for b in range(batch_size):
            out_idx = 0
            valid_boxes = boxes[b][box_masks[b]]
            valid_classes = None if class_ids is None else class_ids[b][box_masks[b]]

            for box_i, box in enumerate(valid_boxes):
                if out_idx >= self.num_max_objs:
                    break

                x, y, z, h, w, l, yaw = box
                if not (x_min <= x < x_max and y_min <= y < y_max):
                    continue

                coord_x = (x - x_min) / stride_x
                coord_y = (y - y_min) / stride_y
                center_x = torch.clamp(coord_x, min=0.0, max=width - 0.5)
                center_y = torch.clamp(coord_y, min=0.0, max=height - 0.5)
                center_x_int = center_x.int()
                center_y_int = center_y.int()

                size_x = torch.clamp(l / stride_x, min=1e-2)
                size_y = torch.clamp(w / stride_y, min=1e-2)
                radius = self._gaussian_radius(
                    size_y, size_x, min_overlap=self.gaussian_overlap
                )
                radius = max(int(radius.item()), self.min_radius)

                cls_id = 0
                if valid_classes is not None:
                    cls_id = int(valid_classes[box_i].item())

                self._draw_gaussian_to_heatmap(
                    heatmap[b, cls_id],
                    center_x_int.item(),
                    center_y_int.item(),
                    radius,
                )
                inds[b, out_idx] = center_y_int.long() * width + center_x_int.long()
                masks[b, out_idx] = True
                target_boxes[b, out_idx, 0] = center_x - center_x_int.float()
                target_boxes[b, out_idx, 1] = center_y - center_y_int.float()
                target_boxes[b, out_idx, 2] = z
                target_boxes[b, out_idx, 3:6] = torch.log(
                    torch.clamp(torch.stack([h, w, l]), min=1e-2)
                )
                target_boxes[b, out_idx, 6] = torch.cos(yaw)
                target_boxes[b, out_idx, 7] = torch.sin(yaw)
                out_idx += 1

        return heatmap, target_boxes, inds, masks

    @staticmethod
    def _focal_loss_center_net(pred_logits, target):
        pred = torch.sigmoid(pred_logits).clamp(min=1e-4, max=1.0 - 1e-4)
        pos_inds = target.eq(1.0)
        neg_inds = target.lt(1.0)
        neg_weights = torch.pow(1.0 - target, 4)

        pos_loss = -torch.log(pred) * torch.pow(1.0 - pred, 2) * pos_inds
        neg_loss = (
            -torch.log(1.0 - pred)
            * torch.pow(pred, 2)
            * neg_weights
            * neg_inds
        )
        num_pos = torch.clamp(pos_inds.float().sum(), min=1.0)
        return (pos_loss.sum() + neg_loss.sum()) / num_pos

    def forward(self, output_dict, target_dict, suffix=""):
        pred_dict = output_dict.get("center_head_preds", None)
        if pred_dict is None:
            pred_dict = {
                "hm": output_dict["center_preds"],
                "center": output_dict["offset_preds"],
                "center_z": output_dict["z_preds"],
                "dim": output_dict["size_preds"],
                "rot": output_dict["yaw_preds"],
            }

        heatmap, target_boxes, inds, masks = self._build_targets(
            pred_dict, target_dict
        )
        hm_loss = self._focal_loss_center_net(pred_dict["hm"], heatmap)

        pred_boxes = torch.cat(
            [
                pred_dict["center"],
                pred_dict["center_z"],
                pred_dict["dim"],
                pred_dict["rot"],
            ],
            dim=1,
        )
        pred_boxes = self._transpose_and_gather_feat(pred_boxes, inds)
        mask = masks.unsqueeze(-1).float()
        num_pos = torch.clamp(masks.float().sum(), min=1.0)

        code_weights = torch.as_tensor(
            self.code_weights, device=pred_boxes.device, dtype=pred_boxes.dtype
        ).view(1, 1, -1)
        loc_loss = F.l1_loss(
            pred_boxes * mask,
            target_boxes * mask,
            reduction="none",
        )
        loc_loss = (loc_loss * code_weights).sum() / num_pos
        total_loss = self.cls_weight * hm_loss + self.loc_weight * loc_loss

        with torch.no_grad():
            loc_per_code = (loc_loss.detach() / code_weights.numel()).item()
        self.loss_dict = {
            "total_loss": total_loss.item(),
            "hm_loss": hm_loss.item(),
            "loc_loss": loc_loss.item(),
            "loc_per_code": loc_per_code,
            "num_pos": float(masks.float().sum().item()),
        }
        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer=None, suffix=""):
        loss_text = (
            f"[epoch {epoch}][{batch_id + 1}/{batch_len}] || "
            f"Loss: {self.loss_dict['total_loss']:.4f} || "
            f"HM Loss: {self.loss_dict['hm_loss']:.4f} || "
            f"Loc Loss: {self.loss_dict['loc_loss']:.4f} || "
            f"Pos: {self.loss_dict['num_pos']:.0f}"
        )
        print(loss_text)
        if writer is not None:
            step = epoch * batch_len + batch_id
            for key, value in self.loss_dict.items():
                if math.isfinite(value):
                    writer.add_scalar(key + suffix, value, step)
