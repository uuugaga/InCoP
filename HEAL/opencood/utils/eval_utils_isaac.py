# -*- coding: utf-8 -*-
"""Isaac-specific CenterHead postprocess and class-aware AP utilities."""

import os

import numpy as np
import torch

from opencood.hypes_yaml import yaml_utils
from opencood.utils.center_head_utils import decode_center_boxes_for_output
from opencood.utils import box_utils, common_utils


def init_multiclass_result_stat(class_names):
    return {
        thresh: {
            class_name: {"tp": [], "fp": [], "gt": 0, "score": []}
            for class_name in class_names
        }
        for thresh in (0.3, 0.5, 0.7)
    }


def _calculate_tp_fp_single_class(det_boxes, det_score, gt_boxes, result_item, iou_thresh):
    fp = []
    tp = []
    gt = 0 if gt_boxes is None else gt_boxes.shape[0]

    if det_boxes is not None and det_boxes.shape[0] > 0:
        det_boxes_np = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score_np = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes_np = (
            np.zeros((0, 8, 3), dtype=det_boxes_np.dtype)
            if gt_boxes is None
            else common_utils.torch_tensor_to_numpy(gt_boxes)
        )

        score_order_descend = np.argsort(-det_score_np)
        det_score_np = det_score_np[score_order_descend]
        det_polygon_list = list(common_utils.convert_format(det_boxes_np))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes_np))

        for det_idx in score_order_descend:
            det_polygon = det_polygon_list[det_idx]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)
            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)
            gt_polygon_list.pop(np.argmax(ious))

        result_item["score"] += det_score_np.tolist()

    result_item["fp"] += fp
    result_item["tp"] += tp
    result_item["gt"] += gt


def calculate_tp_fp_multiclass_isaac(
    det_boxes,
    det_score,
    det_labels,
    gt_boxes,
    gt_labels,
    result_stat,
    iou_thresh,
    class_names,
):
    if det_labels is None or gt_labels is None:
        return
    det_labels = det_labels.long()
    gt_labels = gt_labels.long()

    for class_id, class_name in enumerate(class_names):
        det_mask = det_labels == class_id
        gt_mask = gt_labels == class_id
        cur_det_boxes = None
        cur_det_score = None
        if det_boxes is not None and det_mask.any():
            cur_det_boxes = det_boxes[det_mask]
            cur_det_score = det_score[det_mask]
        cur_gt_boxes = gt_boxes[gt_mask] if gt_boxes is not None else None
        _calculate_tp_fp_single_class(
            cur_det_boxes,
            cur_det_score,
            cur_gt_boxes,
            result_stat[iou_thresh][class_name],
            iou_thresh,
        )


def _calculate_ap_from_item(result_item):
    if result_item["gt"] == 0:
        return None, [], []

    fp = np.asarray(result_item["fp"])
    tp = np.asarray(result_item["tp"])
    score = np.asarray(result_item["score"])
    if len(fp) == 0:
        return 0.0, [0.0, 1.0], [0.0, 0.0]

    sorted_index = np.argsort(-score)
    fp = fp[sorted_index].tolist()
    tp = tp[sorted_index].tolist()

    fp = np.cumsum(fp).tolist()
    tp = np.cumsum(tp).tolist()
    rec = [float(x) / result_item["gt"] for x in tp]
    prec = [float(t) / max(float(t + f), 1e-12) for t, f in zip(tp, fp)]

    rec_for_ap = [0.0] + rec + [1.0]
    prec_for_ap = [0.0] + prec + [0.0]
    for idx in range(len(prec_for_ap) - 2, -1, -1):
        prec_for_ap[idx] = max(prec_for_ap[idx], prec_for_ap[idx + 1])

    ap = 0.0
    for idx in range(1, len(rec_for_ap)):
        if rec_for_ap[idx] != rec_for_ap[idx - 1]:
            ap += (rec_for_ap[idx] - rec_for_ap[idx - 1]) * prec_for_ap[idx]
    return ap, rec_for_ap, prec_for_ap


def eval_final_results_multiclass_isaac(result_stat, save_path, class_names, infer_info=None):
    dump_dict = {}
    for iou_thresh in (0.3, 0.5, 0.7):
        per_class = {}
        micro = {"tp": [], "fp": [], "gt": 0, "score": []}
        weighted_sum = 0.0
        weighted_gt = 0
        macro_values = []

        for class_name in class_names:
            item = result_stat[iou_thresh][class_name]
            ap, _, _ = _calculate_ap_from_item(item)
            per_class[class_name] = {
                "ap": 0.0 if ap is None else ap,
                "gt": item["gt"],
                "num_pred": len(item["score"]),
            }
            micro["tp"] += item["tp"]
            micro["fp"] += item["fp"]
            micro["score"] += item["score"]
            micro["gt"] += item["gt"]
            if ap is not None:
                macro_values.append(ap)
                weighted_sum += ap * item["gt"]
                weighted_gt += item["gt"]

        micro_ap, _, _ = _calculate_ap_from_item(micro)
        dump_dict[f"iou_{iou_thresh:.1f}"] = {
            "mAP": float(np.mean(macro_values)) if macro_values else 0.0,
            "weighted_mAP": weighted_sum / weighted_gt if weighted_gt > 0 else 0.0,
            "micro_AP": 0.0 if micro_ap is None else micro_ap,
            "per_class": per_class,
        }

    save_name = "eval_multiclass_isaac.yaml"
    if infer_info is not None:
        save_name = f"eval_multiclass_isaac_{infer_info}.yaml"
    yaml_utils.save_yaml(dump_dict, os.path.join(save_path, save_name))

    print(
        "Isaac multi-class mAP: "
        f"AP@0.3={dump_dict['iou_0.3']['mAP']:.4f}, "
        f"AP@0.5={dump_dict['iou_0.5']['mAP']:.4f}, "
        f"AP@0.7={dump_dict['iou_0.7']['mAP']:.4f}"
    )
    return dump_dict


def _select_topk_center_head(pred_dict, score_threshold, pre_nms_topk):
    heatmap = pred_dict["hm"].sigmoid()[0]
    num_classes, height, width = heatmap.shape
    flat_scores = heatmap.permute(1, 2, 0).reshape(-1)
    keep = flat_scores > score_threshold
    if not keep.any():
        device = heatmap.device
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=heatmap.dtype, device=device),
        )

    scores = flat_scores[keep]
    flat_inds = torch.nonzero(keep, as_tuple=False).view(-1)
    if scores.numel() > pre_nms_topk:
        scores, order = torch.topk(scores, pre_nms_topk)
        flat_inds = flat_inds[order]

    labels = flat_inds % num_classes
    cell_inds = flat_inds // num_classes
    return cell_inds.long(), labels.long(), scores


def center_head_post_process_isaac(data_dict, output_dict, postprocess_params):
    pred_box3d_list = []
    pred_score_list = []
    pred_label_list = []
    score_threshold = postprocess_params["target_args"].get("score_threshold", 0.1)
    pre_nms_topk = postprocess_params.get("pre_nms_topk", 4096)
    post_nms_topk = postprocess_params.get("post_nms_topk", 512)

    for cav_id, cav_output in output_dict.items():
        if cav_id not in data_dict or "center_head_preds" not in cav_output:
            continue
        cell_inds, labels, scores = _select_topk_center_head(
            cav_output["center_head_preds"], score_threshold, pre_nms_topk
        )
        if scores.numel() == 0:
            continue
        boxes3d = decode_center_boxes_for_output(
            cav_output,
            cell_inds,
            fallback_params=postprocess_params,
        )

        boxes3d_corner = box_utils.boxes_to_corners_3d(
            boxes3d, order=postprocess_params["order"]
        )
        projected_boxes3d = box_utils.project_box3d(
            boxes3d_corner, data_dict[cav_id]["transformation_matrix"]
        )
        pred_box3d_list.append(projected_boxes3d)
        pred_score_list.append(scores)
        pred_label_list.append(labels)

    if not pred_box3d_list:
        gt_box3d_tensor, gt_labels = generate_gt_bbx_with_classes_isaac(
            data_dict, postprocess_params
        )
        return None, None, None, gt_box3d_tensor, gt_labels

    pred_box3d_tensor = torch.vstack(pred_box3d_list)
    scores = torch.cat(pred_score_list)
    labels = torch.cat(pred_label_list)

    keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
    keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
    keep_index = torch.logical_and(keep_index_1, keep_index_2)
    pred_box3d_tensor = pred_box3d_tensor[keep_index]
    scores = scores[keep_index]
    labels = labels[keep_index]

    keep_all = []
    for class_id in torch.unique(labels):
        class_mask = labels == class_id
        class_indices = torch.nonzero(class_mask, as_tuple=False).view(-1)
        class_keep = box_utils.nms_rotated(
            pred_box3d_tensor[class_indices],
            scores[class_indices],
            postprocess_params["nms_thresh"],
        )
        keep_all.append(class_indices[class_keep])
    if keep_all:
        keep_index = torch.cat(keep_all)
        _, order = torch.sort(scores[keep_index], descending=True)
        keep_index = keep_index[order[:post_nms_topk]]
        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]
        labels = labels[keep_index]

    pred_box3d_np = pred_box3d_tensor.detach().cpu().numpy()
    pred_box3d_np, mask = box_utils.mask_boxes_outside_range_numpy(
        pred_box3d_np,
        postprocess_params["gt_range"],
        order=None,
        return_mask=True,
    )
    pred_box3d_tensor = torch.from_numpy(pred_box3d_np).to(device=scores.device)
    mask = torch.from_numpy(mask).to(device=scores.device, dtype=torch.bool)
    scores = scores[mask]
    labels = labels[mask]

    gt_box3d_tensor, gt_labels = generate_gt_bbx_with_classes_isaac(
        data_dict, postprocess_params
    )
    return pred_box3d_tensor, scores, labels, gt_box3d_tensor, gt_labels


def generate_gt_bbx_with_classes_isaac(data_dict, postprocess_params):
    gt_box3d_list = []
    label_list = []
    object_id_list = []

    for _, cav_content in data_dict.items():
        transformation_matrix = cav_content["transformation_matrix_clean"]
        object_bbx_center = cav_content["object_bbx_center"]
        object_bbx_mask = cav_content["object_bbx_mask"].bool()
        object_ids = cav_content["object_ids"]
        class_ids = cav_content["label_dict"].get("object_class_ids")
        if class_ids is None:
            class_ids = torch.zeros_like(object_bbx_mask, dtype=torch.long)

        valid_boxes = object_bbx_center[object_bbx_mask]
        valid_classes = class_ids.to(object_bbx_mask.device).long()[object_bbx_mask]
        if valid_boxes.numel() == 0:
            continue

        object_bbx_corner = box_utils.boxes_to_corners_3d(
            valid_boxes, postprocess_params["order"]
        )
        projected_object_bbx_corner = box_utils.project_box3d(
            object_bbx_corner.float(), transformation_matrix
        )
        gt_box3d_list.append(projected_object_bbx_corner)
        label_list.append(valid_classes)
        object_id_list += object_ids

    if not gt_box3d_list:
        device = next(iter(data_dict.values()))["object_bbx_center"].device
        return (
            torch.empty((0, 8, 3), device=device),
            torch.empty((0,), dtype=torch.long, device=device),
        )

    gt_box3d_tensor = torch.vstack(gt_box3d_list)
    gt_labels = torch.cat(label_list).to(gt_box3d_tensor.device)

    seen = set()
    selected_indices = []
    for idx, object_id in enumerate(object_id_list):
        if object_id in seen:
            continue
        seen.add(object_id)
        selected_indices.append(idx)

    selected_indices = torch.as_tensor(
        selected_indices, dtype=torch.long, device=gt_box3d_tensor.device
    )
    gt_box3d_tensor = gt_box3d_tensor[selected_indices]
    gt_labels = gt_labels[selected_indices]

    mask = box_utils.get_mask_for_boxes_within_range_torch(
        gt_box3d_tensor, postprocess_params["gt_range"]
    )
    return gt_box3d_tensor[mask], gt_labels[mask]
