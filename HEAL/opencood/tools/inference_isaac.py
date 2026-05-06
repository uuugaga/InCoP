# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import time
from collections import OrderedDict
import importlib
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.utils import eval_utils_isaac
from opencood.visualization import vis_utils, my_vis, simple_vis_isaac
from opencood.utils.common_utils import update_dict
from opencood.utils import box_utils
from opencood.utils.transformation_utils import x1_to_x2
torch.multiprocessing.set_sharing_strategy('file_system')


def is_isaac_center_head(hypes):
    loss_name = hypes.get("loss", {}).get("core_method", "")
    return loss_name in ("center_head_loss", "center_head_depth_loss")


def inference_isaac_center_head(batch_data, model, postprocess_params, fusion_method):
    """CenterHead inference path that preserves predicted/GT class ids."""
    if fusion_method == "late":
        output_dict = OrderedDict()
        for cav_id, cav_content in batch_data.items():
            output_dict[cav_id] = model(cav_content)
        post_data = batch_data
    elif fusion_method == "single":
        post_data = {"ego": batch_data["ego"]}
        output_dict = OrderedDict()
        output_dict["ego"] = model(batch_data["ego"])
    elif fusion_method in ("no", "no_w_uncertainty", "early", "intermediate"):
        output_dict = OrderedDict()
        output_dict["ego"] = model(batch_data["ego"])
        post_data = batch_data
    else:
        raise NotImplementedError(
            "Isaac CenterHead class-aware inference supports no, single, "
            "late, early, intermediate, and no_w_uncertainty fusion."
        )

    pred_box_tensor, pred_score, pred_label, gt_box_tensor, gt_label = (
        eval_utils_isaac.center_head_post_process_isaac(
            post_data, output_dict, postprocess_params
        )
    )
    return {
        "pred_box_tensor": pred_box_tensor,
        "pred_score": pred_score,
        "pred_label": pred_label,
        "gt_box_tensor": gt_box_tensor,
        "gt_label": gt_label,
    }


def _format_float_tag(value):
    if value is None:
        return "none"
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def build_infer_info(opt, hypes):
    postprocess_params = hypes["postprocess"]
    score_threshold = postprocess_params.get("target_args", {}).get(
        "score_threshold",
        postprocess_params.get("anchor_args", {}).get("score_threshold"),
    )
    nms_thresh = postprocess_params.get("nms_thresh")
    return (
        f"{opt.fusion_method}{opt.note}"
        f"_score{_format_float_tag(score_threshold)}"
        f"_nms{_format_float_tag(nms_thresh)}"
    )


def summarize_false_positive_metrics(result_stat, sample_count, save_path, infer_info):
    summary = {}
    sample_count = max(int(sample_count), 1)

    for iou_thresh, item in result_stat.items():
        tp_count = int(np.sum(item["tp"]))
        fp_count = int(np.sum(item["fp"]))
        gt_count = int(item["gt"])
        pred_count = tp_count + fp_count
        precision = tp_count / max(pred_count, 1)
        recall = tp_count / max(gt_count, 1)
        false_discovery_rate = fp_count / max(pred_count, 1)
        false_positives_per_image = fp_count / sample_count

        summary[f"iou_{iou_thresh:.1f}"] = {
            "tp": tp_count,
            "fp": fp_count,
            "gt": gt_count,
            "pred": pred_count,
            "precision": precision,
            "recall": recall,
            "false_discovery_rate": false_discovery_rate,
            "false_positives_per_image": false_positives_per_image,
        }

    yaml_utils.save_yaml(
        summary,
        os.path.join(save_path, f"eval_fp_metrics_{infer_info}.yaml"),
    )
    fp50 = summary["iou_0.5"]["false_positives_per_image"]
    prec50 = summary["iou_0.5"]["precision"]
    print(
        "False-positive summary: "
        f"FPPI@0.5={fp50:.3f}, Precision@0.5={prec50:.3f}"
    )
    return summary

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=200,
                        help='interval of saving BEV visualization; <=0 disables visualization')
    parser.add_argument('--save_vis_max', type=int, default=None,
                        help='optional maximum number of BEV visualizations to save')
    parser.add_argument('--save_npy', action='store_true',
                        help='deprecated for Isaac testing; npy export is ignored')
    parser.add_argument('--range', type=str, default="",
                        help="optional Isaac front range as x_min,y_min,x_max,y_max; defaults to the saved yaml range")
    parser.add_argument('--score_threshold', type=float, default=None,
                        help="optional override for postprocess target score_threshold")
    parser.add_argument('--nms_thresh', type=float, default=None,
                        help="optional override for postprocess.nms_thresh")
    parser.add_argument('--max_samples', type=int, default=None,
                        help="optional limit for quick Isaac visualization/debug runs")
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    opt = parser.parse_args()
    return opt


def build_isaac_multicav_lidar_for_vis(opencood_dataset, idx, ego_lidar):
    """Load one non-ego CAV lidar and project it into ego frame for BEV plots."""
    try:
        base_data_dict = opencood_dataset.retrieve_base_data(idx)
    except Exception as exc:
        print(f"Skip Isaac multi-CAV lidar visualization at sample {idx}: {exc}")
        return ego_lidar

    ego_id = None
    for cav_id, cav_content in base_data_dict.items():
        if cav_content.get("ego", False):
            ego_id = cav_id
            break
    if ego_id is None:
        return ego_lidar

    ego_pose = base_data_dict[ego_id]["params"]["lidar_pose"]
    extra_lidar = []
    for cav_id, cav_content in base_data_dict.items():
        if cav_id == ego_id:
            continue
        lidar_np = np.asarray(cav_content.get("lidar_np", []))
        if lidar_np.size == 0:
            continue
        transform = x1_to_x2(cav_content["params"]["lidar_pose"], ego_pose)
        projected_xyz = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transform)
        projected_lidar = np.zeros((projected_xyz.shape[0], max(4, lidar_np.shape[1])),
                                   dtype=np.float32)
        projected_lidar[:, :3] = projected_xyz
        if lidar_np.shape[1] > 3:
            projected_lidar[:, 3] = lidar_np[:, 3]
        extra_lidar.append(projected_lidar[:, :4])
        break

    if not extra_lidar:
        return ego_lidar
    return [ego_lidar] + extra_lidar


def main():
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)

    if opt.score_threshold is not None:
        if 'target_args' in hypes['postprocess']:
            hypes['postprocess']['target_args']['score_threshold'] = opt.score_threshold
        hypes['postprocess']['anchor_args']['score_threshold'] = opt.score_threshold
    if opt.nms_thresh is not None:
        hypes['postprocess']['nms_thresh'] = opt.nms_thresh

    if 'heter' in hypes and opt.range:
        # hypes['heter']['lidar_channels'] = 16
        # opt.note += "_16ch"

        range_values = [float(v.strip()) for v in opt.range.split(',') if v.strip()]
        is_isaac_dataset = "IsaacSim" in hypes.get("test_dir", "")
        if is_isaac_dataset and len(range_values) != 4:
            raise ValueError(
                "IsaacSim --range must be x_min,y_min,x_max,y_max "
                "to preserve the front-only lidar range."
            )
        if len(range_values) == 2:
            x_min, x_max = -range_values[0], range_values[0]
            y_min, y_max = -range_values[1], range_values[1]
        elif len(range_values) == 4:
            x_min, y_min, x_max, y_max = range_values
        else:
            raise ValueError("--range should be either x_half,y_half or x_min,y_min,x_max,y_max")
        opt.note += f"_{x_max}_{y_max}"

        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
                            x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]

        # replace all appearance
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range
        })

        # reload anchor
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)

        
    
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # setting noise
    np.random.seed(303)
    
    # build dataset for each noise setting
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    # opencood_dataset_subset = Subset(opencood_dataset, range(640,2100))
    # data_loader = DataLoader(opencood_dataset_subset,
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    center_head_eval = is_isaac_center_head(hypes)
    class_names = hypes.get("postprocess", {}).get("class_names", [])
    multiclass_result_stat = None
    if center_head_eval and class_names:
        multiclass_result_stat = eval_utils_isaac.init_multiclass_result_stat(class_names)

    if opt.save_npy:
        print("--save_npy is ignored by inference_isaac.py.")

    infer_info = build_infer_info(opt, hypes)
    evaluated_samples = 0
    saved_vis_count = 0
    print(f"Isaac BEV/debug range: {hypes['postprocess']['gt_range']}")
    print(
        "BEV visualization: "
        f"interval={opt.save_vis_interval}, max={opt.save_vis_max}"
    )

    for i, batch_data in enumerate(data_loader):
        if opt.max_samples is not None and i >= opt.max_samples:
            break
        print(f"{infer_info}_{i}")
        if batch_data is None:
            continue
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            if center_head_eval:
                infer_result = inference_isaac_center_head(
                    batch_data, model, hypes["postprocess"], opt.fusion_method
                )
            elif opt.fusion_method == 'late':
                infer_result = inference_utils.inference_late_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'early':
                infer_result = inference_utils.inference_early_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no_w_uncertainty':
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'single':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset,
                                                                single_gt=True)
            else:
                raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                        'fusion is supported.')

            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']
            evaluated_samples += 1
            
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.7)
            if multiclass_result_stat is not None:
                pred_label = infer_result.get("pred_label")
                gt_label = infer_result.get("gt_label")
                eval_utils_isaac.calculate_tp_fp_multiclass_isaac(
                    pred_box_tensor,
                    pred_score,
                    pred_label,
                    gt_box_tensor,
                    gt_label,
                    multiclass_result_stat,
                    0.3,
                    class_names,
                )
                eval_utils_isaac.calculate_tp_fp_multiclass_isaac(
                    pred_box_tensor,
                    pred_score,
                    pred_label,
                    gt_box_tensor,
                    gt_label,
                    multiclass_result_stat,
                    0.5,
                    class_names,
                )
                eval_utils_isaac.calculate_tp_fp_multiclass_isaac(
                    pred_box_tensor,
                    pred_score,
                    pred_label,
                    gt_box_tensor,
                    gt_label,
                    multiclass_result_stat,
                    0.7,
                    class_names,
                )

            if not opt.no_score:
                infer_result.update({'score_tensor': pred_score})

            if getattr(opencood_dataset, "heterogeneous", False):
                cav_box_np, agent_modality_list = inference_utils.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np, \
                                     "agent_modality_list": agent_modality_list})

            save_bev_vis = (
                opt.save_vis_interval > 0
                and i % opt.save_vis_interval == 0
                and (opt.save_vis_max is None or saved_vis_count < opt.save_vis_max)
                and (pred_box_tensor is not None or gt_box_tensor is not None)
            )
            if save_bev_vis:
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)

                # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                # simple_vis.visualize(infer_result,
                #                     batch_data['ego'][
                #                         'origin_lidar'][0],
                #                     hypes['postprocess']['gt_range'],
                #                     vis_save_path,
                #                     method='3d',
                #                     left_hand=left_hand)
                 
                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                vis_lidar = build_isaac_multicav_lidar_for_vis(
                                    opencood_dataset,
                                    i,
                                    batch_data['ego']['origin_lidar'][0])
                simple_vis_isaac.visualize(infer_result,
                                    vis_lidar,
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=left_hand)
                saved_vis_count += 1
        torch.cuda.empty_cache()

    if opt.max_samples is not None and all(v['gt'] == 0 for v in result_stat.values()):
        print("Skip AP calculation: no GT boxes were seen in this limited Isaac debug run.")
    else:
        _, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                    opt.model_dir, infer_info)
        summarize_false_positive_metrics(
            result_stat, evaluated_samples, opt.model_dir, infer_info
        )
        if multiclass_result_stat is not None:
            eval_utils_isaac.eval_final_results_multiclass_isaac(
                multiclass_result_stat, opt.model_dir, class_names, infer_info
            )

if __name__ == '__main__':
    main()
