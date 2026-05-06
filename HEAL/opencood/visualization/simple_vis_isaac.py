"""IsaacSim-specific lightweight visualization helpers."""

import copy

import cv2
import numpy as np

import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev
from opencood.visualization import simple_vis


def _to_numpy(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, (list, tuple)):
        return [_to_numpy(item) for item in tensor if item is not None]
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def _iter_lidar_clouds(pcd_np):
    """Return separate CAV point clouds when visualize data keeps that axis."""
    if pcd_np is None:
        return []
    if isinstance(pcd_np, (list, tuple)):
        clouds = []
        for item in pcd_np:
            clouds.extend(_iter_lidar_clouds(item))
        return clouds
    pcd_np = np.asarray(pcd_np)
    if pcd_np.ndim == 2:
        return [pcd_np]
    if pcd_np.ndim == 3:
        return [pcd_np[i] for i in range(pcd_np.shape[0])]
    return [pcd_np.reshape(-1, pcd_np.shape[-1])]


def visualize(infer_result, pcd, pc_range, save_path, method="bev", left_hand=False,
              pixels_per_meter=24):
    if method != "bev":
        return simple_vis.visualize(infer_result, pcd, pc_range, save_path,
                                    method=method, left_hand=left_hand)

    pc_range = [float(v) for v in pc_range]
    height = max(1, int(round((pc_range[3] - pc_range[0]) * pixels_per_meter)))
    width = max(1, int(round((pc_range[4] - pc_range[1]) * pixels_per_meter)))
    canvas = canvas_bev.Canvas_BEV_heading_right(
        canvas_shape=(height, width),
        canvas_x_range=(pc_range[0], pc_range[3]),
        canvas_y_range=(pc_range[1], pc_range[4]),
        left_hand=left_hand,
    )

    pcd_np = _to_numpy(pcd)
    lidar_colors = [
        (235, 235, 235),  # ego
        (0, 190, 255),    # first non-ego CAV
        (255, 170, 0),
        (120, 220, 120),
    ]
    for cav_idx, cav_pcd_np in enumerate(_iter_lidar_clouds(pcd_np)):
        canvas_xy, valid_mask = canvas.get_canvas_coords(cav_pcd_np)
        canvas.draw_canvas_points(
            canvas_xy[valid_mask],
            radius=1,
            colors=lidar_colors[min(cav_idx, len(lidar_colors) - 1)],
        )

    gt_box_np = _to_numpy(infer_result.get("gt_box_tensor", None))
    if gt_box_np is not None:
        canvas.draw_boxes(gt_box_np, colors=(0, 255, 0), texts=["gt"] * len(gt_box_np),
                          box_line_thickness=2, box_text_size=0.4)

    pred_box_np = _to_numpy(infer_result.get("pred_box_tensor", None))
    if pred_box_np is not None:
        score = _to_numpy(infer_result.get("score_tensor", None))
        if score is None:
            pred_name = ["pred"] * len(pred_box_np)
        else:
            pred_name = [f"{score[i]:.2f}" for i in range(len(score))]
        canvas.draw_boxes(pred_box_np, colors=(255, 0, 0), texts=pred_name,
                          box_line_thickness=2, box_text_size=0.4)

    agent_modality_list = infer_result.get("agent_modality_list", None)
    cav_box_np = infer_result.get("cav_box_np", None)
    if agent_modality_list is not None and cav_box_np is not None:
        cav_box_np = copy.deepcopy(cav_box_np)
        colors = {
            "m1": (0, 191, 255),
            "m2": (255, 185, 15),
            "m3": (138, 211, 222),
            "m4": (32, 60, 160),
        }
        for i, modality_name in enumerate(agent_modality_list):
            canvas.draw_boxes(
                cav_box_np[i:i + 1],
                colors=colors.get(modality_name, (66, 66, 66)),
                texts=[modality_name],
                box_line_thickness=2,
                box_text_size=0.4,
            )

    cv2.imwrite(save_path, cv2.cvtColor(canvas.canvas, cv2.COLOR_RGB2BGR))
