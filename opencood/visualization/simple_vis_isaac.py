"""IsaacSim-specific lightweight visualization helpers."""

import copy

import cv2
import numpy as np

import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev
from opencood.visualization import simple_vis


PREDICTION_CLASS_NAMES = (
    "potted_plant",
    "chair",
    "medical_bag",
    "traffic_cone",
    "wet_floor_sign",
    "fire_extinguisher",
    "trash_can",
)

PREDICTION_CLASS_LABELS = (
    "PL",
    "CH",
    "MB",
    "TC",
    "WF",
    "FE",
    "TR",
)

# RGB colors intentionally avoid white, blue, and green so prediction boxes
# cannot be confused with ego lidar, partner lidar, or ground-truth boxes.
PREDICTION_CLASS_COLORS = np.asarray(
    [
        (125, 0, 255),    # potted_plant: electric violet
        (255, 125, 205),  # chair: light neon pink
        (190, 0, 90),     # medical_bag: deep raspberry
        (255, 110, 0),    # traffic_cone: vivid orange
        (255, 240, 0),    # wet_floor_sign: fluorescent yellow
        (255, 25, 25),    # fire_extinguisher: signal red
        (145, 145, 145),  # trash_can: neutral gray
    ],
    dtype=np.uint8,
)
PREDICTION_FALLBACK_COLOR = np.asarray((128, 64, 0), dtype=np.uint8)


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


def _prediction_styles(infer_result, box_count):
    labels = _to_numpy(infer_result.get("pred_label", None))
    if labels is not None:
        labels = np.asarray(labels).reshape(-1)

    colors = []
    texts = []
    for box_idx in range(box_count):
        label_id = int(labels[box_idx]) if labels is not None and box_idx < len(labels) else -1
        if 0 <= label_id < len(PREDICTION_CLASS_NAMES):
            colors.append(PREDICTION_CLASS_COLORS[label_id])
            texts.append(PREDICTION_CLASS_LABELS[label_id])
        else:
            colors.append(PREDICTION_FALLBACK_COLOR)
            texts.append("UNK")
    return np.asarray(colors, dtype=np.uint8), texts


def visualize(infer_result, pcd, pc_range, save_path=None, method="bev", left_hand=False,
              pixels_per_meter=24, return_image=False):
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
        # IsaacSim OPV2V exports share a mirrored image-row convention here;
        # keep all Isaac BEV snapshots in the corrected orientation.
        flip_y_axis=True,
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
        canvas.draw_boxes(gt_box_np, colors=(0, 255, 0), texts=[""] * len(gt_box_np),
                          box_line_thickness=2, box_text_size=0.4)

    pred_box_np = _to_numpy(infer_result.get("pred_box_tensor", None))
    if pred_box_np is not None:
        pred_colors, pred_texts = _prediction_styles(infer_result, len(pred_box_np))
        for box, color, _text in zip(pred_box_np, pred_colors, pred_texts):
            canvas.draw_boxes(
                box[None, ...],
                colors=tuple(int(value) for value in color),
                texts=[""],
                box_line_thickness=2,
                box_text_size=0.4,
            )

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

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(canvas.canvas, cv2.COLOR_RGB2BGR))
    if return_image:
        return canvas.canvas.copy()
    return None
