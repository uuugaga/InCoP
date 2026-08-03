"""IsaacSim training entrypoint with fixed-ego validation.

This keeps the stock OpenCOOD ``train.py`` behavior untouched while making
IsaacSim best-validation checkpoint selection keep robot 0 as ego and robot 1
as the collaborative agent.
"""

import argparse
import os
import sys
from collections import OrderedDict

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import eval_utils, eval_utils_isaac
from opencood.tools.inference_isaac import (
    IOU_THRESHOLDS,
    inference_isaac_center_head,
    init_object_result_stat,
    calculate_ap_from_item_with_sort_option,
    summarize_multiclass_map,
)
from opencood.tools.isaac_pretrained_utils import (
    apply_isaac_freeze_config,
    load_isaac_pretrained_weights,
    set_isaac_frozen_modules_eval,
)

# inference_isaac.py uses file_system sharing for inference. Training passes
# many CPU tensors through multiprocessing DataLoader workers, so prefer the
# default file_descriptor path and avoid torch_shm_manager socket failures.
torch.multiprocessing.set_sharing_strategy("file_descriptor")


def train_parser():
    parser = argparse.ArgumentParser(description="IsaacSim fixed-ego training")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help="data generation yaml file needed")
    parser.add_argument("--model_dir", default="",
                        help="continued training path")
    parser.add_argument("--pretrained_model_dir", default="",
                        help="initialize weights from a trained model without resuming epoch/log dir")
    parser.add_argument("--fusion_method", "-f", default=None,
                        help="passed to inference; if omitted, infer from the config")
    parser.add_argument("--bestval_iou", type=float, default=0.3,
                        choices=list(IOU_THRESHOLDS),
                        help="IoU threshold used to choose the best validation checkpoint")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader worker count; overrides train_params.num_workers")
    parser.add_argument("--prefetch_factor", type=int, default=None,
                        help="DataLoader prefetch factor when num_workers > 0")
    return parser.parse_args()


def is_isaac_config(hypes):
    return (
        hypes.get("fusion", {}).get("dataset", "") == "isaacsim"
        or "IsaacSim" in hypes.get("validate_dir", "")
        or "IsaacSim" in hypes.get("test_dir", "")
    )


def build_isaac_validate_dataset(hypes):
    validate_dataset = build_dataset(hypes, visualize=False, train=False)
    if is_isaac_config(hypes):
        print("Isaac fixed-ego validation enabled: robot 0 is ego, robot 1 is collaborative.")
    return validate_dataset


def infer_default_fusion_method(hypes, opt_fusion_method):
    if opt_fusion_method is not None:
        return opt_fusion_method
    fusion_core_method = hypes.get("fusion", {}).get("core_method", "")
    if fusion_core_method in ("intermediate", "intermediateheter"):
        return "intermediate"
    if fusion_core_method in ("late", "lateheter"):
        return "late"
    return "no"


def summarize_object_map(result_stat, global_sort_detections=False):
    summary = OrderedDict()
    for iou_thresh in IOU_THRESHOLDS:
        ap, _, _ = calculate_ap_from_item_with_sort_option(
            result_stat[iou_thresh], global_sort_detections
        )
        summary[f"mAP@{iou_thresh:.1f}"] = 0.0 if ap is None else float(ap)
    return summary


def validate_isaac_ap(model, val_loader, hypes, device, fusion_method, bestval_iou):
    """Run Isaac validation and select bestval by 3D multiclass mAP."""
    model.eval()
    result_stat = init_object_result_stat()
    class_names = hypes.get("postprocess", {}).get("class_names", []) or []
    multiclass_result_stat = (
        eval_utils_isaac.init_multiclass_result_stat(class_names)
        if class_names else None
    )
    evaluated_samples = 0

    with torch.no_grad():
        for batch_data in val_loader:
            if batch_data is None:
                continue
            batch_data = train_utils.to_device(batch_data, device)

            infer_result = inference_isaac_center_head(
                batch_data, model, hypes['postprocess'], fusion_method
            )

            pred_box_tensor = infer_result['pred_box_tensor']
            pred_score = infer_result['pred_score']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_label = infer_result.get('pred_label')
            gt_label = infer_result.get('gt_label')
            class_gt_box_tensor = gt_box_tensor
            evaluated_samples += 1

            for iou_thresh in IOU_THRESHOLDS:
                eval_utils.caluclate_tp_fp(
                    pred_box_tensor,
                    pred_score,
                    gt_box_tensor,
                    result_stat,
                    iou_thresh,
                )

            if multiclass_result_stat is not None:
                if gt_label is None:
                    class_gt_box_tensor, gt_label = (
                        eval_utils_isaac.generate_gt_bbx_with_classes_isaac(
                            batch_data, hypes['postprocess']
                        )
                    )
                for iou_thresh in IOU_THRESHOLDS:
                    eval_utils_isaac.calculate_tp_fp_multiclass_isaac(
                        pred_box_tensor,
                        pred_score,
                        pred_label,
                        class_gt_box_tensor,
                        gt_label,
                        multiclass_result_stat,
                        iou_thresh,
                        class_names,
                    )

    object_map_summary = summarize_object_map(
        result_stat, global_sort_detections=True
    )
    if multiclass_result_stat is None:
        key = f'mAP@{bestval_iou:.1f}'
        selected_map = object_map_summary[key]
        return selected_map, object_map_summary, evaluated_samples

    multiclass_summary, _ = summarize_multiclass_map(
        multiclass_result_stat,
        class_names,
        global_sort_detections=True,
    )
    key = f'mAP@{bestval_iou:.1f}'
    selected_map = multiclass_summary[key]
    return selected_map, multiclass_summary, evaluated_samples


def main():
    opt = train_parser()
    if opt.model_dir and opt.pretrained_model_dir:
        raise ValueError("--model_dir and --pretrained_model_dir are mutually exclusive")
    print(f"Training with config: {opt.hypes_yaml}")
    print(f"Model directory: {opt.model_dir}")
    print(f"Pretrained model directory: {opt.pretrained_model_dir}")
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print("Dataset Building")
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    train_collate_dataset = opencood_train_dataset
    opencood_validate_dataset = build_isaac_validate_dataset(hypes)
    validate_collate_fn = opencood_validate_dataset.collate_batch_test

    num_workers = opt.num_workers
    if num_workers is None:
        num_workers = int(hypes.get("train_params", {}).get("num_workers", 4))
    prefetch_factor = opt.prefetch_factor
    if prefetch_factor is None:
        prefetch_factor = int(hypes.get("train_params", {}).get("prefetch_factor", 2))

    train_loader_kwargs = {
        "dataset": opencood_train_dataset,
        "batch_size": hypes["train_params"]["batch_size"],
        "num_workers": num_workers,
        "collate_fn": train_collate_dataset.collate_batch_train,
        "shuffle": True,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": True,
    }
    val_loader_kwargs = {
        "dataset": opencood_validate_dataset,
        "batch_size": 1,
        "num_workers": num_workers,
        "collate_fn": validate_collate_fn,
        "shuffle": False,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
        val_loader_kwargs["prefetch_factor"] = prefetch_factor
    print(f"DataLoader workers: {num_workers}, prefetch_factor: {prefetch_factor if num_workers > 0 else 'disabled'}")
    print(f"Torch multiprocessing sharing strategy: {torch.multiprocessing.get_sharing_strategy()}")

    train_loader = DataLoader(**train_loader_kwargs)
    val_loader = DataLoader(**val_loader_kwargs)

    print("Creating Model")
    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_val_ap = -1.0
    best_val_epoch = -1

    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        best_val_epoch = init_epoch
        print(f"resume from {init_epoch} epoch.")
    else:
        init_epoch = 0
        saved_path = train_utils.setup_train(hypes)
        if opt.pretrained_model_dir:
            _, model = train_utils.load_saved_model(opt.pretrained_model_dir, model)
            print(f"initialized weights from {opt.pretrained_model_dir}.")
        load_isaac_pretrained_weights(model, hypes, context="train")

    apply_isaac_freeze_config(model, hypes)
    criterion = train_utils.create_loss(hypes)
    optimizer = train_utils.setup_optimizer(hypes, model)
    scheduler = train_utils.setup_lr_schedular(
        hypes, optimizer, init_epoch=init_epoch if opt.model_dir else None
    )

    if torch.cuda.is_available():
        model.to(device)

    model_ema = train_utils.setup_model_ema(hypes, model, len(train_loader))

    writer = SummaryWriter(saved_path)

    print("Training start")
    epoches = hypes["train_params"]["epoches"]
    supervise_single_flag = (
        False if not hasattr(opencood_train_dataset, "supervise_single")
        else opencood_train_dataset.supervise_single
    )

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print("learning rate %f" % param_group["lr"])

        model.train()
        try:
            model.model_train_init()
        except Exception:
            print("No model_train_init function")
        set_isaac_frozen_modules_eval(model, hypes)

        for i, batch_data in enumerate(train_loader):
            model.zero_grad()
            optimizer.zero_grad()

            if batch_data is None or batch_data["ego"]["object_bbx_mask"].sum() == 0:
                continue
            batch_data = train_utils.to_device(batch_data, device)
            batch_data["ego"]["epoch"] = epoch
            output_dict = model(batch_data["ego"])
            final_loss = criterion(output_dict, batch_data["ego"]["label_dict"])
            criterion.logging(epoch, i, len(train_loader), writer)

            if supervise_single_flag:
                single_weight = hypes["train_params"].get("single_weight", 1)
                final_loss += criterion(
                    output_dict,
                    batch_data["ego"]["label_dict_single"],
                    suffix="_single",
                ) * single_weight
                criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")

            fusion_aux_losses = output_dict.get("fusion_aux_losses", {})
            fusion_aux_weights = output_dict.get("fusion_aux_loss_weights", {})
            for aux_name, aux_loss in fusion_aux_losses.items():
                if not torch.is_tensor(aux_loss):
                    continue
                aux_weight = float(fusion_aux_weights.get(aux_name, 1.0))
                final_loss = final_loss + aux_weight * aux_loss
                writer.add_scalar(
                    "FusionAux/" + aux_name,
                    float(aux_loss.detach().cpu().item()),
                    epoch * len(train_loader) + i,
                )

            final_loss.backward()
            optimizer.step()
            if model_ema is not None:
                model_ema.update(model)

        if epoch % hypes["train_params"]["save_freq"] == 0:
            torch.save(
                model.state_dict(),
                os.path.join(saved_path, "net_epoch%d.pth" % (epoch + 1)),
            )

        skip_eval_epoch0 = bool(hypes["train_params"].get("skip_eval_epoch0", False))
        should_eval = epoch % hypes['train_params']['eval_freq'] == 0
        if should_eval and skip_eval_epoch0 and epoch == 0:
            print("Skip validation at epoch 0 due to train_params.skip_eval_epoch0.")
            should_eval = False

        if should_eval:
            fusion_method = infer_default_fusion_method(hypes, opt.fusion_method)
            eval_with_ema = model_ema is not None and model_ema.use_for_eval
            if eval_with_ema:
                with model_ema.average_parameters(model):
                    val_map, val_map_summary, val_samples = validate_isaac_ap(
                        model,
                        val_loader,
                        hypes,
                        device,
                        fusion_method,
                        opt.bestval_iou,
                    )
            else:
                val_map, val_map_summary, val_samples = validate_isaac_ap(
                    model,
                    val_loader,
                    hypes,
                    device,
                    fusion_method,
                    opt.bestval_iou,
                )
            eval_label = 'EMA validation' if eval_with_ema else 'validation'
            print(
                f'At epoch {epoch}, Isaac {eval_label} '
                f'mAP@{opt.bestval_iou:.1f} is {val_map:.6f} '
                f'over {val_samples} samples'
            )
            if val_map_summary is not None:
                for key, value in val_map_summary.items():
                    if not key.startswith('mAP@'):
                        continue
                    writer.add_scalar(f'Validate_mAP/{key}', value, epoch)

            if val_map > best_val_ap:
                best_val_ap = val_map
                best_state_dict = (
                    model_ema.state_dict() if eval_with_ema else model.state_dict()
                )
                torch.save(
                    best_state_dict,
                    os.path.join(saved_path, 'net_epoch_bestval_at%d.pth' % (epoch + 1)),
                )
                if best_val_epoch != -1 and os.path.exists(
                    os.path.join(saved_path, 'net_epoch_bestval_at%d.pth' % best_val_epoch)
                ):
                    os.remove(
                        os.path.join(saved_path, 'net_epoch_bestval_at%d.pth' % best_val_epoch)
                    )
                best_val_epoch = epoch + 1

        scheduler.step(epoch)
        if hasattr(opencood_train_dataset, "reinitialize"):
            opencood_train_dataset.reinitialize()

    print("Training Finished, checkpoints saved to %s" % saved_path)


if __name__ == "__main__":
    main()
