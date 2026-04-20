#!/bin/bash

# pyramid

CUDA_VISIBLE_DEVICES=3 python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/opv2v/LiDAROnly/lidar_pyramid.yaml 
CUDA_VISIBLE_DEVICES=7 python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/opv2v/CameraOnly/camera_pyramid.yaml


# V2X-ViT

CUDA_VISIBLE_DEVICES=7 python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/opv2v/LiDAROnly/lidar_v2xvit.yaml
CUDA_VISIBLE_DEVICES=7 python opencood/tools/inference.py --model_dir opencood/logs/HeterBaseline_opv2v_lidar_v2xvit_2026_04_13_08_17_40 --fusion_method intermediate
CUDA_VISIBLE_DEVICES=6 python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/opv2v/CameraOnly/camera_v2xvit.yaml



# Single

CUDA_VISIBLE_DEVICES=3 python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/opv2v/Single/m1_pointpillar_pretrain.yaml
CUDA_VISIBLE_DEVICES=2 python opencood/tools/inference.py --model_dir opencood/logs/HEAL_opv2v_m1_pointpillar_pretrain_2026_04_14_15_07_02 --fusion_method no
CUDA_VISIBLE_DEVICES=2 python opencood/tools/inference.py --model_dir opencood/logs/HEAL_opv2v_m1_pointpillar_pretrain_2026_04_14_15_07_02 --fusion_method late

CUDA_VISIBLE_DEVICES=2 python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/opv2v/Single/m2_LSSeff48_pretrain.yaml
