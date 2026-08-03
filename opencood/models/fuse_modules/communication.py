# -*- coding: utf-8 -*-
"""Communication accounting shared by cooperative fusion baselines.

The papers normally report feature payload bytes and ignore protocol headers.
This helper keeps feature values, sparse indices, and metadata/control values
separate so both the paper convention and a more realistic total are visible.
"""

from copy import deepcopy

import torch.nn.functional as F
from torch import nn


class SpatialDownsampleAdapter(nn.Module):
    """Learned stride-2 encoder/decoder around a fusion-resolution grid.

    HEAL supplies a common 112x112 message feature, whereas several official
    baselines collaborate at a lower backbone level.  This adapter preserves
    the public fusion-module input/output shape while making the actual
    transmitted and fused grid method-specific.  The convolutions are
    depthwise so this geometry adapter does not give one baseline a large
    extra channel-mixing parameter budget.
    """

    def __init__(self, channels, stages=0):
        super(SpatialDownsampleAdapter, self).__init__()
        self.channels = int(channels)
        self.stages = int(stages)
        if self.channels <= 0:
            raise ValueError("Spatial adapter channels must be positive.")
        if self.stages < 0:
            raise ValueError("Spatial downsample stages cannot be negative.")
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    self.channels, self.channels, kernel_size=3, stride=2,
                    padding=1, groups=self.channels, bias=False,
                ),
                nn.BatchNorm2d(self.channels, eps=1.0e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            )
            for _ in range(self.stages)
        ])
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(
                    self.channels, self.channels, kernel_size=2, stride=2,
                    groups=self.channels, bias=False,
                ),
                nn.BatchNorm2d(self.channels, eps=1.0e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            )
            for _ in range(self.stages)
        ])

    def encode(self, x):
        for stage in self.encoder:
            x = stage(x)
        return x

    def decode(self, x, output_size):
        for stage in self.decoder:
            x = stage(x)
        if x.shape[-2:] != tuple(output_size):
            x = F.interpolate(
                x, size=tuple(output_size), mode="bilinear", align_corners=False
            )
        return x


def count_trainable_parameters(module):
    """Return the number of trainable parameters in ``module``."""

    return sum(parameter.numel() for parameter in module.parameters()
               if parameter.requires_grad)


def count_all_parameters(module):
    """Return the number of trainable and frozen parameters in ``module``."""

    return sum(parameter.numel() for parameter in module.parameters())


class CommunicationStatsMixin:
    """Mixin that records the communication produced by the latest forward.

    Parameters are read from the fusion-method YAML block:

    ``communication_dtype_bytes``
        Bytes for one transmitted feature scalar. Defaults to FP32 (4).
    ``communication_index_bytes``
        Bytes for one sparse index. Defaults to int32 (4).
    ``communication_metadata_dtype_bytes``
        Bytes for one metadata/control scalar. Defaults to FP32 (4).
    """

    def _init_communication_stats(self, args=None, method=None):
        args = args or {}
        self.communication_method = method or self.__class__.__name__
        self.communication_dtype_bytes = int(
            args.get("communication_dtype_bytes", 4)
        )
        self.communication_index_bytes = int(
            args.get("communication_index_bytes", 4)
        )
        self.communication_metadata_dtype_bytes = int(
            args.get("communication_metadata_dtype_bytes", 4)
        )
        for name, value in (
            ("communication_dtype_bytes", self.communication_dtype_bytes),
            ("communication_index_bytes", self.communication_index_bytes),
            (
                "communication_metadata_dtype_bytes",
                self.communication_metadata_dtype_bytes,
            ),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        self.latest_communication_stats = None

    def _record_communication(
        self,
        *,
        feature_elements,
        index_elements=0,
        metadata_elements=0,
        message_count=0,
        batch_size=1,
        dense_feature_elements=None,
        mode="dense_bev",
        rounds=1,
        per_sample_feature_elements=None,
        notes=None,
    ):
        feature_elements = int(feature_elements)
        index_elements = int(index_elements)
        metadata_elements = int(metadata_elements)
        message_count = int(message_count)
        batch_size = max(int(batch_size), 1)

        feature_bytes = feature_elements * self.communication_dtype_bytes
        index_bytes = index_elements * self.communication_index_bytes
        metadata_bytes = (
            metadata_elements * self.communication_metadata_dtype_bytes
        )
        total_bytes = feature_bytes + index_bytes + metadata_bytes
        dense_feature_bytes = None
        compression_ratio = None
        if dense_feature_elements is not None:
            dense_feature_bytes = (
                int(dense_feature_elements) * self.communication_dtype_bytes
            )
            if dense_feature_bytes > 0:
                compression_ratio = float(feature_bytes) / dense_feature_bytes

        per_sample_bytes = float(total_bytes) / batch_size
        stats = {
            "method": self.communication_method,
            "mode": mode,
            "rounds": int(rounds),
            "batch_size": batch_size,
            "message_count": message_count,
            "feature_elements": feature_elements,
            "index_elements": index_elements,
            "metadata_elements": metadata_elements,
            "feature_dtype_bytes": self.communication_dtype_bytes,
            "index_dtype_bytes": self.communication_index_bytes,
            "metadata_dtype_bytes": self.communication_metadata_dtype_bytes,
            "feature_bytes": feature_bytes,
            "index_bytes": index_bytes,
            "metadata_bytes": metadata_bytes,
            "total_bytes": total_bytes,
            "total_KB": float(total_bytes) / 1000.0,
            "total_MB": float(total_bytes) / 1_000_000.0,
            "total_KiB": float(total_bytes) / 1024.0,
            "total_MiB": float(total_bytes) / (1024.0 ** 2),
            "per_sample_bytes": per_sample_bytes,
            "per_sample_KB": per_sample_bytes / 1000.0,
            "per_sample_MB": per_sample_bytes / 1_000_000.0,
        }
        if dense_feature_bytes is not None:
            stats["dense_feature_bytes"] = dense_feature_bytes
            stats["feature_payload_ratio_to_dense"] = compression_ratio
        if per_sample_feature_elements is not None:
            stats["per_sample_feature_elements"] = [
                int(value) for value in per_sample_feature_elements
            ]
        if notes:
            stats["notes"] = str(notes)
        self.latest_communication_stats = stats
        return stats

    def get_communication_stats(self):
        """Return a copy of communication statistics from the latest forward."""

        if self.latest_communication_stats is None:
            return None
        return deepcopy(self.latest_communication_stats)

    def parameter_count(self):
        """Return fusion-module parameter counts for audit scripts."""

        return {
            "all": count_all_parameters(self),
            "trainable": count_trainable_parameters(self),
        }
