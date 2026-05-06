# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import copy
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.models.sub_modules.lss_submodule import Up, CamEncode, BevEncode, CamEncode_Resnet101
from opencood.utils.camera_utils import gen_dx_bx, cumsum_trick, QuickCumsum, depth_discretization
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression


class ConvFuser(nn.Sequential):
    """BEVFusion-style convolution fuser for aligned BEV features.

    The official BEVFusion ConvFuser concatenates sensor BEV features and
    applies a 3x3 Conv-BN-ReLU block. Keeping it here lets the multimodal
    fusion live inside the encoder, while OpenCOOD's cooperative fusion and
    prediction heads stay unchanged.
    """

    def __init__(self, in_channels, out_channels):
        if isinstance(in_channels, int):
            in_channels = [in_channels]
        self.in_channels = [int(channel) for channel in in_channels]
        self.out_channels = int(out_channels)
        super().__init__(
            nn.Conv2d(
                sum(self.in_channels),
                self.out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return super().forward(torch.cat(inputs, dim=1))


def _find_encoder_class(core_method):
    target_model_name = core_method.replace("_", "").lower()
    for module_name in (
        "opencood.models.heter_encoders_dsvt",
        "opencood.models.heter_encoders",
    ):
        encoder_lib = importlib.import_module(module_name)
        for name, cls in encoder_lib.__dict__.items():
            if name.lower() == target_model_name:
                return cls
    raise RuntimeError(f"Unknown BEVFusion branch encoder {core_method}")


class BEVFusion(nn.Module):
    """Local LiDAR-camera BEVFusion encoder for OpenCOOD.

    This mirrors the BEVFusion recipe at the encoder boundary:
    LiDAR branch -> BEV feature, camera branch -> BEV feature, ConvFuser ->
    fused BEV feature. The returned tensor follows the existing heterogeneous
    encoder convention and can feed the original OpenCOOD BEV backbone,
    cooperative fusion, and detection head.
    """

    def __init__(self, args):
        super().__init__()
        lidar_cfg = copy.deepcopy(args["lidar_encoder"])
        camera_cfg = copy.deepcopy(args["camera_encoder"])
        fuser_cfg = copy.deepcopy(args.get("fuser", {}))

        self.lidar_input_key = lidar_cfg.get("input_key", "lidar")
        self.camera_input_key = camera_cfg.get("input_key", "camera")

        lidar_cls = _find_encoder_class(lidar_cfg["core_method"])
        camera_cls = _find_encoder_class(camera_cfg["core_method"])
        self.lidar_encoder = lidar_cls(lidar_cfg["encoder_args"])
        self.camera_encoder = camera_cls(camera_cfg["encoder_args"])

        self.depth_supervision = bool(
            camera_cfg["encoder_args"].get("depth_supervision", False)
        )
        self.depth_items = None

        in_channels = fuser_cfg.get(
            "in_channels",
            [lidar_cfg.get("out_channels"), camera_cfg.get("out_channels")],
        )
        if any(channel is None for channel in in_channels):
            raise ValueError(
                "BEVFusion fuser needs in_channels, or branch out_channels."
            )
        out_channels = fuser_cfg.get("out_channels", in_channels[0])
        self.fuser = ConvFuser(in_channels, out_channels)

        if lidar_cfg.get("freeze", False):
            self._freeze_module(self.lidar_encoder)
        if camera_cfg.get("freeze", False):
            self._freeze_module(self.camera_encoder)
        if fuser_cfg.get("freeze", False):
            self._freeze_module(self.fuser)

    @staticmethod
    def _freeze_module(module):
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _match_hw(feature, target_hw):
        target_h, target_w = target_hw
        _, _, height, width = feature.shape

        if height > target_h:
            top = (height - target_h) // 2
            feature = feature[:, :, top:top + target_h, :]
            height = target_h
        if width > target_w:
            left = (width - target_w) // 2
            feature = feature[:, :, :, left:left + target_w]
            width = target_w

        pad_h = max(target_h - height, 0)
        pad_w = max(target_w - width, 0)
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            feature = F.pad(feature, (pad_left, pad_right, pad_top, pad_bottom))
        return feature

    def _branch_data_dict(self, data_dict, modality_name, branch_key):
        inputs = data_dict[f"inputs_{modality_name}"]
        if branch_key not in inputs:
            raise KeyError(
                f"BEVFusion expected inputs_{modality_name}['{branch_key}']."
            )
        branch_data = dict(data_dict)
        branch_data[f"inputs_{modality_name}"] = inputs[branch_key]
        return branch_data

    def forward(self, data_dict, modality_name):
        lidar_data = self._branch_data_dict(
            data_dict, modality_name, self.lidar_input_key
        )
        camera_data = self._branch_data_dict(
            data_dict, modality_name, self.camera_input_key
        )

        lidar_feature = self.lidar_encoder(lidar_data, modality_name)
        camera_feature = self.camera_encoder(camera_data, modality_name)

        if self.depth_supervision and hasattr(self.camera_encoder, "depth_items"):
            self.depth_items = self.camera_encoder.depth_items

        camera_feature = self._match_hw(camera_feature, lidar_feature.shape[-2:])
        return self.fuser([lidar_feature, camera_feature])



class PointPillar(nn.Module):
    def __init__(self, args):
        super(PointPillar, self).__init__()
        grid_size = (np.array(args['lidar_range'][3:6]) - np.array(args['lidar_range'][0:3])) / \
                            np.array(args['voxel_size'])
        grid_size = np.round(grid_size).astype(np.int64)
        args['point_pillar_scatter']['grid_size'] = grid_size

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])


    def forward(self, data_dict, modality_name):
        voxel_features = data_dict[f'inputs_{modality_name}']['voxel_features']
        voxel_coords = data_dict[f'inputs_{modality_name}']['voxel_coords']
        voxel_num_points = data_dict[f'inputs_{modality_name}']['voxel_num_points']
    
        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        lidar_feature_2d = batch_dict['spatial_features'] # H0, W0
        return lidar_feature_2d

class SECOND(nn.Module):
    def __init__(self, args):
        super(SECOND, self).__init__()
        lidar_range = np.array(args['lidar_range'])
        grid_size = np.round((lidar_range[3:6] - lidar_range[:3]) /
                                np.array(args['voxel_size'])).astype(np.int64)
        self.vfe = MeanVFE(args['mean_vfe'],
                            args['mean_vfe']['num_point_features'])
        self.spconv_block = VoxelBackBone8x(args['spconv'],
                                            input_channels=args['spconv'][
                                                'num_features_in'],
                                            grid_size=grid_size)
        self.map_to_bev = HeightCompression(args['map2bev'])

    def forward(self, data_dict, modality_name):
        voxel_features = data_dict[f'inputs_{modality_name}']['voxel_features']
        voxel_coords = data_dict[f'inputs_{modality_name}']['voxel_coords']
        voxel_num_points = data_dict[f'inputs_{modality_name}']['voxel_num_points']
        batch_size = voxel_coords[:,0].max() + 1


        batch_dict = {'voxel_features': voxel_features,
                    'voxel_coords': voxel_coords,
                    'voxel_num_points': voxel_num_points,
                    'batch_size': batch_size}

        batch_dict = self.vfe(batch_dict)
        batch_dict = self.spconv_block(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)
        return batch_dict['spatial_features']

class LiftSplatShoot(nn.Module):
    def __init__(self, args): 
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = args['grid_conf']   # 网格配置参数
        self.data_aug_conf = args['data_aug_conf']   # 数据增强配置参数
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                self.grid_conf['ybound'],
                                self.grid_conf['zbound'],
                                )  # 划分网格

        self.dx = dx.clone().detach().requires_grad_(False).to(torch.device("cuda"))  # [0.4,0.4,20]
        self.bx = bx.clone().detach().requires_grad_(False).to(torch.device("cuda"))  # [-49.8,-49.8,0]
        self.nx = nx.clone().detach().requires_grad_(False).to(torch.device("cuda"))  # [250,250,1]
        self.depth_supervision = args['depth_supervision']
        self.downsample = args['img_downsample']  # 下采样倍数
        self.camC = args['img_features']  # 图像特征维度
        self.frustum = self.create_frustum().clone().detach().requires_grad_(False).to(torch.device("cuda"))  # frustum: DxfHxfWx3(41x8x16x3)
        self.use_quickcumsum = True
        self.D, _, _, _ = self.frustum.shape  # D: 41
        self.camera_encoder_type = args['camera_encoder']
        if self.camera_encoder_type == 'EfficientNet':
            self.camencode = CamEncode(self.D, self.camC, self.downsample, \
                self.grid_conf['ddiscr'], self.grid_conf['mode'], args['use_depth_gt'], args['depth_supervision'])
        elif self.camera_encoder_type == 'Resnet101':
            self.camencode = CamEncode_Resnet101(self.D, self.camC, self.downsample, \
                self.grid_conf['ddiscr'], self.grid_conf['mode'], args['use_depth_gt'], args['depth_supervision'])
    
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']  # 原始图片大小  ogfH:128  ogfW:288
        fH, fW = ogfH // self.downsample, ogfW // self.downsample  # 下采样16倍后图像大小  fH: 12  fW: 22
        # ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)  # 在深度方向上划分网格 ds: DxfHxfW(41x12x22)
        ds = torch.tensor(depth_discretization(*self.grid_conf['ddiscr'], self.grid_conf['mode']), dtype=torch.float).view(-1,1,1).expand(-1, fH, fW)

        D, _, _ = ds.shape # D: 41 表示深度方向上网格的数量
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)  # 在0到288上划分18个格子 xs: DxfHxfW(41x12x22)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)  # 在0到127上划分8个格子 ys: DxfHxfW(41x12x22)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)  # 堆积起来形成网格坐标, frustum[i,j,k,0]就是(i,j)位置，深度为k的像素的宽度方向上的栅格坐标   frustum: DxfHxfWx3
        return frustum

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape  # B:4(batchsize)    N: 4(相机数目)

        # undo post-transformation
        # B x N x D x H x W x 3
        # 抵消数据增强及预处理对像素的变化
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],  # points[:, :, :, :, :, 2:3] ranges from [4, 45) meters
                            points[:, :, :, :, :, 2:3]
                            ), 5)  # 将像素坐标(u,v,d)变成齐次坐标(du,dv,d)
        # d[u,v,1]^T=intrins*rots^(-1)*([x,y,z]^T-trans)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)  # 将像素坐标d[u,v,1]^T转换到车体坐标系下的[x,y,z]
        
        return points  # B x N x D x H x W x 3 (4 x 4 x 41 x 16 x 22 x 3) 

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape  # B: 4  N: 4  C: 3  imH: 256  imW: 352

        x = x.view(B*N, C, imH, imW)  # B和N两个维度合起来  x: 16 x 4 x 256 x 352
        depth_items, x = self.camencode(x) # 进行图像编码  x: B*N x C x D x fH x fW(24 x 64 x 41 x 16 x 22)
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)  #将前两维拆开 x: B x N x C x D x fH x fW(4 x 6 x 64 x 41 x 16 x 22)
        x = x.permute(0, 1, 3, 4, 5, 2)  # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64)

        return x, depth_items

    def voxel_pooling(self, geom_feats, x):
        # geom_feats: B x N x D x H x W x 3 (4 x 6 x 41 x 16 x 22 x 3), D is discretization in "UD" or "LID"
        # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64), D is num_bins

        B, N, D, H, W, C = x.shape  # B: 4  N: 6  D: 41  H: 16  W: 22  C: 64
        Nprime = B*N*D*H*W  # Nprime

        # flatten x
        x = x.reshape(Nprime, C)  # 将图像展平，一共有 B*N*D*H*W 个点

        # flatten indices

        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()  # 将[-48,48] [-10 10]的范围平移到 [0, 240), [0, 1) 计算栅格坐标并取整
        geom_feats = geom_feats.view(Nprime, 3)  # 将像素映射关系同样展平  geom_feats: B*N*D*H*W x 3 
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])  # 每个点对应于哪个batch
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # geom_feats: B*N*D*H*W x 4, geom_feats[:,3]表示batch_id

        # filter out points that are outside box
        # 过滤掉在边界线之外的点 x:0~240  y: 0~240  z: 0
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept] 
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]  # 给每一个点一个rank值，rank相等的点在同一个batch，并且在在同一个格子里面
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]  # 按照rank排序，这样rank相近的点就在一起了
        # x: 168648 x 64  geom_feats: 168648 x 4  ranks: 168648

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)  # 一个batch的一个格子里只留一个点 x: 29072 x 64  geom_feats: 29072 x 4

        # griddify (B x C x Z x X x Y)
        # final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)  # final: 4 x 64 x Z x X x Y
        # final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x  # 将x按照栅格坐标放到final中

        # modify griddify (B x C x Z x Y x X) by Yifan Lu 2022.10.7
        # ------> x
        # |
        # |
        # y
        final = torch.zeros((B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device)  # final: 4 x 64 x Z x Y x X
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x  # 将x按照栅格坐标放到final中

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)  # 消除掉z维

        return final  # final: 4 x 64 x 240 x 240  # B, C, H, W

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)  # 像素坐标到自车中坐标的映射关系 geom: B x N x D x H x W x 3 (4 x N x 42 x 16 x 22 x 3)
        x_img, depth_items = self.get_cam_feats(x)  # 提取图像特征并预测深度编码 x: B x N x D x fH x fW x C(4 x N x 42 x 16 x 22 x 64)
        x = self.voxel_pooling(geom, x_img)  # x: 4 x 64 x 240 x 240

        return x, depth_items

    def forward(self, data_dict, modality_name):
        # x: [4,4,3,256, 352]
        # rots: [4,4,3,3]
        # trans: [4,4,3]
        # intrins: [4,4,3,3]
        # post_rots: [4,4,3,3]
        # post_trans: [4,4,3]
        image_inputs_dict = data_dict[f'inputs_{modality_name}']
        x, rots, trans, intrins, post_rots, post_trans = \
            image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'], image_inputs_dict['post_rots'], image_inputs_dict['post_trans']
        x, depth_items = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)  # 将图像转换到BEV下，x: B x C x 240 x 240 (4 x 64 x 240 x 240)
        
        if self.depth_supervision:
            self.depth_items = depth_items

        return x


class LiftSplatShootVoxel(LiftSplatShoot):
    def voxel_pooling(self, geom_feats, x):
        # geom_feats: B x N x D x H x W x 3 (4 x 6 x 41 x 16 x 22 x 3), D is discretization in "UD" or "LID"
        # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64), D is num_bins

        B, N, D, H, W, C = x.shape  # B: 4  N: 6  D: 41  H: 16  W: 22  C: 64
        Nprime = B*N*D*H*W  # Nprime

        # flatten x
        x = x.reshape(Nprime, C)  # 将图像展平，一共有 B*N*D*H*W 个点

        # flatten indices

        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()  # 将[-48,48] [-10 10]的范围平移到 [0, 240), [0, 1) 计算栅格坐标并取整
        geom_feats = geom_feats.view(Nprime, 3)  # 将像素映射关系同样展平  geom_feats: B*N*D*H*W x 3 
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])  # 每个点对应于哪个batch
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # geom_feats: B*N*D*H*W x 4, geom_feats[:,3]表示batch_id

        # filter out points that are outside box
        # 过滤掉在边界线之外的点 x:0~240  y: 0~240  z: 0
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept] 
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]  # 给每一个点一个rank值，rank相等的点在同一个batch，并且在在同一个格子里面
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]  # 按照rank排序，这样rank相近的点就在一起了
        # x: 168648 x 64  geom_feats: 168648 x 4  ranks: 168648

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)  # 一个batch的一个格子里只留一个点 x: 29072 x 64  geom_feats: 29072 x 4

        # griddify (B x C x Z x X x Y)
        # final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)  # final: 4 x 64 x Z x X x Y
        # final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x  # 将x按照栅格坐标放到final中

        # modify griddify (B x C x Z x Y x X) by Yifan Lu 2022.10.7
        # ------> x
        # |
        # |
        # y
        final = torch.zeros((B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device)  # final: 4 x 64 x Z x Y x X
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x  # 将x按照栅格坐标放到final中

        # collapse Z
        #final = torch.max(final.unbind(dim=2), 1)[0]  # 消除掉z维
        final = torch.max(final, 2)[0]  # 消除掉z维
        return final  # final: 4 x 64 x 240 x 240  # B, C, H, W 
