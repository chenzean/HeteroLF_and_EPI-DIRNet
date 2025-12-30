# -*- coding: utf-8 -*-
# 修正版 SAV_parallel 模块（兼容 PyTorch 初始化）
# Author: Chen Zean (Revised by ChatGPT)
# Based on: Cheng et al., TCI 2022 (SAVNet)

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 1️⃣ SAS_conv：Spatial-Angular-Separable Convolution
# =========================
class SAS_conv(nn.Module):
    def __init__(self, fn=64, act='leaky_relu'):
        super(SAS_conv, self).__init__()

        # -------- 激活函数映射 + 初始化策略 --------
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
            self.init_indicator = 'relu'
            a = 0
        elif act in ['lrelu', 'leaky_relu']:
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.init_indicator = 'leaky_relu'
            a = 0.2
        else:
            raise ValueError(f"Unsupported activation type: {act}")

        # -------- Spatial + Angular 卷积 --------
        self.spaconv = nn.Conv2d(fn, fn, 3, 1, 1, bias=False)

        self.angconv = nn.Conv2d(fn, fn, 3, 1, 1, bias=False)


    def forward(self, x):
        """
        输入: x [N, C, U, V, H, W]
        输出: x [N, C, U, V, H, W]
        """
        N, C, U, V, H, W = x.shape

        # Spatial 维卷积：在 (H, W) 平面上操作
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(N * U * V, C, H, W)
        out = self.act(self.spaconv(x))  # [N*U*V, C, H, W]

        # Angular 维卷积：在 (U, V) 平面上操作
        out = out.view(N, U * V, C, H * W).transpose(1, 3).contiguous()
        out = out.view(N * H * W, C, U, V)
        out = self.act(self.angconv(out))  # [N*H*W, C, U, V]

        # 恢复维度
        out = out.view(N, H * W, C, U * V).transpose(1, 3).contiguous()
        out = out.view(N, U, V, C, H, W).permute(0, 3, 1, 2, 4, 5).contiguous()
        return out


# =========================
# 2️⃣ SAC_conv：Spatial-Angular-Coupled Convolution
# =========================
class SAC_conv(nn.Module):
    def __init__(self, act='leaky_relu', symmetry=True, max_k_size=3, fn=64):
        super(SAC_conv, self).__init__()

        # -------- 激活函数映射 + 初始化 --------
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
            self.init_indicator = 'relu'
            a = 0
        elif act in ['lrelu', 'leaky_relu']:
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.init_indicator = 'leaky_relu'
            a = 0.2
        else:
            raise ValueError(f"Unsupported activation type: {act}")

        # -------- 卷积核尺寸设置 --------
        k_ang = max_k_size
        k_spa = max_k_size if symmetry else max_k_size - 2

        # -------- 竖向 / 水平卷积 --------
        self.verconv = nn.Conv2d(fn, fn, (k_ang, k_spa), 1,
                                 padding=(k_ang // 2, k_spa // 2), bias=False)


        self.horconv = nn.Conv2d(fn, fn, (k_ang, k_spa), 1,
                                 padding=(k_ang // 2, k_spa // 2), bias=False)


    def forward(self, x):
        """
        输入: x [N, C, U, V, H, W]
        输出: x [N, C, U, V, H, W]
        """
        N, C, U, V, H, W = x.shape

        # 在 Angular 维和 Spatial 维交替卷积
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous().view(N * V * W, C, U, H)
        out = self.act(self.verconv(x))  # [N*V*W, C, U, H]

        out = out.view(N, V * W, C, U * H).transpose(1, 3).contiguous()
        out = out.view(N * U * H, C, V, W)
        out = self.act(self.horconv(out))  # [N*U*H, C, V, W]

        out = out.view(N, U * H, C, V * W).transpose(1, 3).contiguous()
        out = out.view(N, V, W, C, U, H).permute(0, 3, 4, 1, 5, 2).contiguous()
        return out


# =========================
# 3️⃣ SAV_parallel：并行融合模块
# =========================
class SAV_parallel(nn.Module):
    def __init__(self, fn=64, act='leaky_relu', feature_concat=True):
        super(SAV_parallel, self).__init__()
        self.feature_concat = feature_concat

        # 子模块
        self.SAS_conv = SAS_conv(act=act, fn=fn)
        self.SAC_conv = SAC_conv(act=act, fn=fn)

        self.act = nn.LeakyReLU(0.1, inplace=True)

        if self.feature_concat:
            # 融合两个特征并通道压缩
            self.channel_reduce = nn.Conv3d(
                in_channels=2 * fn,
                out_channels=fn,
                kernel_size=1,
                stride=1,
                padding=0,
                bias = False
            )

    def forward(self, lf_input):
        """
        Args:
            lf_input: [N, C, U, V, H, W]
        Returns:
            res:      [N, C, U, V, H, W]
        """
        N, C, U, V, H, W = lf_input.shape

        sas_feat = self.SAS_conv(lf_input)   # [N, C, U, V, H, W]
        sac_feat = self.SAC_conv(lf_input)   # [N, C, U, V, H, W]

        if self.feature_concat:
            # 拼接通道 -> [N, 2C, U, V, H, W]
            concat_feat = torch.cat((sas_feat, sac_feat), dim=1)

            # 简单激活
            concat_feat = self.act(concat_feat)

            # 将 (U,V) 合并为 3D 的 D 维： [N, 2C, U*V, H, W]
            feat5d = concat_feat.view(N, 2 * C, U * V, H, W)

            # 3D 1x1x1 压缩通道： [N, C, U*V, H, W]
            res5d = self.channel_reduce(feat5d)

            # 残差也展平到 5D 再相加
            skip5d = lf_input.view(N, C, U * V, H, W)
            res5d = res5d + skip5d

            # 还原回 6D： [N, C, U, V, H, W]
            res6d = res5d.view(N, C, U, V, H, W)
        else:
            # 直接在 6D 上做残差
            res6d = sas_feat + sac_feat + lf_input

        return res6d
