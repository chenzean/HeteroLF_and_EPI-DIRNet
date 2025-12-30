# -*- coding: utf-8 -*-
# @Time    : 2025-10-22 17:55
# @Author  : Chen Zean
# @Site    : 
# @File    : EPIT.py
# @Software: PyCharm
'''
@Article{EPIT,
    author    = {Liang, Zhengyu and Wang, Yingqian and Wang, Longguang and Yang, Jungang and Zhou Shilin and Guo, Yulan},
    title     = {Learning Non-Local Spatial-Angular Correlation for Light Field Image Super-Resolution},
    journal   = {arXiv preprint arXiv:2302.08058},
    year      = {2023},
}
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        channels = 64
        self.angRes = args.angRes_in
        self.scale = args.scale_factor

        #################### Initial Feature Extraction #####################
        self.conv_init0 = nn.Sequential(nn.Conv3d(1, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False))
        self.conv_init = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        ############# Deep Spatial-Angular Correlation Learning #############
        self.altblock = nn.Sequential(
            AltFilter(self.angRes, channels),
            AltFilter(self.angRes, channels),
            AltFilter(self.angRes, channels),
            AltFilter(self.angRes, channels),
            AltFilter(self.angRes, channels),
        )

        ########################### UP-Sampling #############################
        self.upsampling = nn.Sequential(
            nn.Conv2d(channels, channels * self.scale ** 2, kernel_size=1, padding=0, bias=False),
            nn.PixelShuffle(self.scale),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, lr, info=None):
        lr = rearrange(lr, 'b c (u h) (v w) -> b c u v h w', u=self.angRes, v=self.angRes)
        [b, c, u, v, h, w] = lr.size()

        sr_y = LF_interpolate(lr, scale_factor=self.scale, mode='bicubic')
        sr_y = rearrange(sr_y, 'b c u v h w -> b c (u h) (v w)', u=u, v=v)

        # Initial Feature Extraction
        x = rearrange(lr, 'b c u v h w -> b c (u v) h w')
        buffer = self.conv_init0(x)
        buffer = self.conv_init(buffer) + buffer

        # Deep Spatial-Angular Correlation Learning
        buffer = self.altblock(buffer) + buffer

        # UP-Sampling
        buffer = rearrange(buffer, 'b c (u v) h w -> b c (u h) (v w)', u=u, v=v)
        y = self.upsampling(buffer) + sr_y
        # y = rearrange(y, 'b c (u h) (v w) -> b c u v h w', u=u, v=v)

        return y


class BasicTrans(nn.Module):
    def __init__(self, channels, spa_dim, num_heads=8, dropout=0.):
        super(BasicTrans, self).__init__()
        self.linear_in = nn.Linear(channels, spa_dim, bias=False)
        self.norm = nn.LayerNorm(spa_dim)
        self.attention = nn.MultiheadAttention(spa_dim, num_heads, dropout, bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(spa_dim),
            nn.Linear(spa_dim, spa_dim*2, bias=False),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(spa_dim*2, spa_dim, bias=False),
            nn.Dropout(dropout)
        )
        self.linear_out = nn.Linear(spa_dim, channels, bias=False)

    def gen_mask(self, h: int, w: int, k_h: int, k_w: int):
        attn_mask = torch.zeros([h, w, h, w])
        k_h_left = k_h // 2
        k_h_right = k_h - k_h_left
        k_w_left = k_w // 2
        k_w_right = k_w - k_w_left
        for i in range(h):
            for j in range(w):
                temp = torch.zeros(h, w)
                temp[max(0, i - k_h_left):min(h, i + k_h_right), max(0, j - k_w_left):min(w, j + k_w_right)] = 1
                attn_mask[i, j, :, :] = temp

        attn_mask = rearrange(attn_mask, 'a b c d -> (a b) (c d)')
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))

        return attn_mask

    def forward(self, buffer):
        [_, _, n, v, w] = buffer.size()
        attn_mask = self.gen_mask(v, w, self.mask_field[0], self.mask_field[1]).to(buffer.device)

        epi_token = rearrange(buffer, 'b c n v w -> (v w) (b n) c')
        epi_token = self.linear_in(epi_token)

        epi_token_norm = self.norm(epi_token)
        epi_token = self.attention(query=epi_token_norm,
                                   key=epi_token_norm,
                                   value=epi_token,
                                   attn_mask=attn_mask,
                                   need_weights=False)[0] + epi_token

        epi_token = self.feed_forward(epi_token) + epi_token
        epi_token = self.linear_out(epi_token)
        buffer = rearrange(epi_token, '(v w) (b n) c -> b c n v w', v=v, w=w, n=n)

        return buffer


class AltFilter(nn.Module):
    def __init__(self, angRes, channels):
        super(AltFilter, self).__init__()
        self.angRes = angRes
        self.epi_trans = BasicTrans(channels, channels*2)
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        )

    def forward(self, buffer):
        buffer = rearrange(buffer, 'b c u v h w -> b c (u v) h w ', u=self.angRes, v=self.angRes)
        shortcut = buffer
        [_, _, _, h, w] = buffer.size()
        self.epi_trans.mask_field = [self.angRes * 2, 11]

        # Horizontal
        buffer = rearrange(buffer, 'b c (u v) h w -> b c (v w) u h', u=self.angRes, v=self.angRes)
        buffer = self.epi_trans(buffer)
        buffer = rearrange(buffer, 'b c (v w) u h -> b c (u v) h w', u=self.angRes, v=self.angRes, h=h, w=w)
        buffer = self.conv(buffer) + shortcut

        # Vertical
        buffer = rearrange(buffer, 'b c (u v) h w -> b c (u h) v w', u=self.angRes, v=self.angRes)
        buffer = self.epi_trans(buffer)
        buffer = rearrange(buffer, 'b c (u h) v w -> b c (u v) h w', u=self.angRes, v=self.angRes, h=h, w=w)
        buffer = self.conv(buffer) + shortcut

        buffer = rearrange(buffer, 'b c (u v) h w -> b c u v h w', u=self.angRes, v=self.angRes)

        return buffer


def LF_interpolate(LF, scale_factor, mode):
    [b, c, u, v, h, w] = LF.size()
    LF = rearrange(LF, 'b c u v h w -> (b u v) c h w')
    LF_upscale = F.interpolate(LF, scale_factor=scale_factor, mode=mode, align_corners=False)
    LF_upscale = rearrange(LF_upscale, '(b u v) c h w -> b c u v h w', u=u, v=v)
    return LF_upscale


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, out, HR, degrade_info=None):
        loss = self.criterion_Loss(out['SR'], HR)

        return loss


def weights_init(m):
    pass


import torch
import matplotlib.pyplot as plt
import seaborn as sns
from einops import rearrange


def visualize_mask(h=8, w=8, k_h=3, k_w=3):
    # --- 1. 复用你原本的 gen_mask 逻辑 (稍作修改以便可视化) ---
    attn_mask = torch.zeros([h, w, h, w])
    k_h_left = k_h // 2
    k_h_right = k_h - k_h_left
    k_w_left = k_w // 2
    k_w_right = k_w - k_w_left

    for i in range(h):
        for j in range(w):
            temp = torch.zeros(h, w)
            # 设定局部窗口为 1
            temp[max(0, i - k_h_left):min(h, i + k_h_right),
            max(0, j - k_w_left):min(w, j + k_w_right)] = 1
            attn_mask[i, j, :, :] = temp

    # 展平: [H, W, H, W] -> [H*W, H*W]
    attn_mask = rearrange(attn_mask, 'a b c d -> (a b) (c d)')

    # 注意：这里我们不转成 -inf，而是保留 0 和 1，方便绘图
    # 1 (黄色/亮色) = 可见 (Local Window)
    # 0 (紫色/暗色) = 屏蔽 (Masked)

    # --- 2. 绘图 ---
    plt.figure(figsize=(10, 8))

    # 使用 Seaborn 绘制热力图
    sns.heatmap(attn_mask.numpy(), cmap="viridis", cbar=False, square=True)

    plt.title(f"Local Attention Mask\nImage Size: {h}x{w}, Window: {k_h}x{k_w}\n(Yellow=Visible, Purple=Masked)")
    plt.xlabel(f"Key Pixel Index (Total {h * w})")
    plt.ylabel(f"Query Pixel Index (Total {h * w})")

    # 标注一下具体的某一行，帮助理解
    # 选取中间的一个像素点 (例如第 4 行第 4 列的像素，展平后索引是 4*8+4 = 36)
    target_idx = (h // 2) * w + (w // 2)
    plt.axhline(y=target_idx, color='red', linestyle='--', alpha=0.5)
    plt.text(w * h + 1, target_idx, " <-- One Query Pixel", color='red', va='center')

    plt.show()


# 运行可视化
if __name__ == "__main__":
    visualize_mask(h=10, w=10, k_h=3, k_w=3)