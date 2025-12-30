# -*- coding: utf-8 -*-
# @Time    : 2025-12-12 17:57
# @Author  : Chen Zean
# @Site    :
# @File    : Model_251212.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
# @Time    : 2025-12-12 15:53
# @Author  : Chen Zean
# @Site    :
# @File    : New_model_1212.py
# @Software: PyCharm

import math
import numbers
import time
from typing import Dict, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange
from lpips import LPIPS

# Ensure these modules exist in your project structure
from loss import ReconstructionLoss, DetailLoss
from model.LF_INR import INR as LF_INR
from model.LFT import AltFilter as LFT_AltFilter
from model.LFT import PositionEncoding
from model.EPIT import AltFilter as EPIT_AltFilter
from model.SAV import SAV_parallel, SAS_conv


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # Supports [B, C, H, W] input
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, channels, activation='leakyrelu', bias=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=bias)

        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return out + residual


class ResASPP(nn.Module):
    def __init__(self, channel, bias=False):
        super(ResASPP, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(channel,channel, kernel_size=3, stride=1, padding=1, dilation=1, bias=bias),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    )
        self.conv_2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=2, dilation=2, bias=bias),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    )
        self.conv_3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=4, dilation=4, bias=bias),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    )
        self.conv_t = nn.Conv2d(channel*3, channel, kernel_size=1, stride=1, padding=0, bias=bias)

    def __call__(self, x):
        buffer_1 = []
        buffer_1.append(self.conv_1(x))
        buffer_1.append(self.conv_2(x))
        buffer_1.append(self.conv_3(x))
        buffer_1 = self.conv_t(torch.cat(buffer_1, 1))
        return x + buffer_1


class M2MTEPIAttention(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int,
                 angular_size=(7, 7), spatial_size=(32, 32),
                 heads: int = 4, mode: str = 'h', bias=False) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.mode = mode

        U, V = angular_size
        H, W = spatial_size

        if mode == 'h':
            self.seq_len = U * H
            in_channels_ = in_channels * V * W
            self.dims = {'u': U, 'v': V, 'h': H, 'w': W, 'c': in_channels}
        elif mode == 'v':
            self.seq_len = V * W
            in_channels_ = in_channels * U * H
            self.dims = {'u': U, 'v': V, 'h': H, 'w': W, 'c': in_channels}
        else:
            raise ValueError("Mode must be 'h' or 'v'")

        self.norm = nn.LayerNorm(embed_dim)
        self.ff_in = nn.Linear(in_channels_, embed_dim, bias=bias)
        self.q_layer = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_layer = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_layer = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ff_out = nn.Linear(embed_dim, in_channels_, bias=bias)
        self.scale = (embed_dim // heads) ** -0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inp: torch.Tensor):
        x = inp
        if self.mode == 'h':
            x = rearrange(x, 'b c u v h w -> b (u h) (c v w)')
        else:
            x = rearrange(x, 'b c u v h w -> b (v w) (c u h)')

        x = self.ff_in(x)  # [B, Seq, embed_dim]
        x_norm = self.norm(x)

        q = self.q_layer(x_norm)
        k = self.k_layer(x_norm)
        v = self.v_layer(x)

        q = rearrange(q, 'b seq (head d) -> b head seq d', head=self.heads)
        k = rearrange(k, 'b seq (head d) -> b head seq d', head=self.heads)
        v = rearrange(v, 'b seq (head d) -> b head seq d', head=self.heads)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        x = attn @ v

        x = rearrange(x, 'b head seq d -> b seq (head d)')
        x = self.ff_out(x)

        if self.mode == 'h':
            x = rearrange(x, 'b (u h) (c v w) -> b c u v h w',
                          u=self.dims['u'], h=self.dims['h'],
                          v=self.dims['v'], w=self.dims['w'], c=self.dims['c'])
        else:
            x = rearrange(x, 'b (v w) (c u h) -> b c u v h w',
                          u=self.dims['u'], h=self.dims['h'],
                          v=self.dims['v'], w=self.dims['w'], c=self.dims['c'])
        return x



class EPI_DWConv(nn.Module):
    """
    Asymmetric Depth-wise Convolution for EPI features.
    """

    def __init__(self, dim, kernel_size=7, epi_type='H', bias=False):
        super().__init__()
        assert epi_type in ['H', 'V']
        self.epi_type = epi_type
        self.act = nn.LeakyReLU(0.1, inplace=True)

        self.conv3x3 = nn.Conv2d(dim, dim, 3, padding=(1, 1), groups=dim, bias=bias)
        self.conv3x5 = nn.Conv2d(dim, dim, (3, 5), padding=(1, 2), groups=dim, bias=bias)
        self.conv3x7 = nn.Conv2d(dim, dim, (3, 7), padding=(1, 3), groups=dim, bias=bias)
        self.fuse = nn.Conv2d(dim * 3, dim, 1, bias=bias)

    def forward(self, x):
        # Input: [B, U, V, C, H, W]
        B, U, V, C, H, W = x.shape
        if self.epi_type == 'H':
            # EPI mode 1
            x_epi = rearrange(x, 'b u v c h w -> (b u h) c v w')
            out1 = self.act(self.conv3x3(x_epi))
            out2 = self.act(self.conv3x5(x_epi))
            out3 = self.act(self.conv3x7(x_epi))
            out = torch.cat([out1, out2, out3], dim=1)
            out = self.fuse(out)
            out = rearrange(out, '(b u h) c v w -> b u v c h w', b=B, u=U, h=H)
        else:
            # EPI mode 2
            x_epi = rearrange(x, 'b u v c h w -> (b v w) c u h')
            out1 = self.act(self.conv3x3(x_epi))
            out2 = self.act(self.conv3x5(x_epi))
            out3 = self.act(self.conv3x7(x_epi))
            out = torch.cat([out1, out2, out3], dim=1)
            out = self.fuse(out)
            out = rearrange(out, '(b v w) c u h -> b u v c h w', b=B, v=V, w=W)
        return out

# =============================================================================
# EPI-Conv Transformer Branch
# =============================================================================
class DirectionalEPIBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, kernel_size=7, bias=False):
        super().__init__()
        dw_channel = c * DW_Expand
        ffn_channel = c * FFN_Expand

        self.conv1 = nn.Conv2d(c, dw_channel, 1, bias=bias)
        self.attn_h = EPI_DWConv(dw_channel, kernel_size, epi_type='H')
        self.attn_v = EPI_DWConv(dw_channel, kernel_size, epi_type='V')

        self.sg = SimpleGate()
        self.sca = ChannelAttention(dw_channel // 2)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, bias=bias)

        # FFN
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, bias=bias)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, bias=bias)

        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        # Input: [B, C, U, V, H, W]
        B, C, U, V, H, W = x.shape

        x_epi = rearrange(x, 'b c u v h w -> (b u v) c h w')

        # Norm 1
        x_norm = x_epi.permute(0, 2, 3, 1)
        x_norm = self.norm1(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)

        # Expansion
        x_expand = self.conv1(x_norm)
        x_expand_epi = rearrange(x_expand, '(b u v) c h w -> b u v c h w', b=B, u=U, v=V)

        # EPI Conv
        attn_h = self.attn_h(x_expand_epi)
        attn_v = self.attn_v(x_expand_epi)
        attn = attn_h + attn_v
        attn = rearrange(attn, 'b u v c h w -> (b u v) c h w', b=B, u=U, v=V)

        # Gate + Channel Attention
        out = self.sg(attn)
        out = self.sca(out)
        out = self.conv3(out)

        # Residual 1
        y = x_epi + out * self.beta

        # FFN & Residual 2
        x_ffn = y.permute(0, 2, 3, 1)
        x_ffn = self.norm2(x_ffn)
        x_ffn = x_ffn.permute(0, 3, 1, 2)

        x_ffn = self.conv4(x_ffn)
        x_ffn = self.sg(x_ffn)
        x_ffn = self.conv5(x_ffn)

        out = y + x_ffn * self.gamma
        out = rearrange(out, '(b u v) c h w -> b c u v h w', b=B, u=U, v=V)
        return out


# =============================================================================
# Intra-Intra EPI Transformer
# =============================================================================


class EPIFG(nn.Module):
    """EPI Fusion Gate"""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(dim * 2, dim // 4, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 4, dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        g = self.gate(torch.cat([x1, x2], dim=1))
        return x1 * g + x2 * (1 - g)


class BasicTrans(nn.Module):
    def __init__(self, channels, spa_dim, num_heads=8, dropout=0., bias=False):
        super(BasicTrans, self).__init__()
        self.linear_in = nn.Linear(channels, spa_dim, bias=bias)
        self.norm = nn.LayerNorm(spa_dim)
        self.attention = nn.MultiheadAttention(spa_dim, num_heads, dropout, bias=bias)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(spa_dim),
            nn.Linear(spa_dim, spa_dim * 2, bias=bias),
            nn.LeakyReLU(0.1, inplace=bias),
            nn.Linear(spa_dim * 2, spa_dim, bias=bias),
        )
        self.linear_out = nn.Linear(spa_dim, channels, bias=bias)
        self.mask_field = [14, 11]  # Default placeholder

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

        epi_token = self.attention(
            query=epi_token_norm,
            key=epi_token_norm,
            value=epi_token,
            attn_mask=attn_mask,
            need_weights=False
        )[0] + epi_token

        epi_token = self.feed_forward(epi_token) + epi_token
        epi_token = self.linear_out(epi_token)
        buffer = rearrange(epi_token, '(v w) (b n) c -> b c n v w', v=v, w=w, n=n)
        return buffer


class Intra_SA_LF_Parallel(nn.Module):
    def __init__(self, args, dim, bias=False, fusion='EPIFG', is_weighted_shared=False):
        super().__init__()
        self.args = args
        self.angRes = args.angRes
        self.fusion = fusion

        if is_weighted_shared == False:                     # Weight not shared
            self.h = BasicTrans(dim, dim * 2)
            self.conv_h = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), bias=bias),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), bias=bias),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), bias=bias),
            )


            self.v = BasicTrans(dim, dim * 2)
            self.conv_v = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), bias=bias),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), bias=bias),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), bias=bias),
            )

            self.h.mask_field = [self.angRes * 2, 11]
            self.v.mask_field = [self.angRes * 2, 11]
        else:                                               # Weight shared
            self.v = self.h = BasicTrans(dim, dim * 2)
            self.conv_v = self.conv_h = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), bias=bias),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), bias=bias),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), bias=bias),
            )

            self.v.mask_field = self.h.mask_field = [self.angRes * 2, 11]

        if fusion == 'EPIFG':
            self.epi_fusion = EPIFG(dim)
        elif fusion == 'cat':
            self.epi_fusion = nn.Conv2d(dim * 2, dim, 1)
        else:
            self.epi_fusion = None

    def forward(self, x_lf):
        B_total, C, H, W = x_lf.shape
        angRes = self.args.angRes
        B_real = B_total // (angRes * angRes)

        # EPI extraction
        epi_1 = rearrange(x_lf, '(b u v) c h w -> b c (v w) u h', b=B_real, u=angRes, v=angRes)
        epi_2 = rearrange(x_lf, '(b u v) c h w -> b c (u h) v w', b=B_real, u=angRes, v=angRes)

        out_1 = self.h(epi_1)
        out_2 = self.v(epi_2)

        out_1 = rearrange(out_1, 'b c (v w) u h -> (b u v) c h w', b=B_real, v=angRes, w=W)
        out_1 = self.conv_h(out_1) + x_lf
        out_2 = rearrange(out_2, 'b c (u h) v w -> (b u v) c h w', b=B_real, u=angRes, h=H)
        out_2 = self.conv_v(out_2) + x_lf

        if self.fusion == 'sum':
            out = out_1 + out_2
        elif self.fusion == 'concat':
            out = torch.cat((out_1, out_2), dim=1)
            out = self.epi_fusion(out)
        elif self.fusion == 'EPIFG' and self.epi_fusion is not None:
            out = self.epi_fusion(out_1, out_2)
        else:
            out = out_1 + out_2

        return out





class Inter_SA_LF_Parallel(nn.Module):
    def __init__(self, args, dim, bias=False, fusion='EPIFG', is_weighted_shared=False):
        super().__init__()
        self.args = args
        self.fusion = fusion

        if is_weighted_shared == False:             # Weight not shared
            self.h = M2MTEPIAttention(in_channels=dim, embed_dim=dim * 4, mode='h')
            self.v = M2MTEPIAttention(in_channels=dim, embed_dim=dim * 4, mode='v')

            self.FFN_h = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2, bias=bias),
                nn.LeakyReLU(0.1, inplace=bias),
                nn.Linear(dim * 2, dim, bias=bias),
            )

            self.FFN_v = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2, bias=bias),
                nn.LeakyReLU(0.1, inplace=bias),
                nn.Linear(dim * 2, dim, bias=bias),
            )

            self.conv_h = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), bias=bias),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), bias=bias),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), bias=bias),
            )
            self.conv_v = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), bias=bias),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), bias=bias),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), bias=bias),
            )
        else:                                       # Weight shared
            self.h = self.v = M2MTEPIAttention(in_channels=dim, embed_dim=dim * 4, mode='h')
            self.FFN_h = self.FFN_v = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2, bias=bias),
                nn.LeakyReLU(0.1, inplace=bias),
                nn.Linear(dim * 2, dim, bias=bias),
            )
            self.conv_h = self.conv_v = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), bias=bias),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), bias=bias),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), bias=bias),
            )

        if fusion == 'EPIFG':
            self.epi_fusion = EPIFG(dim)
        elif fusion == 'concat':
            self.epi_fusion = nn.Conv2d(dim * 2, dim, 1)
        else:
            self.epi_fusion = None


    def forward(self, x_lf):
        B_total, C, H, W = x_lf.shape
        angRes = self.args.angRes
        B = B_total // (angRes * angRes)

        x_6d = rearrange(x_lf, '(b u v) c h w -> b c u v h w', b=B, u=angRes, v=angRes)

        # Horizontal
        out_h = self.h(x_6d) + x_6d
        out_h = rearrange(out_h, 'b c u v h w -> (b u v) h w c', u=angRes, v=angRes)
        out_h = self.FFN_h(out_h) + out_h
        out_h = rearrange(out_h, '(b u v) h w c -> (b u v) c h w', u=angRes, v=angRes, h=H, w=W)
        out_h = self.conv_h(out_h)


        # Vertical
        out_v = self.v(x_6d) + x_6d
        out_v = rearrange(out_v, 'b c u v h w -> (b u v) h w c', u=angRes, v=angRes)
        out_v = self.FFN_v(out_v) + out_v
        out_v = rearrange(out_v, ' (b u v) h w c-> (b u v) c h w', u=angRes, v=angRes, h=H, w=W)
        out_v = self.conv_v(out_v)

        if self.fusion == 'sum':
            out = out_h + out_v
        elif self.fusion == 'concat':
            out = torch.cat((out_h, out_v), dim=1)
            if self.epi_fusion is not None:
                out = self.epi_fusion(out)
        elif self.fusion == 'EPIFG' and self.epi_fusion is not None:
            out = self.epi_fusion(out_h, out_v)
        else:
            out = out_h + out_v


        return out


class LF_former(nn.Module):
    """
    LF-Stripformer backbone structure.
    """

    def __init__(self, args, channels, head_num=8):
        super().__init__()
        self.args = args

        self.input_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 7, 1, 3, groups=channels, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

        # Intra-EPI
        self.block1 = nn.Sequential(
            Intra_SA_LF_Parallel(args, channels),
        )

        # Inter-EPI
        self.block2 = nn.Sequential(
            Inter_SA_LF_Parallel(args, channels),
        )

        self.output_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 7, 1, 3, groups=channels, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

    def forward(self, x_lf):
        B, C, U, V, H, W = x_lf.shape
        x = rearrange(x_lf, 'b c a1 a2 h w -> (b a1 a2) c h w', a1=U, a2=V)
        x = self.input_proj(x)

        # Intra
        x = self.block1(x)
        # x = rearrange(x, '(b a1 a2) c h w -> b c a1 a2 h w', a1=U, a2=V, h=H, w=W)

        # Inter
        x = self.block2(x)
        # x = rearrange(x, 'b c a1 a2 h w -> (b a1 a2) c h w', a1=U, a2=V, h=H, w=W)

        lf = rearrange(self.output_proj(x), '(b a1 a2) c h w -> b c a1 a2 h w', a1=U, a2=V, h=H, w=W)

        return lf + x_lf


# =============================================================================
# Feature Alignment
# =============================================================================

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class FeedForward(nn.Module):
    """Gated-Dconv Feed-Forward Network (GDFN)"""

    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2,
                                kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Cross_Attention(nn.Module):
    def __init__(self, dim, ref_dim, num_heads, bias):
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv = nn.Conv2d(ref_dim, ref_dim * 2, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_dwconv = nn.Conv2d(ref_dim * 2, ref_dim * 2, kernel_size=3, stride=1, padding=1, groups=ref_dim * 2,
                                   bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.att_ca = nn.Conv2d(dim // num_heads, 2 * dim // num_heads, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, lr, ref):
        b, c, h, w = lr.shape

        q = self.q_dwconv(self.q(lr))
        kv = self.kv_dwconv(self.kv(ref))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn0 = attn.softmax(dim=-1)
        attn1 = self.relu(attn) ** 2
        attn1 = F.gelu(attn1) * attn1

        attn1 = rearrange(attn1, 'b head L c -> b c head L')
        x_att = self.att_ca(attn1)
        x_att = rearrange(x_att, 'b c head L-> b head L c')
        scale, shift = x_att.chunk(2, dim=-1)

        attn = attn0 * (1 + scale) + shift
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm_ref = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.attn = Cross_Attention(dim, dim, num_heads, bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, lr, ref):
        lr_ln = self.norm1(lr)
        ref_ln = self.norm_ref(ref)
        lr = lr + self.attn(lr_ln, ref_ln)
        out = lr + self.ffn(self.norm2(lr))
        return out


class Fusion_Block(nn.Module):
    def __init__(self, args, hidden_dim):
        super(Fusion_Block, self).__init__()
        self.args = args
        self.channels = hidden_dim // 2

        self.fusion_type = 'cross_attention'  # 'cross_attention' or 'cat'
        self.model_type = 'proposed'  # 'LF_former_and_SAS', 'LF_former_and_SAV', 'EPIT_and_EPIConv', 'LFT_and_EPIConv', 'proposed'

        # Feature Alignment
        if self.fusion_type == 'cross_attention':
            self.cross_fusion = TransformerBlock(hidden_dim)
        elif self.fusion_type == 'cat':
            self.cross_fusion = nn.Sequential(
                nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            )
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        # Global and Local feature extraction configuration
        self.pos_encoding = None

        if self.model_type == 'LF_former_and_SAS':
            self.epi_block1 = nn.Sequential(LF_former(args, hidden_dim // 2))
            self.local_conv1 = nn.Sequential(SAS_conv(hidden_dim // 2), SAS_conv(hidden_dim // 2))
            self.pos_encoding = PositionEncoding(temperature=10000)

        elif self.model_type == 'LF_former_and_SAV':
            self.epi_block1 = nn.Sequential(LF_former(args, hidden_dim // 2))
            self.local_conv1 = nn.Sequential(SAV_parallel(hidden_dim // 2), SAV_parallel(hidden_dim // 2))

        elif self.model_type == 'EPIT_and_EPIConv':
            self.epi_block1 = nn.Sequential(EPIT_AltFilter(args.angRes, hidden_dim // 2))
            self.local_conv1 = nn.Sequential(DirectionalEPIBlock(hidden_dim // 2), DirectionalEPIBlock(hidden_dim // 2))

        elif self.model_type == 'LFT_and_EPIConv':
            self.epi_block1 = nn.Sequential(LFT_AltFilter(args.angRes, hidden_dim // 2))
            self.local_conv1 = nn.Sequential(DirectionalEPIBlock(hidden_dim // 2), DirectionalEPIBlock(hidden_dim // 2))
            self.pos_encoding = PositionEncoding(temperature=10000)

        elif self.model_type == 'proposed':
            self.epi_block1 = nn.Sequential(LF_former(args, hidden_dim // 2))
            self.local_conv1 = nn.Sequential(DirectionalEPIBlock(hidden_dim // 2), DirectionalEPIBlock(hidden_dim // 2))

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.conv_RB = ResidualBlock(hidden_dim)

    def forward(self, warp, ref):
        b, c, h, w = warp.shape

        # Feature alignment
        if self.fusion_type == 'cross_attention':
            warping_fea = self.cross_fusion(warp, ref)
        else:
            warping_fea = self.cross_fusion(torch.cat([warp, ref], dim=1))

        x_1, x_2 = warping_fea.chunk(2, dim=1)

        x_ = rearrange(x_1, '(b a1 a2) c h w -> b c a1 a2 h w', a1=self.args.angRes, a2=self.args.angRes, h=h, w=w)
        x_temp = rearrange(x_, 'b c a1 a2 h w -> b c (a1 a2) h w', a1=self.args.angRes, a2=self.args.angRes, h=h, w=w)

        # Position Encoding logic for LFT
        if self.model_type == 'LFT_and_EPIConv' and self.pos_encoding is not None:
            spa_position = self.pos_encoding(x_temp, dim=[3, 4], token_dim=self.channels)
            ang_position = self.pos_encoding(x_temp, dim=[2], token_dim=self.channels)
            for m in self.modules():
                if hasattr(m, 'h'): m.h = 32
                if hasattr(m, 'w'): m.w = 32
                if hasattr(m, 'spa_position'): m.spa_position = spa_position
                if hasattr(m, 'ang_position'): m.ang_position = ang_position

        # Global feature
        x1 = self.epi_block1(x_)
        x1 = rearrange(x1, 'b c a1 a2 h w -> (b a1 a2) c h w', a1=self.args.angRes, a2=self.args.angRes, h=h, w=w)

        # Local feature
        x_2 = rearrange(x_2, '(b a1 a2) c h w -> b c a1 a2 h w', a1=self.args.angRes, a2=self.args.angRes, h=h, w=w)
        x2 = self.local_conv1(x_2)
        x2 = rearrange(x2, 'b c a1 a2 h w -> (b a1 a2) c h w', a1=self.args.angRes, a2=self.args.angRes, h=h, w=w)

        # Merge
        x = torch.cat([x1, x2], dim=1)
        x_out = self.conv_RB(x)

        return x_out


class ConfidenceFusion(nn.Module):
    def __init__(self, in_ch=9, mid_ch=32, bias=False):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_ch, 2, 1, 1, 0, bias=bias),
            nn.Softmax(dim=1)
        )

    def forward(self, rec, warp):
        diff = torch.abs(rec - warp)
        fea = torch.cat([rec, warp, diff], dim=1)
        alpha, beta = torch.split(self.fuse(fea), 1, dim=1)
        return alpha * rec + beta * warp


class get_model(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.angRes = args.angRes
        self.channels = args.channels

        # Ablation configurations
        self.inr_model = 'LF_INR'  # 'LF_INR' or 'Conv'
        self.use_confidence_fusion = True

        # Initial Feature Extraction
        self.Conv_init1 = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(3 * 4, self.channels, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            ResASPP(self.channels, self.channels)
        )
        self.Conv_init2 = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(3 * 4, self.channels, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            ResASPP(self.channels, self.channels)
        )

        # Dense Residual Fusion Blocks
        self.Fusion_blocks = nn.ModuleList([
            Fusion_Block(args, self.channels) for _ in range(4)
        ])
        self.compress = nn.Sequential(
            ChannelAttention(self.channels * 4),
            nn.Conv2d(self.channels * 4, self.channels, 1, bias=False)
        )

        # Reconstruction & INR
        if self.inr_model == 'LF_INR':
            self.Rec_Block = LF_INR(args, local_ensemble=True, feat_unfold=True, cell_decode=True)
        elif self.inr_model == 'Conv':
            self.Rec_Block = nn.Sequential(
                nn.Conv2d(self.channels, self.channels, 3, 1, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(self.channels, self.channels, 3, 1, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(self.channels, 12, 3, 1, 1),
                nn.PixelShuffle(2),
            )
        else:
            raise ValueError(f"Unknown INR model type: {self.inr_model}")

        if self.use_confidence_fusion:
            self.mask_block = ConfidenceFusion(9, self.channels)

    def forward(self, warping_Light_Field, reference_Light_Field):
        # Reshape Inputs
        warp = rearrange(warping_Light_Field, 'b c (a1 h) (a2 w) -> (b a1 a2) c h w',
                         a1=self.angRes, a2=self.angRes)
        ref = rearrange(reference_Light_Field, 'b c (a1 h) (a2 w) -> (b a1 a2) c h w',
                        a1=self.angRes, a2=self.angRes)

        warp_fea = self.Conv_init1(warp)
        ref_fea = self.Conv_init2(ref)

        # Dense Residual Fusion
        x = warp_fea
        fea_list = []

        for fusion in self.Fusion_blocks:
            x = fusion(x, ref_fea)
            fea_list.append(x)

        fea_dense = torch.cat(fea_list, dim=1)
        fea = self.compress(fea_dense)

        # Implicit Reconstruction
        if self.inr_model == 'LF_INR':
            fea = rearrange(fea, '(b an1 an2) c h w -> b c an1 an2 h w',
                            an1=self.angRes, an2=self.angRes)
            res = self.Rec_Block(fea)
            rec = rearrange(res, 'b c an1 an2 h w -> (b an1 an2) c h w',
                            an1=self.angRes, an2=self.angRes)
        elif self.inr_model == 'Conv':
            rec = self.Rec_Block(fea)
        else:
            rec = fea  # fallback

        # Confidence Fusion
        if self.use_confidence_fusion:
            out = self.mask_block(rec, warp)
            out = rearrange(out, '(b an1 an2) c h w -> b c (an1 h) (an2 w)',
                            an1=self.angRes, an2=self.angRes)
            rec = rearrange(rec, '(b an1 an2) c h w -> b c (an1 h) (an2 w)',
                            an1=self.angRes, an2=self.angRes)
            return rec, out
        else:
            out = rearrange(rec, '(b an1 an2) c h w -> b c (an1 h) (an2 w)',
                            an1=self.angRes, an2=self.angRes) + warp
            return out


# ###############################################################################
class get_loss(nn.Module):
    """
    Total = rec + w_detail*detail + w_lpips*lpips
    - Safe LPIPS preprocessing (to 3ch, [-1,1])
    - AMP-safe: compute LPIPS in fp32 (autocast disabled)
    - Logs each term in self.last_losses
    """

    def __init__(self, args):
        super(get_loss, self).__init__()
        self.args = args
        self.angRes = args.angRes
        self.rec_loss = ReconstructionLoss()
        self.detail_loss = DetailLoss()

        if getattr(args, "w_lpips", 0) > 0:
            if LPIPS is None:
                raise ImportError("lpips not installed. Try: pip install lpips")
            self.vgg_loss = LPIPS(net='vgg').to(args.device)
        else:
            self.vgg_loss = None

        self.input_in_01 = getattr(args, "input_in_01", True)
        self.last_losses: Dict[str, float] = {}

    def _prep_for_lpips(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B,C,H,W], got {x.shape}")
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        if self.input_in_01:
            x = torch.clamp(x * 2.0 - 1.0, -1.0, 1.0)
        else:
            x = torch.clamp(x, -1.0, 1.0)
        return x

    def forward(self, Rec_Light_Field: torch.Tensor, warp_Light_Field: torch.Tensor, Ground_Truth: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_dict: bool = False, **kwargs):
        """
        Inputs:
            Rec_Light_Field: [B,C,H,W]
            Ground_Truth   : [B,C,H,W]
            mask (opt)     : [B,1,H,W] or [B,H,W], 1=valid
        Returns:
            total (and optionally dict of components if return_dict=True)
        """
        losses: Dict[str, torch.Tensor] = {}

        # 1) Reconstruction
        rec = self.rec_loss(Rec_Light_Field, Ground_Truth)
        warp = self.rec_loss(warp_Light_Field, Ground_Truth)
        total = rec + warp
        losses["rec"] = total

        # 2) Detail
        w_detail = float(getattr(self.args, "w_detail", 0.0))
        if w_detail > 0.0:
            det = self.detail_loss(Rec_Light_Field, Ground_Truth)
            warp_det = self.detail_loss(warp_Light_Field, Ground_Truth)
            total = total + w_detail * det + w_detail * warp_det
            losses["detail"] = det + warp_det

        # 3) LPIPS
        w_lpips = float(getattr(self.args, "w_lpips", 0.0))
        if w_lpips > 0.0 and self.vgg_loss is not None:
            # AMP 下用 fp32 计算更稳
            with torch.amp.autocast('cuda', enabled=False):
                Rec_Light_Field_Stack = rearrange(Rec_Light_Field, 'b c (a1 h) (a2 w) -> (b a1 a2) c h w',
                                                  a1=self.angRes, a2=self.angRes)
                Warp_Light_Field_Stack = rearrange(warp_Light_Field, 'b c (a1 h) (a2 w) -> (b a1 a2) c h w',
                                                   a1=self.angRes, a2=self.angRes)
                Ground_Truth_Stack = rearrange(Ground_Truth, 'b c (a1 h) (a2 w) -> (b a1 a2) c h w',
                                               a1=self.angRes, a2=self.angRes)
                x_lp = self._prep_for_lpips(Rec_Light_Field_Stack.float())
                y_lp = self._prep_for_lpips(Ground_Truth_Stack.float())
                lp_1 = self.vgg_loss(x_lp, y_lp).mean()

                x_lp = self._prep_for_lpips(Warp_Light_Field_Stack.float())
                y_lp = self._prep_for_lpips(Ground_Truth_Stack.float())
                lp_2 = self.vgg_loss(x_lp, y_lp).mean()

                lp = lp_1 + lp_2

            total = total + w_lpips * lp
            losses["lpips"] = lp

        losses["total"] = total

        try:
            self.last_losses = {k: v.detach().item() for k, v in losses.items()}
        except Exception:
            pass

        if return_dict:
            return total, losses
        return total


def weights_init(m):
    pass


# =============================================================================
# Main Execution Block (Profiling)
# =============================================================================

if __name__ == '__main__':
    try:
        from config import args
    except ImportError:
        # Mock args if config.py is missing
        class Args:
            angRes = 7
            channels = 64
            device = 'cuda'


        args = Args()

    try:
        from thop import profile, clever_format
    except ImportError:
        print("Please install thop: pip install thop")
        exit()

    # 1. Load Model
    model = get_model(args).cuda()
    model.eval()

    # 2. Prepare Inputs
    h, w = 7 * 64, 7 * 64
    lr_input1 = torch.randn(1, 3, h, w).cuda()
    lr_input2 = torch.randn(1, 3, h, w).cuda()

    print("------------------------------------------------")
    print(f"Model: HIASRNet | Input shape: {lr_input1.shape}")

    # 3. Calculate FLOPs
    print("Calculating FLOPs and Params...")
    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    # Note: Pass tuple of inputs to profile
    macs, params = profile(model, inputs=(lr_input1, lr_input2), verbose=False)
    macs_fmt, params_fmt = clever_format([macs, total_params], "%.3f")

    print(f"FLOPs (MACs): {macs_fmt}")
    print(f"Params:       {params_fmt}M")
    print("------------------------------------------------")