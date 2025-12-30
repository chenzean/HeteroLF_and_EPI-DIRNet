# -*- coding: utf-8 -*-
# @Time    : 2025-10-07 10:40
# @Author  : Chen Zean
# @Site    : 
# @File    : moudle.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F

# ################################################################################
# from https://github.com/c-yn/SANet/blob/main/Desnowing/models/layers.py
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        if filter:
            self.cubic_11 = cubic_attention(in_channel // 2, group=1, kernel=11)
            self.cubic_7 = cubic_attention(in_channel // 2, group=1, kernel=7)
        self.filter = filter

    def forward(self, x):
        out = self.conv1(x)
        if self.filter:
            out = torch.chunk(out, 2, dim=1)
            out_11 = self.cubic_11(out[0])
            out_7 = self.cubic_7(out[1])
            out = torch.cat((out_11, out_7), dim=1)
        out = self.conv2(out)
        return out + x


class cubic_attention(nn.Module):
    def __init__(self, dim, group, kernel) -> None:
        super().__init__()

        self.H_spatial_att = spatial_strip_att(dim, group=group, kernel=kernel)
        self.W_spatial_att = spatial_strip_att(dim, group=group, kernel=kernel, H=False)
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)
        return self.gamma * out + x * self.beta


class spatial_strip_att(nn.Module):
    def __init__(self, dim, kernel=5, group=2, H=True) -> None:
        super().__init__()

        self.k = kernel
        pad = kernel // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel // 2, 1) if H else (1, kernel // 2)

        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group * kernel, kernel_size=1, stride=1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Sigmoid()

    def forward(self, x):
        filter = self.ap(x)
        filter = self.conv(filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel).reshape(n, self.group, c // self.group, self.k, h * w)
        n, c1, p, q = filter.shape
        filter = filter.reshape(n, c1 // self.k, self.k, p * q).unsqueeze(2)
        filter = self.filter_act(filter)
        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)
        return out

# ################################################################################
# from https://github.com/pp00704831/Stripformer-ECCV-2022-/blob/main/models/Stripformer.py

import math
from einops import rearrange

# ------------------------------
# PEG (Conditional Positional Encoding)
# ------------------------------
class PEG(nn.Module):
    def __init__(self, hidden_size):
        super(PEG, self).__init__()
        self.PEG = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size)
    def forward(self, x):
        return self.PEG(x) + x


# ------------------------------
# Attention
# ------------------------------
class Attention(nn.Module):
    def __init__(self, head_num):
        super(Attention, self).__init__()
        self.num_attention_heads = head_num
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        B, N, C = x.size()
        attention_head_size = int(C / self.num_attention_heads)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, q, k, v):
        B, N, C = q.size()
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        att = self.softmax(att)
        out = torch.matmul(att, v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(B, N, C)
        return out


class Mlp(nn.Module):
    def __init__(self, hidden_size):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)
        self.act_fn = torch.nn.functional.gelu
    def forward(self, x):
        return self.fc2(self.act_fn(self.fc1(x)))



# ------------------------------
# Light Field Intra-Strip Attention
# ------------------------------
class Intra_SA_LF(nn.Module):
    def __init__(self, dim, head_num, ang_res_u=7, ang_res_v=7):
        super().__init__()
        self.hidden_size = dim // 2
        self.head_num = head_num
        self.attention_norm = nn.LayerNorm(dim)
        self.conv_input = nn.Conv2d(dim, dim, 1)
        self.qkv_local_h = nn.Linear(self.hidden_size, self.hidden_size * 3)
        self.qkv_local_v = nn.Linear(self.hidden_size, self.hidden_size * 3)
        self.fuse_out = nn.Conv2d(dim, dim, 1)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = Mlp(dim)
        self.attn = Attention(head_num=head_num)
        self.PEG = PEG(dim)

    def forward(self, x_lf, u_idx, v_idx):
        # x_lf: [B, C, U, V, H, W]
        B, C, U, V, H, W = x_lf.shape

        epi = rearrange(x_lf, 'b c u v h w -> (b v w) c u h')

        h = epi
        B, C, H_, W_ = epi.shape
        x = epi.view(B, C, H_ * W_).permute(0, 2, 1)
        x = self.attention_norm(x).permute(0, 2, 1).view(B, C, H_, W_)

        # split horizontal & vertical
        x_input = torch.chunk(self.conv_input(x), 2, dim=1)
        f_h = x_input[0].permute(0, 2, 3, 1).view(B * H_, W_, C // 2)   # horizontal
        f_v = x_input[1].permute(0, 3, 2, 1).view(B * W_, H_, C // 2)   # vertical

        qkv_h = torch.chunk(self.qkv_local_h(f_h), 3, dim=2)
        qkv_v = torch.chunk(self.qkv_local_v(f_v), 3, dim=2)
        q_h, k_h, v_h = qkv_h
        q_v, k_v, v_v = qkv_v

        attn_h = self.attn(q_h, k_h, v_h)
        attn_v = self.attn(q_v, k_v, v_v)

        attn_h = attn_h.view(B, H_, W_, C // 2).permute(0, 3, 1, 2)
        attn_v = attn_v.view(B, W_, H_, C // 2).permute(0, 3, 2, 1)
        attn_out = self.fuse_out(torch.cat((attn_h, attn_v), dim=1))
        x = attn_out + h

        # FFN
        x = x.view(Bv, C, H_ * W_).permute(0, 2, 1)
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        x = x.permute(0, 2, 1).view(Bv, C, H_, W_)
        x = self.PEG(x)
        return x.view(B, V, C, U, H, W).permute(0, 2, 3, 1, 4, 5).contiguous()


# ------------------------------
# Light Field Inter-Strip Attention
# ------------------------------
class Inter_SA_LF(nn.Module):
    def __init__(self, dim, head_num):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.conv_input = nn.Conv2d(dim, dim, 1)
        self.conv_h = nn.Conv2d(dim // 2, 3 * (dim // 2), 1)
        self.conv_v = nn.Conv2d(dim // 2, 3 * (dim // 2), 1)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = Mlp(dim)
        self.fuse_out = nn.Conv2d(dim, dim, 1)
        self.attn = Attention(head_num=head_num)
        self.PEG = PEG(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        h = x
        x = x.view(B, C, H * W).permute(0, 2, 1)
        x = self.attention_norm(x).permute(0, 2, 1).view(B, C, H, W)

        x_input = torch.chunk(self.conv_input(x), 2, dim=1)
        f_h = torch.chunk(self.conv_h(x_input[0]), 3, dim=1)
        f_v = torch.chunk(self.conv_v(x_input[1]), 3, dim=1)
        qh, kh, vh = f_h
        qv, kv, vv = f_v

        qh = qh.permute(0, 2, 3, 1).view(B * H, W, -1)
        kh = kh.permute(0, 2, 3, 1).view(B * H, W, -1)
        vh = vh.permute(0, 2, 3, 1).view(B * H, W, -1)
        qv = qv.permute(0, 3, 2, 1).view(B * W, H, -1)
        kv = kv.permute(0, 3, 2, 1).view(B * W, H, -1)
        vv = vv.permute(0, 3, 2, 1).view(B * W, H, -1)

        attn_h = self.attn(qh, kh, vh)
        attn_v = self.attn(qv, kv, vv)

        attn_h = attn_h.view(B, H, W, C // 2).permute(0, 3, 1, 2)
        attn_v = attn_v.view(B, W, H, C // 2).permute(0, 3, 2, 1)
        attn_out = self.fuse_out(torch.cat((attn_h, attn_v), dim=1))
        x = attn_out + h

        x = x.view(B, C, H * W).permute(0, 2, 1)
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return self.PEG(x)


# ------------------------------
# LF-Stripformer: 主结构
# ------------------------------
class LF_Stripformer(nn.Module):
    def __init__(self, in_ch=3, dim=128, head_num=4, ang_res_u=7, ang_res_v=7):
        super().__init__()
        self.input_proj = nn.Conv3d(in_ch, dim, 3, padding=1)
        self.block1 = Intra_SA_LF(dim, head_num, ang_res_u, ang_res_v)
        self.block2 = Inter_SA_LF(dim, head_num)
        self.block3 = Intra_SA_LF(dim, head_num, ang_res_u, ang_res_v)
        self.block4 = Inter_SA_LF(dim, head_num)
        self.output_proj = nn.Conv3d(dim, in_ch, 3, padding=1)

    def forward(self, x_lf, u_idx=3, v_idx=3):
        # x_lf: [B, C, U, V, H, W]
        B, C, U, V, H, W = x_lf.shape
        x = self.input_proj(x_lf)  # [B, dim, U, V, H, W]
        x = self.block1(x, u_idx, v_idx)
        x = self.block2(x.view(B, -1, H, W))
        x = self.block3(x, u_idx, v_idx)
        x = self.block4(x.view(B, -1, H, W))
        return self.output_proj(x) + x_lf
