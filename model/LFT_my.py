# -*- coding: utf-8 -*-
# @Time    : 2025-12-22 9:27
# @Author  : Chen Zean
# @Site    : 
# @File    : LFT_my.py
# @Software: PyCharm
import torch.nn as nn
import torch
from einops import rearrange
import math
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3,
                                   bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, lr):
        b, c, h, w = lr.shape

        qkv = self.qkv_dwconv(self.qkv(lr))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn0 = attn.softmax(dim=-1)
        out = (attn0 @ v)

        out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


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


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SpaTrans(nn.Module):
    def __init__(self, channels, angRes, MHSA_params=None):
        super(SpaTrans, self).__init__()
        self.angRes = angRes
        self.kernel_field = 3
        self.spa_dim = channels * 2

        # token embedding: unfold 后每个 token = channels * k^2
        self.MLP = nn.Linear(channels * (self.kernel_field ** 2), self.spa_dim, bias=False)

        # ✅ Pre-LN: attention 前的 LN
        self.norm_attn = nn.LayerNorm(self.spa_dim)
        # ✅ Pre-LN: ffn 前的 LN
        self.norm_ffn = nn.LayerNorm(self.spa_dim)

        # ✅ Conv2d Attention / FFN
        self.attention = Attention(dim=self.spa_dim, num_heads=4, bias=False)
        self.feed_forward = FeedForward(self.spa_dim, ffn_expansion_factor=2.66, bias=False)

        # spa_dim -> channels
        self.proj_out = nn.Conv3d(self.spa_dim, channels, kernel_size=1, bias=False)

    def SAI2Token(self, buffer):
        """
        buffer: [b, c, a, h, w]
        token : [L=h*w, N=b*a, D=spa_dim]
        """
        buffer2d = rearrange(buffer, 'b c a h w -> (b a) c h w')
        token = F.unfold(buffer2d, kernel_size=self.kernel_field,
                         padding=self.kernel_field // 2).permute(2, 0, 1)  # [L, N, c*k*k]
        token = self.MLP(token)  # [L, N, D]
        return token

    def Token2SAI(self, token, b, a, h, w):
        """
        token: [L=h*w, N=b*a, D]
        -> out: [b, channels, a, h, w]
        """
        x = rearrange(token, '(h w) (b a) d -> b d a h w', h=h, w=w, b=b, a=a)  # [b, D, a, h, w]
        x = self.proj_out(x)  # [b, channels, a, h, w]
        return x

    def forward(self, buffer):
        """
        buffer: [b, c, a, h, w]
        """
        b, c, a, h, w = buffer.shape
        assert a == self.angRes ** 2, f"Expected a=angRes^2={self.angRes**2}, but got {a}"

        # -------------------------
        # Tokenize
        # -------------------------
        token = self.SAI2Token(buffer)  # [L, N, D], L=h*w, N=b*a

        # =========================================================
        # 1) Pre-LN Attention Block: token + Attn(LN(token))
        # =========================================================
        token_norm = self.norm_attn(token)  # [L,N,D]
        x = rearrange(token_norm, '(h w) n d -> n d h w', h=h, w=w)     # [N,D,h,w]

        attn_out = self.attention(x)  # [N,D,h,w]
        attn_out = rearrange(attn_out, 'n d h w -> (h w) n d')          # [L,N,D]

        token = token + attn_out  # residual

        # =========================================================
        # 2) Pre-LN FFN Block: token + FFN(LN(token))
        # =========================================================
        token_ffn_norm = self.norm_ffn(token)  # [L,N,D]
        x2 = rearrange(token_ffn_norm, '(h w) n d -> n d h w', h=h, w=w)  # [N,D,h,w]

        ffn_out = self.feed_forward(x2)  # [N,D,h,w]
        ffn_out = rearrange(ffn_out, 'n d h w -> (h w) n d')            # [L,N,D]

        token = token + ffn_out  # residual

        # -------------------------
        # Detokenize
        # -------------------------
        out = self.Token2SAI(token, b=b, a=a, h=h, w=w)
        return out


import torch
import torch.nn as nn
from einops import rearrange

class AngTrans(nn.Module):
    """
    AngTrans（严格 Pre-LN，按 SpaTrans 的最终写法对齐）：
    - 不用位置编码
    - 不用 mask
    - Attention/FFN 是 Conv2d
    - Transformer 标准：每个子层前都有 LN（Pre-LN）
    输入:  buffer [b, c, a, h, w], a=angRes^2
    输出:  buffer [b, c, a, h, w]
    """
    def __init__(self, channels, angRes, ffn_expansion_factor=2.66, bias=False, num_heads=4):
        super().__init__()
        self.angRes = angRes
        self.ang_dim = channels

        # ✅ Pre-LN: attn 前 LN、ffn 前 LN
        self.norm_attn = nn.LayerNorm(self.ang_dim)
        self.norm_ffn  = nn.LayerNorm(self.ang_dim)

        # Conv2d Attention / FFN（你自己的实现）
        self.attention = Attention(dim=self.ang_dim, num_heads=num_heads, bias=bias)
        self.feed_forward = FeedForward(self.ang_dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

    def SAI2Token(self, buffer):
        # [b,c,a,h,w] -> [a, b*h*w, c]
        return rearrange(buffer, 'b c a h w -> a (b h w) c')

    def Token2SAI(self, token, b, h, w):
        # [a, b*h*w, c] -> [b,c,a,h,w]
        a = self.angRes ** 2
        return rearrange(token, 'a (b h w) c -> b c a h w', a=a, b=b, h=h, w=w)

    def forward(self, buffer):
        b, c, a, h, w = buffer.shape
        assert a == self.angRes ** 2, f"Expected a=angRes^2={self.angRes**2}, but got {a}"
        assert c == self.ang_dim, f"Expected channels={self.ang_dim}, but got {c}"

        U = V = self.angRes  # A = U*V

        # -------------------------
        # Tokenize
        # -------------------------
        token = self.SAI2Token(buffer)  # [A, N, C], N=b*h*w

        # =========================================================
        # 1) Pre-LN Attention: token + Attn(LN(token))
        # =========================================================
        token_attn_norm = self.norm_attn(token)  # [A,N,C]
        x = rearrange(token_attn_norm, '(u v) n c -> n c u v', u=U, v=V)  # [N,C,U,V]

        attn_out = self.attention(x)  # [N,C,U,V]
        attn_out = rearrange(attn_out, 'n c u v -> (u v) n c')            # [A,N,C]

        token = token + attn_out  # residual

        # =========================================================
        # 2) Pre-LN FFN: token + FFN(LN(token))
        # =========================================================
        token_ffn_norm = self.norm_ffn(token)  # [A,N,C]
        x2 = rearrange(token_ffn_norm, '(u v) n c -> n c u v', u=U, v=V)  # [N,C,U,V]

        ffn_out = self.feed_forward(x2)  # [N,C,U,V]
        ffn_out = rearrange(ffn_out, 'n c u v -> (u v) n c')              # [A,N,C]

        token = token + ffn_out  # residual

        # -------------------------
        # Detokenize
        # -------------------------
        out = self.Token2SAI(token, b=b, h=h, w=w)
        return out


class AltFilter(nn.Module):
    def __init__(self, angRes, channels, MHSA_params=8):
        super(AltFilter, self).__init__()
        self.angRes = angRes
        self.spa_trans = SpaTrans(channels, angRes, MHSA_params)
        self.ang_trans = AngTrans(channels, angRes, MHSA_params)

    def forward(self, buffer):
        buffer = rearrange(buffer, 'b c u v h w -> b c (u v) h w ', u = self.angRes, v = self.angRes)
        buffer = self.ang_trans(buffer)
        buffer = self.spa_trans(buffer)

        buffer = rearrange(buffer, 'b c (u v) h w -> b c u v h w ', u=self.angRes, v=self.angRes)
        return buffer
