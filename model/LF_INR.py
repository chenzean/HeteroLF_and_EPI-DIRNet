import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
from einops import rearrange


hidden_list = [128, 128, 128]
L = 8

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    # ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


# Implicit Neural Representation (INR)
class INR(nn.Module):
    def __init__(self, args, local_ensemble=True, feat_unfold=True, cell_decode=True, device=None):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.device = args.device
        in_dim = args.channels

        if self.feat_unfold:
            in_dim *= 9 * 9

        in_dim += 4 + 8 * L    # attach coord

        if self.cell_decode:
            in_dim += 4

        # self.imnet = nn.Sequential(
        #     MLP(in_dim=in_dim, out_dim=3, hidden_list=hidden_list),
        #     nn.Sigmoid())
        self.imnet = MLP(in_dim=in_dim, out_dim=3 * 2 * 2, hidden_list=hidden_list)

        self.up = nn.PixelShuffle(2)

    def query_rgb(self, inp, coord, cell=None, in_b=1, in_ah=7, in_aw=7):
        feat = inp
        if self.feat_unfold:
            Baw, C0, H, W = feat.shape

            # 空间 3×3 展开：NCHW -> (N, 9C, H, W)
            feat = functional.unfold(feat, kernel_size=3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3]) # feat: [b*an1*an2, c, h, w]
            feat = feat.view(Baw, 9 * C0, H, W)

            # 角度 3×3 局部展开
            feat = rearrange(feat, '(b ah aw) c h w -> (b h w) c ah aw', b=in_b, ah=in_ah, aw=in_aw)
            feat = functional.unfold(feat, kernel_size=3, padding=1)  # (b*h*w, 9*(9C), ah*aw)
            feat = feat.view(in_b * H * W, 9 * 9 * C0, in_ah, in_aw)
            feat = rearrange(feat, '(b h w) c ah aw -> (b ah aw) c h w', b=in_b, h=H, w=W)
            # print(feat.shape)  # [49, 2592, 32, 32]


        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        B, H, W = in_b, feat.shape[-2], feat.shape[-1]

        # -------- 1) 空间坐标 (x,y) -> [B, 2, ah, aw, H, W] --------
        # 先得到 [H, W, 2] 的 xy 网格，归一化到 [-1,1]
        xy_hw = make_coord((H, W), flatten=False).to(self.device)  # [H, W, 2]
        # 变为 [2, H, W]，再扩到 [B, 2, ah, aw, H, W]
        xy_map = xy_hw.permute(2, 0, 1)  # [2, H, W]
        xy_map = xy_map.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1, 2, 1, 1, H, W]
        xy_map = xy_map.expand(B, 2, in_ah, in_aw, H, W)  # [B, 2, ah, aw, H, W]

        # -------- 2) 角度坐标 (u,v) -> [B, 2, ah, aw, H, W] --------
        # 先得到 [ah, aw, 2] 的 uv 网格，归一化到 [-1,1]
        uv_aa = make_coord((in_ah, in_aw), flatten=False).to(self.device)  # [ah, aw, 2]
        # 变为 [2, ah, aw]，再扩到 [B, 2, ah, aw, H, W]
        uv_map = uv_aa.permute(2, 0, 1)  # [2, ah, aw]
        uv_map = uv_map.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, 2, ah, aw, 1, 1]
        uv_map = uv_map.expand(B, 2, in_ah, in_aw, H, W)  # [B, 2, ah, aw, H, W]

        # -------- 3) 如需四维坐标 [x,y,u,v] 合并到一起：--------
        coord4_map = torch.cat([xy_map, uv_map], dim=1)  # [B, 4, ah, aw, H, W]

        preds = []
        areas = []
        # coord = coord.repeat([in_b * in_ah * in_aw, 1, 1])
        coord = rearrange(coord, 'b (an1 an2) q n -> b (an1 an2 q) n', b=in_b, an1=in_ah, an2=in_aw)
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                bs, q, h, w = feat.shape
                bs = bs // (in_ah * in_aw)
                # q_feat = feat.view(bs, q, -1).permute(0, 2, 1)   # [b*an2,h*w,ch*9]
                q_feat = rearrange(feat, '(b an1 an2) c h w -> b (an1 an2 h w) c', b=bs, an1=in_ah, an2=in_aw)   # [b,an1 * an2 * h * w, ch*9]
                # print(q_feat.shape)  # [1, 50176, 2592]

                bs, q, an1, an2, h, w = coord4_map.shape  # [1, 4, 7, 7, 32, 32]
                q_coord = coord4_map.view(bs, q, -1).permute(0, 2, 1)  # [b,an1 * an2 * h * w,ch*9]

                points_enc = self.positional_encoding(q_coord, L=L)
                q_coord = torch.cat([q_coord, points_enc], dim=-1)  # [B,...,6L+3]
                # print(q_coord.shape)  # [1, 50176, 68]

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    # rel_cell = cell.clone()
                    # rel_cell = rel_cell.repeat([in_b * in_ah * in_aw, 1, 1])
                    rel_cell = rearrange(cell, 'b (an1 an2) q n -> b (an1 an2 q) n', b=in_b, an1=in_ah, an2=in_aw)
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]

                # print(inp.shape)   # [1, 50176, 2664]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                # print(pred.shape)  # [1, 50176, 12]
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]
            areas[0] = areas[3]
            areas[3] = t
            t = areas[1]
            areas[1] = areas[2]
            areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        bs, q, h, w = feat.shape
        ret = ret.view(bs, h, w, -1).permute(0, 3, 1, 2)

        ret = self.up(ret)

        ret = rearrange(ret, '(b ah aw) c h w -> b c ah aw h w', b=in_b, ah=in_ah, aw=in_aw)
        return ret

    def forward(self, inp):
        '''
        Arguments:
            inp: [b, c, ah, aw, h, w]
        Returns:
            ret: [b, 3, ah, aw, h, w]
        '''
        in_b, _, in_ah, in_aw, _, _ = inp.shape
        inp = rearrange(inp, 'b c ah aw h w -> (b ah aw) c h w')

        B = in_b * in_ah * in_aw
        h, w = inp.shape[2], inp.shape[3]

        # --- 空间坐标 (x,y) in [-1,1]：先 [H*W,2] 再扩到 [B,H*W,2]
        xy = make_coord((h, w), flatten=True).to(self.device)  # [H*W, 2]
        xy = xy.unsqueeze(0).expand(B, h * w, 2)  # [B, H*W, 2]

        # --- 角度坐标 (u,v) in [-1,1]：先 [ah*aw,2]，再铺到 [B,H*W,2]
        uv = make_coord((in_ah, in_aw), flatten=True).to(self.device)  # [ah*aw, 2]
        uv = uv.unsqueeze(0).repeat(in_b, 1, 1).view(B, 1, 2).expand(B, h * w, 2)  # [B,H*W,2]

        # --- 四维查询坐标 [B,H*W,4]
        coord = torch.cat([xy, uv], dim=-1)  # [an1*an2, H*W, 4]

        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w

        cell[:, 2] *= 2 / in_ah
        cell[:, 3] *= 2 / in_aw

        cell = cell.unsqueeze(0)     # [1, an1*an2, H*W, 4]
        coord = coord.unsqueeze(0)  # [1, an1*an2, H*W, 4]

        points_enc = self.positional_encoding(coord, L=L)         # [1, an1*an2, H*W, 64]
        coord = torch.cat([coord, points_enc], dim=-1)     # [1, an1*an2, H*W, 68]

        return self.query_rgb(inp, coord, cell, in_b, in_ah, in_aw)

    def positional_encoding(self, input, L):   # [B,...,N]
        shape = input.shape
        freq = 2 ** torch.arange(L, dtype=torch.float32).to(self.device) * np.pi    # [L]
        spectrum = input[..., None] * freq              # [B,...,N,L]
        sin, cos = spectrum.sin(), spectrum.cos()       # [B,...,N,L]
        input_enc = torch.stack([sin, cos], dim=-2)     # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1], -1)     # [B,...,2NL]
        return input_enc


if __name__ == '__main__':
    model = INR(in_dim=32, device='cuda:0').to('cuda:0')
    x = torch.randn(1, 32, 7, 7, 32, 32).to('cuda:0')
    with torch.no_grad():
        y = model(x)
    print(y.shape)