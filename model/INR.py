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
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
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
    def __init__(self, in_dim, local_ensemble=True, feat_unfold=True, cell_decode=True, device=None):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.device = device

        if self.feat_unfold:
            in_dim *= 9
        in_dim += 2 + 4 * L    # attach coord
        if self.cell_decode:
            in_dim += 2

        # self.imnet = nn.Sequential(
        #     MLP(in_dim=in_dim, out_dim=3, hidden_list=hidden_list),
        #     nn.Sigmoid())
        self.imnet = MLP(in_dim=in_dim, out_dim=3, hidden_list=hidden_list)

    def query_rgb(self, inp, coord, cell=None, in_b=1, in_ah=7, in_aw=7):
        feat = inp
        if self.feat_unfold:
            feat = functional.unfold(feat, kernel_size=3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(self.device) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        coord = coord.repeat([in_b * in_ah * in_aw, 1, 1])
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                bs, q, h, w = feat.shape
                q_feat = feat.view(bs, q, -1).permute(0, 2, 1)   # [b*an2,h*w,ch*9]

                bs, q, h, w = feat_coord.shape
                q_coord = feat_coord.view(bs, q, -1).permute(0, 2, 1)

                points_enc = self.positional_encoding(q_coord, L=L)
                q_coord = torch.cat([q_coord, points_enc], dim=-1)  # [B,...,6L+3]

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell = rel_cell.repeat([in_b * in_ah * in_aw, 1, 1])
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
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
        ret = rearrange(ret, '(b ah aw) c h w -> b c ah aw h w', b=in_b, ah=in_ah, aw=in_aw)
        return ret

    def forward(self, inp):
        in_b, _, in_ah, in_aw, _, _ = inp.shape
        inp = rearrange(inp, 'b c ah aw h w -> (b ah aw) c h w')
        h, w = inp.shape[2], inp.shape[3]
        coord = make_coord((h, w)).to(self.device)
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        cell = cell.unsqueeze(0)
        coord = coord.unsqueeze(0)  # [1,h*w,2]

        points_enc = self.positional_encoding(coord, L=L)
        coord = torch.cat([coord, points_enc], dim=-1)    # [B,...,6L+3]
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