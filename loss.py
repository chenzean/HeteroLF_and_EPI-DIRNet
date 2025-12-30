import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import torchvision.models as models
import os
import sys

def L1_Charbonnier_loss(X, Y):
    eps = 1e-6
    diff = torch.add(X, -Y)
    error = torch.sqrt(diff * diff + eps)
    Charbonnier_loss = torch.sum(error) / torch.numel(error)
    return Charbonnier_loss


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, sr, hr):
        rec_loss = L1_Charbonnier_loss(sr, hr)
        return rec_loss


class DetailLoss(nn.Module):
    def __init__(self):
        super(DetailLoss, self).__init__()

    def forward(self, sr, hr):    # [B*U*V, 1, H, W]
        sr_grad1 = sr[:, :, 1:, :] - sr[:, :, :-1, :]
        hr_grad1 = hr[:, :, 1:, :] - hr[:, :, :-1, :]
# -------------------------------------------------------------------------------------------------------------
        sr_grad2 = sr[:, :, :, 1:] - sr[:, :, :, :-1]
        hr_grad2 = hr[:, :, :, 1:] - hr[:, :, :, :-1]

        detail_4d_loss1 = L1_Charbonnier_loss(sr_grad1, hr_grad1)
        detail_4d_loss2 = L1_Charbonnier_loss(sr_grad2, hr_grad2)
        detail_4d_loss = (detail_4d_loss1 + detail_4d_loss2) / 2.
        return detail_4d_loss
