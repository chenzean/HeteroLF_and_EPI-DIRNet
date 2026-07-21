# -*- coding: utf-8 -*-
"""
单张中心视图 + 深度/视差 → 合成 U×V 视图（不做滑块）

本版改动要点（满足你的新要求）：
1) index(001/002/…) → test_mapping.json[index_to_name] → 场景名（如 Scenes_025）
2) 解析 D:\Test_demo\disp_results\results.txt，抓取该场景的 disp_max(px)
3) invdepth：disp(x,y) = normalize01(depth)(x,y) * disp_max(px)
   depth / disparity / dxdy 模式保留，不受影响
"""

import os
import re
import json
import argparse
import struct
from typing import Optional, Tuple, Dict

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import softsplat

# ==============================
# 基础 I/O
# ==============================

def read_pfm(filename: str) -> np.ndarray:
    with open(filename, 'rb') as f:
        tag = f.readline().decode('latin-1').strip()
        if tag not in ('PF', 'Pf'):
            raise ValueError(f'Not a PFM: {filename}')
        line = f.readline().decode('latin-1').strip()
        while line.startswith('#'):
            line = f.readline().decode('latin-1').strip()
        w, h = map(int, line.split())
        scale_line = f.readline().decode('latin-1').strip()
        while scale_line.startswith('#'):
            scale_line = f.readline().decode('latin-1').strip()
        scale = float(scale_line)
        big = scale > 0
        if scale < 0:
            scale = -scale
        data = f.read()
        n = w * h * (3 if tag == 'PF' else 1)
        fmt = ('>' if big else '<') + f'{n}f'
        arr = np.array(struct.unpack(fmt, data), dtype=np.float32)
        if tag == 'PF':
            arr = arr.reshape(h, w, 3)
            arr = np.flipud(arr)
            # 灰度化
            arr = arr[...,0]*0.299 + arr[...,1]*0.587 + arr[...,2]*0.114
        else:
            arr = arr.reshape(h, w)
            arr = np.flipud(arr)
        arr *= scale
        return arr.astype(np.float32)

def load_depth_any(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pfm':
        return read_pfm(path)
    if ext == '.npy':
        return np.load(path).astype(np.float32)
    img = Image.open(path).convert('F')
    return np.array(img, dtype=np.float32)

# ==============================
# 解析 results.txt 得到每个场景的 disp_max(px)
# ==============================

_SCENE_COL_RE = re.compile(r'^\s*(Scenes_\d{3})\s+([-\d\.Ee+]+)\s+([-\d\.Ee+]+)')

def load_scene_to_dispmax(results_txt: str) -> Dict[str, float]:
    """
    解析类似：
    Scene                 disp_min(px)  disp_max(px)     Pairs
    Scenes_001                0.000496      3.405984         2
    ...
    返回：{'Scenes_001': 3.405984, ...}
    """
    scene2max = {}
    if not (results_txt and os.path.isfile(results_txt)):
        return scene2max
    with open(results_txt, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.strip()
            m = _SCENE_COL_RE.match(line)
            if m:
                scene = m.group(1)
                # group(2) 是 disp_min, group(3) 是 disp_max
                try:
                    disp_max = float(m.group(3))
                except Exception:
                    continue
                scene2max[scene] = disp_max
    return scene2max

# ==============================
# 几何：反投影
# ==============================

def coordinate_transform(x, scale):
    # 与参考实现保持一致：像素中心修正
    return x / scale - 0.5 * (1 - 1.0 / scale)

# def warp_back_projection_no_range(x, flo, scale=1.0, padding_mode="zeros"):
#     """
#     x:   [B, C, H, W]
#     flo: [B, 2, H, W]  （像素位移，右/下为正）
#     """
#     B, _, H, W = flo.shape
#     device = x.device
#
#     xx = torch.arange(0, W, device=device).view(1, -1).repeat(H, 1)     # [H,W]
#     yy = torch.arange(0, H, device=device).view(-1, 1).repeat(1, W)     # [H,W]
#     xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1).float()                 # [B,1,H,W]
#     yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1).float()                 # [B,1,H,W]
#     grid = torch.cat((xx, yy), dim=1)                                   # [B,2,H,W]
#
#     vgrid = grid - flo
#     vgrid = coordinate_transform(vgrid, 1.0 / scale)
#
#     vgridx = 2.0 * vgrid[:, 0:1] / max(W * scale - 1, 1) - 1.0
#     vgridy = 2.0 * vgrid[:, 1:2] / max(H * scale - 1, 1) - 1.0
#     vgrid = torch.cat([vgridx, vgridy], dim=1).permute(0, 2, 3, 1)      # [B,H,W,2]
#
#     out = softsplat.FunctionSoftsplat(tenInput=x, tenFlow=vgrid, tenMetric=None, strType='average')    # [B,1,h,w]
#     return out

import torch
import torch.nn.functional as F
# 假设已安装作者的 softsplat 扩展：pip install softsplat 或 from models import softsplat
# from softsplat import softsplat  # 也可能是 from models.softsplat import softsplat
# 这里用占位名 softsplat，与你工程里一致

def warp_back_projection_no_range(
    x: torch.Tensor,            # [B,C,H,W]
    flo: torch.Tensor,          # [B,2,h,w]  像素位移(右/下为正)；可能是 LR
    scale: float = 1.0,         # 若 flow 在 LR 上估计且 x 是 HR，填放大倍数；否则 1.0
    padding_mode: str = "zeros",# 兼容旧接口，占位
    splat_mode: str = "average",# 'summation' | 'average' | 'linear' | 'softmax'
    metric: torch.Tensor = None # [B,1,H,W]，softmax 的 logit；越大越占优
):
    """
    使用 Softsplat 做前向 warping（splatting）。
    注意：这里不再构建 [-1,1] 归一化 grid（那是 backward 的用法）。
    """
    assert x.dim() == 4 and flo.dim() == 4, "x=[B,C,H,W], flo=[B,2,h,w]"
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    # 1) flow 尺寸对齐到 x，并按需求*像素倍数*缩放
    if flo.shape[-2:] != (H, W):
        flo_up = F.interpolate(flo, size=(H, W), mode='bilinear', align_corners=True)
    else:
        flo_up = flo
    flo_up = flo_up * float(scale)              # 仍为像素单位的“前向”光流
    flo_up = flo_up.to(dtype=dtype, device=device).contiguous()

    # 2) 规范 splat_mode 到实现接受的四种写法
    alias = {
        "sum": "summation", "summation": "summation",
        "avg": "average",   "average": "average",
        "lin": "linear",    "linear": "linear",
        "soft": "softmax",  "softmax": "softmax"
    }
    key = splat_mode.lower()
    if key not in alias:
        raise ValueError(f"splat_mode 必须是 {list(alias.keys())} 之一，收到: {splat_mode}")
    strType = alias[key]   # 最终传入 softsplat 的字符串

    # 3) softmax 模式需要 metric（logit）
    tenMetric = None
    if strType == "softmax":
        if metric is None:
            tenMetric = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
        else:
            assert metric.shape == (B, 1, H, W), "metric 需要 [B,1,H,W]"
            tenMetric = metric.to(dtype=dtype, device=device).contiguous()

    # 4) 调用 Softsplat（确保 x 连续、dtype 一致）
    x = x.to(dtype=dtype, device=device).contiguous()

    # 你的工程里应为：from softsplat import softsplat
    # 这里假定已在上层导入：import softsplat
    x = x.contiguous().to("cuda")
    flo_up = flo_up.contiguous().to("cuda")
    out = softsplat.FunctionSoftsplat(
        tenInput=x,
        tenFlow=-flo_up,         # 像素单位前向 flow因为光流为正，前向需要负数
        tenMetric=tenMetric,    # None 或 [B,1,H,W]
        strType=strType         # 'summation'/'average'/'linear'/'softmax'
    )
    return out

@torch.no_grad()
def back_projection_from_HR_ref_view(
    sr_ref: torch.Tensor,               # [B, C, H, W]
    disp_base_2ch: torch.Tensor,        # [B, 2, H, W]  (dx_base, dy_base)，这里我们会让两通道都等于 disp(x,y)
    refPos=(3,3),                       # (u,v) 0-based
    angular_resolution=7,
    scale=1.0,
    padding_mode="zeros",
    flow_sign=+1.0,
):
    device = sr_ref.device
    B, C, H, W = sr_ref.shape
    U = V = angular_resolution
    UV = U * V

    ref_u, ref_v = refPos  # 注意顺序：refPos=(u,v)

    # 角度网格（u 为水平、v 为垂直）
    uu = torch.arange(U, device=device).view(1, U).repeat(V, 1)   # [V,U]
    vv = torch.arange(V, device=device).view(V, 1).repeat(1, U)   # [V,U]
    uu = uu.reshape(1, -1, 1, 1, 1).repeat(B, 1, 1, 1, 1).float() # [B,UV,1,1,1]
    vv = vv.reshape(1, -1, 1, 1, 1).repeat(B, 1, 1, 1, 1).float() # [B,UV,1,1,1]

    du = uu - float(ref_u)
    dv = vv - float(ref_v)

    # 广播位移基底到全视角
    full_disp = disp_base_2ch.unsqueeze(1).repeat(1, UV, 1, 1, 1).to(device)  # [B,UV,2,H,W]
    dx_base = full_disp[:, :, 0:1]   # [B,UV,1,H,W]
    dy_base = full_disp[:, :, 1:2]

    dx =  du * dx_base
    dy =  dv * dy_base
    full_flow = torch.cat([dx, dy], dim=2).view(-1, 2, H, W)            # [B*UV,2,H,W]

    sr_rep = sr_ref.unsqueeze(1).repeat(1, UV, 1, 1, 1).view(-1, C, H, W)
    out = warp_back_projection_no_range(sr_rep, full_flow, scale, padding_mode)   # [B*UV,C,H,W]
    return out.view(B, UV, C, H, W)   # [B,UV,C,H,W]

# ==============================
# 工具
# ==============================

def normalize_depth_to_01(depth_hw: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    d = depth_hw.astype(np.float32)
    mask = np.isfinite(d)
    if not mask.any():
        return np.zeros_like(d, dtype=np.float32)
    dmin = float(np.min(d[mask])); dmax = float(np.max(d[mask]))
    denom = (dmax - dmin)
    if abs(denom) < eps:
        return np.zeros_like(d, dtype=np.float32)
    out = np.zeros_like(d, dtype=np.float32)
    out[mask] = (d[mask] - dmin) / (denom + eps)
    out = np.clip(out, 0.0, 1.0)
    out[~mask] = 0.0
    return out

@torch.no_grad()
def save_views(folder_path: str, lf_buvchw: torch.Tensor, U: int, V: int):
    """
    lf_buvchw: [B, UV, C, H, W]，假定 B=1
    保存为 folder_path/v_u.png（从 1_1.png 到 V_U.png）
    展平顺序：idx = (v-1)*U + (u-1)（与构造一致）
    """
    os.makedirs(folder_path, exist_ok=True)
    assert lf_buvchw.shape[0] == 1, "当前保存逻辑假定 batch=1"
    _, UV, C, H, W = lf_buvchw.shape
    idx = 0
    for v in range(1, V+1):
        for u in range(1, U+1):
            img = lf_buvchw[0, idx].clamp(0,1).cpu().numpy()  # [C,H,W]
            img = (np.transpose(img, (1,2,0)) * 255.0).round().astype(np.uint8)  # [H,W,C]
            Image.fromarray(img).save(os.path.join(folder_path, f"{v}_{u}.png"))
            idx += 1

# ==============================
# 主流程
# ==============================

def main():
    ap = argparse.ArgumentParser("Center+Depth → U×V Views (no patch)")
    # 根据你的默认路径填写
    ap.add_argument('--img_dir', default=r'./new_depth/MidaS/input/test')
    ap.add_argument('--depth_dir', default=r'./new_depth/MidaS/output_demo/test')
    ap.add_argument('--output_dir', default='./warp_out_test_yanzheng/')
    ap.add_argument('--U', type=int, default=7)
    ap.add_argument('--V', type=int, default=7)
    ap.add_argument('--depth_mode', choices=['invdepth','depth','disparity','dxdy'], default='invdepth')
    ap.add_argument('--disp_scale', type=float, default=8.0, help='depth/disparity 模式的全局缩放系数；invdepth 仅作兜底')
    ap.add_argument('--ref_u', type=int, default=None, help='参考 u（0-based），默认用中心')
    ap.add_argument('--ref_v', type=int, default=None, help='参考 v（0-based），默认用中心')

    # 你的两个关键文件：
    ap.add_argument('--results_txt', type=str, default=r'D:\Test_demo\disp_results\results.txt',
                    help='包含每个 Scenes_xxx 的 disp_min/disp_max(px)')
    ap.add_argument('--mapping_json', type=str, default=r'E:\HIASR\data crop\Dataset_only_pic\test_mapping.json',
                    help='index_to_name: {"001": "Scenes_025", ...}')

    # 方向/视差分量控制
    ap.add_argument('--flow_sign', type=float, default=+1.0, help='相机右移→像素左移（-1）')
    ap.add_argument('--horizontal_only', action='store_true', help='仅合成水平位移（禁用垂直位移）')
    ap.add_argument('--use_gpu', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 图像清单
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    names = [n for n in os.listdir(args.img_dir) if os.path.splitext(n)[1].lower() in exts]
    names.sort()
    if not names:
        print('[WARN] img_dir 下没有图像。')
        return

    # 读取映射：index(001) → Scenes_xxx
    index_to_name = {}
    if args.mapping_json and os.path.isfile(args.mapping_json):
        with open(args.mapping_json, 'r', encoding='utf-8') as f:
            j = json.load(f)
            # 你的 JSON 结构示例里是 {"index_to_name": {...}}
            index_to_name = j.get("index_to_name", {})
    else:
        print(f'[WARN] 未找到 mapping_json: {args.mapping_json}')

    # 读取每个场景的 disp_max(px)
    scene2max = load_scene_to_dispmax(args.results_txt)
    if not scene2max:
        print(f'[WARN] 未能从 {args.results_txt} 解析到任何场景的 disp_max(px)，invdepth 将回退到 disp_scale。')

    device = torch.device('cuda' if (args.use_gpu and torch.cuda.is_available()) else 'cpu')
    U, V = args.U, args.V
    ref_u = args.ref_u if args.ref_u is not None else (U // 2)
    ref_v = args.ref_v if args.ref_v is not None else (V // 2)

    for i, fname in enumerate(names, start=1):
        stem = os.path.splitext(fname)[0]
        img_path = os.path.join(args.img_dir, fname)

        # 匹配深度（同名不同后缀）
        depth_path = None
        for ext in ('.pfm','.PFM','.npy','.NPY','.png','.PNG','.jpg','.JPG','.jpeg','.JPEG','.bmp','.BMP'):
            cand = os.path.join(args.depth_dir, stem + ext)
            if os.path.exists(cand):
                depth_path = cand
                break
        if depth_path is None:
            print(f'[WARN] 未找到深度：{fname}，跳过。')
            continue

        # 读取图像
        img = Image.open(img_path).convert('RGB')
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).to(device)  # [1,C,H,W]
        H, W = img_t.shape[2], img_t.shape[3]

        # 读取深度/视差
        dep = load_depth_any(depth_path)

        # 尺寸对齐到图像分辨率
        if dep.ndim == 3 and dep.shape[-1] == 1:
            dep = dep[...,0]
        if dep.ndim == 3 and dep.shape[0] == 1:
            dep = dep[0]
        if dep.shape != (H,W):
            dep = np.array(Image.fromarray(dep).resize((W,H), Image.BILINEAR))
        dep = dep.astype(np.float32)

        # 决定场景名与 disp_max(px)
        index_key = f'{i:03d}'
        scene_name = index_to_name.get(index_key, None)  # 例如 'Scenes_025'
        disp_max_px = None
        if scene_name is not None:
            disp_max_px = scene2max.get(scene_name, None)
        if disp_max_px is None:
            # 兜底：用 disp_scale
            disp_max_px = float(args.disp_scale)

        # 构造 (dx_base, dy_base)
        if args.depth_mode == 'dxdy':
            if dep.ndim == 3 and dep.shape[0] == 2:         # [2,H,W]
                dx_np, dy_np = dep[0], dep[1]
            elif dep.ndim == 3 and dep.shape[-1] >= 2:      # [H,W,2]
                dx_np, dy_np = dep[...,0], dep[...,1]
            else:
                raise ValueError(f'dxdy 模式需要 2 通道位移，得到形状 {dep.shape}')
            dx = torch.from_numpy(dx_np.astype(np.float32)).to(device)
            dy = torch.from_numpy(dy_np.astype(np.float32)).to(device)
            disp_base_2ch = torch.stack([dx, dy], dim=0).unsqueeze(0)       # [1,2,H,W]

        elif args.depth_mode == 'invdepth':
            # 先归一化到 [0,1]，再乘“该场景的 disp_max(px)”
            norm01 = normalize_depth_to_01(dep, eps=1e-6)                   # [H,W]
            disp_np = norm01 * float(disp_max_px)                            # 关键改动
            disp_t = torch.from_numpy(disp_np).to(device)                    # [H,W]
            disp_base_2ch = torch.stack([disp_t, disp_t], dim=0).unsqueeze(0)

        elif args.depth_mode == 'depth':
            # disp = (1/Z)*scale（简单兜底公式，可按相机几何替换）
            disp_np = np.zeros_like(dep, dtype=np.float32)
            mask = dep > 0
            disp_np[mask] = (1.0 / dep[mask]) * float(args.disp_scale)
            disp_t = torch.from_numpy(disp_np).to(device)
            disp_base_2ch = torch.stack([disp_t, disp_t], dim=0).unsqueeze(0)

        else:  # disparity
            disp_np = dep * float(args.disp_scale)
            disp_t = torch.from_numpy(disp_np).to(device)
            disp_base_2ch = torch.stack([disp_t, disp_t], dim=0).unsqueeze(0)

        # 反投影合成
        lf = back_projection_from_HR_ref_view(
            sr_ref=img_t,
            disp_base_2ch=disp_base_2ch,
            refPos=(ref_u, ref_v),
            angular_resolution=U,
            scale=1.0,
            padding_mode="zeros",
            flow_sign=args.flow_sign,

        )  # [1, UV, C, H, W]

        # 保存
        out_folder = os.path.join(args.output_dir, f"{i:03d}")
        save_views(out_folder, lf, U, V)
        print(f"[OK] {index_key} ({scene_name or 'UNKNOWN'}) : {fname} → {U}x{V} 视图已保存到 {out_folder}，disp_max={disp_max_px:.6f}px")

if __name__ == '__main__':
    main()
