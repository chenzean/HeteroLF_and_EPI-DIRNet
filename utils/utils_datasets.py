# -*- coding: utf-8 -*-
# @Time    : 2025-09-10 17:41
# @Author  : Chen Zean
# @File    : utils_datasets.py

import os, re, logging, random
from typing import List, Tuple, Optional
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from einops import rearrange

__all__ = [
    "TrainSetDataLoader",
    "TestSetDataLoader",
    "ValSetDataLoader",
]



# -----------------------------
# Utilities
# -----------------------------
_NUM_RE = re.compile(r"(\d+)")

def _natsort_key(p: str):
    """Natural sort key: '2.h5' < '10.h5'."""
    name = os.path.basename(p)
    return [int(m) if m.isdigit() else m.lower() for m in _NUM_RE.split(name)]

def _list_h5_files(dir_path: str) -> List[str]:
    """List absolute paths of .h5/.hdf5 files in dir_path (natural sorted)."""
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory does not exist: {os.path.abspath(dir_path)}")
    files = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f)) and f.lower().endswith((".h5", ".hdf5"))
    ]
    files.sort(key=_natsort_key)
    return files

def _to_chw_float01(arr: np.ndarray) -> np.ndarray:
    """
    Convert an array to CHW float32 in [0,1].
    Accepts HxW (grayscale), HxWxC, or already CHW. Keeps C in {1,3} if possible.
    """
    if arr.ndim == 2:               # H W -> H W 1
        arr = arr[:, :, None]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        # Looks like CHW already (C,H,W). Keep as-is.
        chw = arr
    elif arr.ndim == 3:             # H W C -> C H W
        chw = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported array shape for image: {arr.shape}")

    # dtype-safe normalization
    if chw.dtype == np.uint8:
        chw = chw.astype(np.float32) / 255.0
    elif chw.dtype == np.uint16:
        chw = chw.astype(np.float32) / 65535.0
    else:
        chw = chw.astype(np.float32, copy=False)
        # If values look like 0..255, scale; if already 0..1, keep
        vmax = float(np.nanmax(chw)) if np.isfinite(chw).any() else 1.0
        if vmax > 1.5:  # heuristic
            chw = np.clip(chw / 255.0, 0.0, 1.0)
        else:
            chw = np.clip(chw, 0.0, 1.0)

    # return np.ascontiguousarray(chw)
    return chw

def _read_h5_keys(file_path: str, k1: str, k2: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read one or two datasets from an H5 file and return numpy arrays."""
    with h5py.File(file_path, "r") as hf:
        if k1 not in hf:
            raise KeyError(f"{file_path} must contain key '{k1}'; got keys={list(hf.keys())}")
        a1 = hf[k1][()]
        a2 = None
        if k2:
            if k2 not in hf:
                raise KeyError(f"{file_path} must contain key '{k2}'; got keys={list(hf.keys())}")
            a2 = hf[k2][()]
    return a1, a2

# -----------------------------
# Augmentation (train only)
# -----------------------------
def augmentation_chw(data: np.ndarray, label: np.ndarray, ref: np.ndarray,
                     rng: Optional[random.Random] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic-friendly geometric augmentation on CHW arrays.
    - Horizontal flip (W), vertical flip (H), 90° rotation, applied consistently to data/label/ref.
    """
    r = rng or random
    # flip W
    if r.random() < 0.5:
        data = data[:, :, ::-1]
        label = label[:, :, ::-1]
        ref = ref[:, :, ::-1]
    # flip H
    if r.random() < 0.5:
        data = data[:, ::-1, :]
        label = label[:, ::-1, :]
        ref = ref[:, ::-1, :]

    # transpose H<->W (i.e., 90° rotate w.r.t. square images)
    if r.random() < 0.5:
        data = data.transpose(0, 2, 1)
        label = label.transpose(0, 2, 1)
        ref = ref.transpose(0, 2, 1)

    return data, label, ref

# -----------------------------
# Base dataset
# -----------------------------
class _PairedH5Dataset(Dataset):
    """
    Base paired dataset:
      left dir contains 'data' & 'label' (SR inputs and GT)
      right dir contains 'LF'          (reference/light field)

    Files are paired by natural filename order.
    """
    def __init__(self, left_dir: str, right_dir: str, split: str,
                 do_augment: bool = False, seed: Optional[int] = None):
        super().__init__()
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.split = split
        self.do_augment = do_augment
        self.rng = random.Random(seed) if seed is not None else None

        self.left_files = _list_h5_files(self.left_dir)
        self.right_files = _list_h5_files(self.right_dir)

        if len(self.left_files) != len(self.right_files):
            raise RuntimeError(
                f"[{split}] Mismatched datasets: the number of files is different.\n"
                f"left (data+label): {len(self.left_files)} files\n"
                f"right (LF): {len(self.right_files)} files\n"
                f"Paths:\n  {self.left_dir}\n  {self.right_dir}"
            )
        if len(self.left_files) == 0:
            raise RuntimeError(
                f"[{split}] No .h5 files found in both directories.\n"
                f"Paths:\n  {self.left_dir}\n  {self.right_dir}"
            )

        self.item_num = len(self.left_files)
        # logging.info(f"[{split}] Loaded {self.item_num} pairs (natural sorted).")

    def __len__(self):
        return self.item_num

    def _load_triplet(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        f_left  = self.left_files[idx]
        f_right = self.right_files[idx]

        # Read arrays
        data_np, label_np = _read_h5_keys(f_left,  "data",  "label")
        ref_np, _         = _read_h5_keys(f_right, "LF",    None)

        # HWC/CHW robust conversion
        data_chw  = _to_chw_float01(data_np)
        label_chw = _to_chw_float01(label_np)
        ref_chw   = _to_chw_float01(ref_np)

        # Augment (train only)
        if self.do_augment:
            data_chw, label_chw, ref_chw = augmentation_chw(data_chw, label_chw, ref_chw, self.rng)

        name = os.path.splitext(os.path.basename(f_left))[0]
        return data_chw, label_chw, ref_chw, name

# -----------------------------
# Training Dataset with Repeat & View-wise Random Crop (SAI-safe)
# -----------------------------
class TrainSetDataLoader(_PairedH5Dataset):
    def __init__(self, args):
        root = args.path_for_train
        self.args = args                                # 保存以便 __getitem__ 使用
        self.ang = int(getattr(args, "angRes", 7))      # 角度分辨率 a×a
        self.repeat_factor = int(getattr(args, "repeat_factor", 10))
        # 这里的 patch_size 指的是**单视图（子孔径）上的裁剪尺寸**，不是 SAI 总图的尺寸
        self.patch_size = int(getattr(args, "patch_size", 64))

        # 重要：为避免在 SAI 平面做增强导致角度网格错位，这里关闭 _PairedH5Dataset 的增强
        super().__init__(
            left_dir=os.path.join(root, "warping_LF_and_label/"),
            right_dir=os.path.join(root, "LF/"),
            split="Train",
            do_augment=True,  # SAI 情况下建议在视图域做增强，如需可在下方对 views_* 做
            seed=getattr(args, "seed", None),
        )

    def __len__(self):
        return self.item_num * self.repeat_factor

    def __getitem__(self, index):
        # 计算真实索引
        true_index = index % self.item_num
        # 取出 SAI：形状为 [C, (ang*H), (ang*W)]，数值范围 [0,1]
        data_sai, label_sai, ref_sai, _ = self._load_triplet(true_index)

        a = self.ang
        C, AH, AW = data_sai.shape
        assert AH % a == 0 and AW % a == 0, \
            f"SAI 高宽必须是 angRes 的整数倍，但得到 {(AH, AW)} 与 angRes={a}"
        H, W = AH // a, AW // a

        # 1) 展开角度维到“内部 batch”： [C, a*H, a*W] -> [a*a, C, H, W]
        data_views  = rearrange(data_sai,  'c (a h) (b w) -> (a b) c h w', a=a, b=a)
        label_views = rearrange(label_sai, 'c (a h) (b w) -> (a b) c h w', a=a, b=a)
        ref_views   = rearrange(ref_sai,   'c (a h) (b w) -> (a b) c h w', a=a, b=a)

        # 2) 在单视图平面做一致的随机裁剪（同一 (top,left) 应用于所有视图）
        ph = pw = self.patch_size
        if H < ph or W < pw:
            raise ValueError(f"单视图尺寸 HxW=({H},{W}) 小于裁剪大小 {ph}x{pw}，请减小 patch_size 或使用更大图像")

        top  = random.randint(0, H - ph)
        left = random.randint(0, W - pw)

        data_views  = data_views[:,  :, top:top+ph, left:left+pw]
        label_views = label_views[:, :, top:top+ph, left:left+pw]
        ref_views   = ref_views[:,   :, top:top+ph, left:left+pw]


        # 3) 拼回 SAI： [a*a, C, ph, pw] -> [C, (a*ph), (a*pw)]
        data_sai_crop  = rearrange(data_views,  '(a b) c h w -> c (a h) (b w)', a=a, b=a)
        label_sai_crop = rearrange(label_views, '(a b) c h w -> c (a h) (b w)', a=a, b=a)
        ref_sai_crop   = rearrange(ref_views,   '(a b) c h w -> c (a h) (b w)', a=a, b=a)
        # print(data_sai_crop.shape, label_sai_crop.shape, ref_sai_crop.shape)
        # print(data_sai_crop.device, label_sai_crop.device, ref_sai_crop.device)

        return data_sai_crop.copy(), label_sai_crop.copy(), ref_sai_crop.copy()




class TestSetDataLoader(_PairedH5Dataset):
    def __init__(self, args):
        root = args.path_for_test
        super().__init__(
            left_dir=os.path.join(root, "warping_LF_and_label/"),
            right_dir=os.path.join(root, "LF/"),
            split="Test",
            do_augment=False,
        )

    def __getitem__(self, index):
        data, label, ref, name = self._load_triplet(index)
        return data, label, ref, name

class ValSetDataLoader(_PairedH5Dataset):
    def __init__(self, args):
        root = args.path_for_val
        super().__init__(
            left_dir=os.path.join(root, "warping_LF_and_label/"),
            right_dir=os.path.join(root, "LF/"),
            split="Val",
            do_augment=False,
        )

    def __getitem__(self, index):
        data, label, ref, name = self._load_triplet(index)
        return data, label, ref, name

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    from config import args

    train_dataset = TrainSetDataLoader(args)
    val_dataset   = ValSetDataLoader(args)
    test_dataset  = TestSetDataLoader(args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # Note: eval loaders should not shuffle
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    for i, (data, label, ref) in enumerate(train_loader):
        print("train:", i, data.shape, label.shape, ref.shape)
        break

    for i, (data, label, ref, name) in enumerate(test_loader):
        print("test:", i, data.shape, label.shape, ref.shape, name)
        break