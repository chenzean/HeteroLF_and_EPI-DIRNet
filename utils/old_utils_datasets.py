# -*- coding: utf-8 -*-
# @Time    : 2025-09-10 12:18
# @Author  : Chen Zean
# @Site    : 
# @File    : old_utils_datasets.py
# @Software: PyCharm

def _list_h5_files(dir_path: str):
    """返回 dir_path 下按文件名排序的 .h5/.hdf5 绝对路径列表"""
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"目录不存在：{os.path.abspath(dir_path)}")
    files = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f)) and f.lower().endswith((".h5", ".hdf5"))
    ]
    files.sort()  # 用文件名字典序保证稳定配对
    return files

class TrainSetDataLoader(Dataset):
    """
    目录结构：
      ./Dataset/train_data/
        ├─ center_view_and_label/   （含 data/label 的 H5）
        └─ LF/                      （含 LF 或 LF_SAI 等 key 的 H5）

    配对逻辑：以“同名基名”匹配，例如 A.h5 在两侧都存在才配成一对。
    """
    def __init__(self, args):
        super().__init__()
        root = args.path_for_train

        self.dataset_dir = os.path.join(root, "center_view_and_label")
        self.ref_dataset_dir = os.path.join(root, "LF")

        self.data_files = _list_h5_files(self.dataset_dir)
        self.ref_files = _list_h5_files(self.ref_dataset_dir)

        if len(self.data_files) != len(self.ref_files):
            raise RuntimeError(
                f"[Train] Mismatched datasets: the number of files is different.\n"
                f"center_view_and_label: {len(self.data_files)} files\n"
                f"LF: {len(self.ref_files)} files\n"
                f"Paths:\n  {self.dataset_dir}\n  {self.ref_dataset_dir}"
            )
        if len(self.data_files) == 0:
            raise RuntimeError(
                f"[Train] No .h5 files found in both directories.\n"
                f"Paths:\n  {self.dataset_dir}\n  {self.ref_dataset_dir}"
            )

        self.item_num = len(self.data_files)

        print(f"[Train] Loaded {self.item_num} pairs...")

    def __len__(self):
        return self.item_num

    def __getitem__(self, index):
        f_data = self.data_files[index]
        f_ref  = self.ref_files[index]

        # get data
        with h5py.File(f_data, "r") as hf1, h5py.File(f_ref, "r") as hf2:
            data = hf1["data"][()]          # Uint8 type
            label = hf1["label"][()]        # Uint8 type
            LF = hf2["LF"][()]              # Uint8 type

            # turn to float32 and normalize to [0,1]
            data = data.astype(np.float32) / 255.0
            label = label.astype(np.float32) / 255.0
            LF = LF.astype(np.float32) / 255.0

        # reshape to [C, H, W]
        data = np.transpose(data, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))
        LF = np.transpose(LF, (2, 0, 1))

        # data augmentation
        data, label, LF = augmentation(data, label, LF)

        return data.copy(), label.copy(), LF.copy()



class TestSetDataLoader(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        root = args.path_for_test

        self.dataset_dir = os.path.join(root, "center_view_and_label")
        self.ref_dataset_dir = os.path.join(root, "LF")

        self.data_paths = _list_h5_files(self.dataset_dir)
        self.ref_paths = _list_h5_files(self.ref_dataset_dir)

        if len(self.data_paths) != len(self.ref_paths):
            raise RuntimeError(
                f"[Test] Mismatched datasets: the number of files is different.\n"
                f"SR/data+label folder: {len(self.data_paths)} files\n"
                f"Ref/LF folder: {len(self.ref_paths)} files\n"
                f"Paths:\n  {self.dataset_dir}\n  {self.ref_dataset_dir}"
            )
        if len(self.data_paths) == 0:
            raise RuntimeError(
                f"[Test] No .h5 files found in both directories.\n"
                f"Paths:\n  {self.dataset_dir}\n  {self.ref_dataset_dir}"
            )

        self.item_num = len(self.data_paths)
        print(f"[Test] Loaded {self.item_num} pairs (sorted by filename).")


    def __len__(self):
        return self.item_num

    def __getitem__(self, index):
        file_name = self.data_paths[index]
        ref_file_name = self.ref_paths[index]

        # get data
        with h5py.File(file_name, 'r') as hf1, h5py.File(ref_file_name, 'r') as hf2:

            data = hf1["data"][()]
            label = hf1["label"][()]
            LF = hf2["LF"][()]

        # turn to float32 and normalize to [0,1]
        data = data.astype(np.float32) / 255.0
        label = label.astype(np.float32) / 255.0
        LF = LF.astype(np.float32) / 255.0

        # reshape to [C, H, W]
        data = np.transpose(data, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))
        LF = np.transpose(LF, (2, 0, 1))

        LF_name = os.path.splitext(os.path.basename(file_name))[0]

        return data, label, LF, LF_name



class ValSetDataLoader(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        root = args.path_for_val

        self.dataset_dir = os.path.join(root, "center_view_and_label")
        self.ref_dataset_dir = os.path.join(root, "LF")

        self.data_paths = _list_h5_files(self.dataset_dir)
        self.ref_paths = _list_h5_files(self.ref_dataset_dir)


        if len(self.data_paths) != len(self.ref_paths):
            raise RuntimeError(
                f"[Test] Mismatched datasets: the number of files is different.\n"
                f"SR/data+label folder: {len(self.data_paths)} files\n"
                f"Ref/LF folder: {len(self.ref_paths)} files\n"
                f"Paths:\n  {self.dataset_dir}\n  {self.ref_dataset_dir}"
            )
        if len(self.data_paths) == 0:
            raise RuntimeError(
                f"[Test] No .h5 files found in both directories.\n"
                f"Paths:\n  {self.dataset_dir}\n  {self.ref_dataset_dir}"
            )

        self.item_num = len(self.data_paths)
        print(f"[Test] Loaded {self.item_num} pairs...")


    def __len__(self):
        return self.item_num

    def __getitem__(self, index):
        file_name = self.data_paths[index]
        ref_file_name = self.ref_paths[index]

        # get data
        with h5py.File(file_name, 'r') as hf1, h5py.File(ref_file_name, 'r') as hf2:

            data = hf1["data"][()]
            label = hf1["label"][()]
            LF = hf2["LF"][()]

        # turn to float32 and normalize to [0,1]
        data = data.astype(np.float32) / 255.0
        label = label.astype(np.float32) / 255.0
        LF = LF.astype(np.float32) / 255.0

        # reshape to [C, H, W]
        data = np.transpose(data, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))
        LF = np.transpose(LF, (2, 0, 1))

        LF_name = os.path.splitext(os.path.basename(file_name))[0]

        return data, label, LF, LF_name


def augmentation(data, label, ref):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:,:, ::-1]
        label = label[:,:, ::-1]
        ref = ref[:, :, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:,::-1, :]
        label = label[:,::-1, :]
        ref = ref[:, ::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(0, 2, 1)
        label = label.transpose(0, 2, 1)
        ref = ref.transpose(0, 2, 1)
    if random.random() < 0.5:  # shuffle the order of RGB channels
        order = [0, 1, 2]
        random.shuffle(order)
        data = data[order, :, :]
        label = label[order, :, :]
        ref = ref[order, :, :]

    return data, label, ref


if __name__ == '__main__':
    from config import args

    train_dataset = TrainSetDataLoader(args)
    test_dataset = TestSetDataLoader(args)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=args.num_workers,
                                               batch_size=args.batch_size, shuffle=True,
                                               pin_memory=True, persistent_workers=args.num_workers > 0, )
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=args.num_workers,
                                               batch_size=args.batch_size, shuffle=True,
                                               pin_memory=True, persistent_workers=args.num_workers > 0, )

    # for i, (data, label, LF) in enumerate(train_loader):
    #     print("train_loader", i, data.shape, label.shape, LF.shape)

    for i, (data, label, LF, name) in enumerate(test_loader):
        print("test_loader",i, data.shape, label.shape, LF.shape, name)
