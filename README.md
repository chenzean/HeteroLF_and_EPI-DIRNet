#  HeteroLF & EPI-DIRNet

> ✅ **Note:** The **complete code**, **trained weights**, **dataset**, and the **results from various methods** are now all available. See [Usage](#-usage) for training/testing commands and [Download](#-download) for the dataset and weights.
>
>  Thank you for your patience and interest!
>
> If HeteroLF & EPI-DIRNet is helpful for you, please help star the GitHub Repo. Thanks!



## 🚩 **New Features/Updates**

- ✅ July 21, 2026. Release the complete code (training & testing) and the trained weights
- ✅ July 21, 2026. Release the HeteroLF Dataset and the results from various methods
- ✅ December 30, 2025. Release the testing code
- ✅ December 10, 2025. Create the repository



## 📚 Dataset Generation

![Dataset Generation](/assets/Dataset_Generation.png)

## 🛠️ Usage

All hyper-parameters are defined in [`config.py`](config.py) and can be overridden from the command line.

### Data Preparation

Download the dataset and trained weights from the [Download](#-download) link, then organize them as follows:

```
HeteroLF_and_EPI-DIRNet
├── Dataset_full_LF
│   ├── train_data_full_image
│   ├── val_data
│   └── test_data
└── pretrain
    └── best_model.pth
```

### Training

```bash
python train.py \
    --model_name proposed_v4 \
    --angRes 7 \
    --batch_size 1 \
    --epoch 500 \
    --lr 2e-4 \
    --path_for_train ./Dataset_full_LF/train_data_full_image/ \
    --path_for_val ./Dataset_full_LF/val_data/ \
    --device cuda:0
```

To train from scratch (without loading a checkpoint), add `--use_pre_ckpt ''`. To resume from / fine-tune a checkpoint, point `--path_pre_pth` to the `.pth` file. Checkpoints and logs are saved under `./log/`.

### Testing

```bash
python test.py \
    --model_name proposed_v4 \
    --angRes 7 \
    --path_pre_pth ./pretrain/best_model.pth \
    --path_for_test ./Dataset_full_LF/test_data/ \
    --device cuda:0
```

Quantitative results (PSNR/SSIM) are written to `evaluation.xls` / `evaluation_allviews.xls`, and the reconstructed views are saved under the result directory when `--save_output True`.

## 📥 Download

The dataset and the results from various methods are available at the following link:

- **Baidu Netdisk:** [https://pan.baidu.com/s/1mll65kvDU5DWglyRkp06qA](https://pan.baidu.com/s/1mll65kvDU5DWglyRkp06qA) (Extraction code: `w93a`)

## 📬 Contact

If you have any questions regarding this work, feel free to reach out:

**✉️ Email:** [chenzean2024@126.com] or 18358481590 (WeChat)



## ⭐ Acknowledgements

Our project is based on [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR), [NeRCo](https://github.com/Ysz2022/NeRCo),[NAFNet](https://github.com/megvii-research/NAFNet/tree/main) and [ASTv2](https://github.com/joshyZhou/ASTv2). Thanks for their awesome works.

Thanks to my lab partners and senior students for helping me check the code.