import os
import argparse

parser = argparse.ArgumentParser(description='Heterogeneous Light Field Image Angular Super-Resolution')

# ------------------------ Model Config ------------------------ #
parser.add_argument('--model_name', type=str, default='proposed_v4', help="Network name")
parser.add_argument('--channels', type=int, default=32, help="Feature channels or transformer embed_dim")

# ---------------------- Task Settings ------------------------ #
parser.add_argument('--angRes', type=int, default=7, help="Angular resolution")

# ---------------------- Training Strategy --------------------- #
parser.add_argument('--epoch', type=int, default=500, help='Total number of training epochs')
parser.add_argument('--lr', type=float, default=2e-4, help='Initial learning rate')
parser.add_argument('--decay_rate', type=float, default=0, help='Weight decay for optimizer')
parser.add_argument('--n_steps', type=int, default=150, help='Epochs between LR decay steps')
parser.add_argument('--gamma', type=float, default=0.5, help='LR decay gamma')
parser.add_argument('--accum_steps', type=int, default=1, help='Gradient accumulation steps')
parser.add_argument('--use_amp', default=False, help='Use mixed precision (AMP)')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
parser.add_argument('--eval_interval', type=int, default=25, help='Evaluate model during training')
parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
parser.add_argument('--repeat_factor', type=int, default=10, help='每个样本重复采样次数')
parser.add_argument('--patch_size', type=int, default=64, help='随机裁剪 patch 大小')

# ---------------------- Checkpoint I/O ------------------------ #
parser.add_argument('--use_pre_ckpt', type=bool, default=True, help='Whether to load pre-trained checkpoint')
parser.add_argument('--path_pre_pth', type=str, default='./pretrain/best_model.pth',
                    help='Path to pretrained .pth model')
parser.add_argument('--path_log', type=str, default='./log/', help='Logging directory')
parser.add_argument('--path_loss', type=str, default='./', help='Path to save loss plot (Deprecated)')

# ---------------------- Validation/Test ----------------------- #
parser.add_argument('--val_interval', type=int, default=1, help='Validate every N epochs during training')
parser.add_argument('--patch_size_for_test', type=int, default=64, help='Test patch size')
parser.add_argument('--stride_for_test', type=int, default=32, help='Test stride')
parser.add_argument('--minibatch_for_test', type=int, default=1, help='Minibatch size for testing')

# ---------------------- Dataset Path -------------------------- #
parser.add_argument('--data_name', type=str, default='ALL', help='Dataset name identifier')
parser.add_argument('--path_for_train', type=str, default='./Dataset_full_LF/train_data_full_image/', help='Training set path')
parser.add_argument('--path_for_val', type=str, default='./Dataset_full_LF/val_data/', help='Validation set path')
parser.add_argument('--path_for_test', type=str, default='./Dataset_full_LF/test_data/', help='Test set path')

# Just for test code debugging
# parser.add_argument('--path_for_train', type=str, default='./Dataset_demo/train_data/', help='Training set path')
# parser.add_argument('--path_for_val', type=str, default='./Dataset_demo/val_data/', help='Validation set path')
# parser.add_argument('--path_for_test', type=str, default='./Dataset_demo/test_data/', help='Test set path')

# ---------------------- Runtime Environment ------------------- #
parser.add_argument('--device', type=str, default='cuda:0', help='Training device')
parser.add_argument('--num_workers', type=int, default=2, help='Dataloader worker threads')
parser.add_argument('--distributed', default=False, help='Use distributed training')
parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0), help='Local rank for distributed')
parser.add_argument('--use_wandb', type=bool, default=False, help='Use Weights & Biases for logging')

# ---------------------- Visualization & Debug ----------------- #
parser.add_argument('--debug', default=False, help='Enable debug mode with image visualization')
# todo
parser.add_argument('--plot_interval', type=int, default=100, help='Plot loss curve every N batches (Deprecated)')
parser.add_argument('--save_output', type=bool, default=True, help='Save output images during validation/testing')

# ---------------------- Loss Function Weights ----------------- #
parser.add_argument('--w_lpips', type=float, default=0, help='Weight for LPIPS loss')
parser.add_argument('--w_rec', type=float, default=1.0, help='Weight for reconstruction loss')
parser.add_argument('--w_detail', type=float, default=1.0, help='Weight for detail-enhancing loss')

args = parser.parse_args()