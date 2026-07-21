import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
import importlib
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.utils_datasets import TrainSetDataLoader, ValSetDataLoader
import imageio
from scipy.io import savemat
import numpy as np
import random
# import wandb
import json
import matplotlib.pyplot as plt
import math
from torch.cuda.amp import GradScaler
import time
import gc
import matplotlib.cm as cm

# todo: add 多卡训练 代码


# add AMP
def main(args):
    ''' Create Dir for Save'''
    log_dir, val_dir, checkpoints_dir = create_dir(args)

    ''' Logger '''
    logger = Logger(log_dir, args)

    ''' CPU or Cuda'''
    device = torch.device(args.device)

    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    ''' DATA Training LOADING '''
    logger.log_string('\nLoad Training Dataset ...')
    train_Dataset = TrainSetDataLoader(args)
    logger.log_string("The number of training data is: %d" % len(train_Dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_Dataset, num_workers=args.num_workers,
                                               batch_size=args.batch_size, shuffle=True,
                                               pin_memory=False, persistent_workers=args.num_workers > 0,)


    ''' DATA Validation LOADING '''
    logger.log_string('\nLoad Validation Dataset ...')
    val_Dataset = ValSetDataLoader(args)
    logger.log_string("The number of validation data is: %d"  % len(val_Dataset))
    val_loader = torch.utils.data.DataLoader(dataset=val_Dataset, num_workers=args.num_workers,
                                             batch_size=1, shuffle=False,)

    '''seed setup'''
    setup_seed(args.seed)

    ''' MODEL LOADING '''
    MODEL_PATH = 'model.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args).to(device)

    '''Optimizers'''
    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay_rate
    )

    '''Scheduler'''
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)

    '''Load Pre-Trained PTH'''
    start_epoch = 0
    best_psnr = 0.0

    if args.use_pre_ckpt:
        try:
            checkpoint = torch.load(args.path_pre_pth, map_location='cuda:0')
            start_epoch = checkpoint.get('epoch', 0)
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            best_psnr = checkpoint.get('best_psnr', 0.0)
            logger.log_string('Pretrained weights loaded successfully!')
        except Exception as e:
            logger.log_string(f'[Error loading pretrain]: {str(e)}')
            net.apply(MODEL.weights_init)
    else:
        logger.log_string('Do not use pre-trained model!')
        net.apply(MODEL.weights_init)

    '''AMP'''
    scaler = GradScaler(enabled=args.use_amp)

    logger.log_string('PARAMETER ...')
    logger.log_string(args)
    total_params = sum(p.numel() for p in net.parameters()) / 1e6
    logger.log_string(f" - Net Parameters   : {total_params:.4f}M")

    if hasattr(MODEL, 'get_loss'):
        criterion = MODEL.get_loss(args).to(device)
    else:
        logger.log_string("[Warning] Model missing get_loss(). Using default L1.")
        criterion = torch.nn.L1Loss().to(device)


    ''' WandB Initialization '''
    if args.use_wandb:
        wandb.init(project=args.project_name, name=args.run_name, config=vars(args))
        wandb.config.update(args)

    # Save args to json for reproducibility
    with open(log_dir / 'train_args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    ''' TRAINING & TEST '''
    for idx_epoch in range(start_epoch, args.epoch):
        logger.log_string(f'\n Epoch {idx_epoch + 1}')
        print(f"Epoch: {idx_epoch + 1}, Learning Rate: {optimizer.param_groups[0]['lr']}")

        loss_epoch_train, psnr_epoch_train, ssim_epoch_train = train(
            train_loader, device, net, criterion, optimizer, logger, args, scaler, accum_steps=args.accum_steps, debug=args.debug)

        logger.log_string(
            f'Train Epoch {idx_epoch + 1}: loss={loss_epoch_train:.5f}, psnr={psnr_epoch_train:.5f}, ssim={ssim_epoch_train:.5f}')

        if args.use_wandb:
            wandb.log({
                "train/loss": loss_epoch_train,
                "train/psnr": psnr_epoch_train,
                "train/ssim": ssim_epoch_train,
                "lr": optimizer.param_groups[0]['lr'],
                "epoch": idx_epoch + 1,
            })

        scheduler.step()

        if args.local_rank == 0:
            save_ckpt_path = str(
                checkpoints_dir / f'{args.model_name}_{args.angRes}x{args.angRes}_epoch_{idx_epoch + 1:02d}_model.pth')
            state = {
                'epoch': idx_epoch + 1,
                'state_dict': net.module.state_dict() if hasattr(net, 'module') else net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_psnr': best_psnr,
            }
            torch.save(state, save_ckpt_path)
            logger.log_string(f'Saving the epoch_{idx_epoch + 1:02d} model at {save_ckpt_path}')

            torch.cuda.empty_cache()

            '''Validation'''
            if (idx_epoch + 1) % args.eval_interval == 0:
                logger.log_string(f'\nValidation at Epoch {idx_epoch + 1}')
                val_start_time = time.time()

                logger.log_string(f'Running on validation set...')
                psnr_all, ssim_all, _, _, _, _ = test(val_loader, device, net, args)
                avg_psnr, avg_ssim = np.mean(psnr_all), np.mean(ssim_all)
                logger.log_string(f'  Avg PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f}')

                val_duration = time.time() - val_start_time
                logger.log_string(f'[Validation Time] {val_duration:.2f} seconds')


                torch.cuda.empty_cache()


                '''save best model'''
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_ckpt_path = checkpoints_dir / 'best_model.pth'
                    state = {
                        'epoch': idx_epoch + 1,
                        'state_dict': net.module.state_dict() if hasattr(net, 'module') else net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_psnr': best_psnr,
                    }
                    torch.save(state, best_ckpt_path)
                    logger.log_string(
                        f'Best model updated at epoch {idx_epoch + 1} with Validation PSNR: {best_psnr:.4f}')

                    # === WandB Artifact ===
                    if args.use_wandb:
                        artifact = wandb.Artifact(f'{args.model_name}_best', type='model')
                        artifact.add_file(str(best_ckpt_path))
                        wandb.log_artifact(artifact)
                        wandb.run.summary['best_psnr'] = best_psnr
                        wandb.run.summary['best_epoch'] = idx_epoch + 1
    if args.use_wandb:
        wandb.finish()



'''
New code version 2 !!!

Version 1 adds:
mixed precision training, 

'''

def train(train_loader, device, net,  criterion, optimizer, logger, args,
          scaler,
          accum_steps,
          debug=False,
          vis_freq=1,
          max_grad_norm=1.,
          show_mem=False):

    loss_list, psnr_list, ssim_list = [], [], []
    dev_id = torch.cuda.current_device() if show_mem else None

    # optimizer.zero_grad(set_to_none=True)
    total_update = math.ceil(len(train_loader) / accum_steps)
    pbar = tqdm(total=total_update, ncols=70, desc='Train')

    for idx_iter, (data, label, LF) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)
        LF = LF.to(device)

        # visualize the data
        if debug and (idx_iter % vis_freq == 0):
            def show_tensor(tensor, title):
                img = tensor[0].float().detach().cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                plt.imshow(img)
                plt.title(title)
                plt.axis('off')
                plt.show()

            show_tensor(data, f'data_{idx_iter}')
            print(f"Data Info (data): {data.shape}")

            show_tensor(LF, f'ref_{idx_iter}')
            print(f"Data Info (ref): {LF.shape}")

            show_tensor(label, f'label_{idx_iter}')
            print(f"Data Info (label): {label.shape}")

        net.train()

        rec, output = net(data, LF)
        loss = (criterion(output,rec, label) / accum_steps)
        # output = net(data, LF)
        # loss = (criterion(output, label) / accum_steps)

        if not torch.isfinite(loss):
            print(f"[Error] Loss became invalid (NaN or Inf): {loss.item()}")
            raise RuntimeError("Loss is NaN or Inf. Training aborted.")

        scaler.scale(loss).backward()

        loss_list.append(loss.item() * accum_steps)

        # Check gradients (debug mode)
        if debug:
            for name, p in net.named_parameters():
                if p.grad is None:  # 没有梯度
                    print(f'{name:30s} │ grad = None')
                else:
                    # 统计信息：均值 / 绝对最大值 / L2 范数
                    grad_mean = p.grad.mean().item()
                    grad_max = p.grad.abs().max().item()
                    grad_norm = p.grad.norm().item()  # L2
                    print(f'{name:30s} │ μ={grad_mean:+.3e} │ |g|_∞={grad_max:.3e} │ ‖g‖₂={grad_norm:.3e}')

        is_update = (idx_iter + 1) % accum_steps == 0 \
                    or (idx_iter + 1) == len(train_loader)

        # ========== 梯度更新 ==========
        if is_update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                net.parameters(),
                max_norm=max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            pbar.update(1)

            if show_mem:
                torch.cuda.synchronize()
                mem_MB = torch.cuda.max_memory_allocated(dev_id) / 1024 ** 2
                print(f"GPU {dev_id} Peak Mem: {mem_MB:.2f} MiB")

        # ---------- 指标 ----------
        psnr, ssim = cal_metrics(args, label, output.detach())
        psnr_list.append(psnr.mean().item())
        ssim_list.append(ssim.mean().item())


    pbar.close()
    return float(np.mean(loss_list)), float(np.mean(psnr_list)), float(np.mean(ssim_list))


def test(test_loader, device, net, args, epoch_dir=None, CenterView_dir=None, mat_file=None, aligned_image_dir=None):
    LF_iter_test = []
    psnr_iter_test = []
    ssim_iter_test = []
    psnr_iter_test_allviews = []
    ssim_iter_test_allviews = []
    single_sence_time_avg = []

    net = net.to(device)

    for idx_iter, (data, label, LF, LF_name) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):

        data = data.squeeze()
        LF = LF.squeeze()

        # crop the patch
        subLFin = LFdivide(data, args.angRes, args.patch_size_for_test, args.stride_for_test)
        subRef_LFin = LFdivide(LF, args.angRes, args.patch_size_for_test, args.stride_for_test)
        numU, numV, c, H, W = subLFin.size()
        subLFin = rearrange(subLFin, 'n1 n2 c a1h a2w -> (n1 n2) c a1h a2w')
        subRef_LFin = rearrange(subRef_LFin, 'n1 n2 c a1h a2w -> (n1 n2) c a1h a2w')

        subOut = torch.zeros_like(subLFin)
        subalpha = torch.zeros_like(subLFin)
        subbeta = torch.zeros_like(subLFin)
        subrefine = torch.zeros_like(subLFin)

        net.eval()


        with torch.no_grad():
            for i in range(0, subLFin.size(0), args.minibatch_for_test):
                # print(f"Processing patch {i // args.minibatch_for_test + 1} / {math.ceil(subLFin.size(0) / args.minibatch_for_test)}")
                data = subLFin[i:i + args.minibatch_for_test]
                ref_patch = subRef_LFin[i:i + args.minibatch_for_test]
                refine, pred, alpha, beta = net(data.to(device), ref_patch.to(device))
                # pred = net(data.to(device), ref_patch.to(device))
                subOut[i:i + args.minibatch_for_test] = pred
                subalpha[i:i + args.minibatch_for_test] = alpha
                subbeta[i:i + args.minibatch_for_test] = beta
                subrefine[i:i + args.minibatch_for_test] = refine

        subOut = rearrange(subOut, '(n1 n2) c a1h a2w -> n1 n2 c a1h a2w', n1=numU, n2=numV)
        subalpha = rearrange(subalpha, '(n1 n2) c a1h a2w -> n1 n2 c a1h a2w', n1=numU, n2=numV)
        subbeta = rearrange(subbeta, '(n1 n2) c a1h a2w -> n1 n2 c a1h a2w', n1=numU, n2=numV)
        subrefine = rearrange(subrefine, '(n1 n2) c a1h a2w -> n1 n2 c a1h a2w', n1=numU, n2=numV)

        Rec_4D = LFintegrate(subOut, args.angRes, args.patch_size_for_test,
                              args.stride_for_test, label.size(-2) // args.angRes,
                              label.size(-1) // args.angRes)
        subalpha = LFintegrate(subalpha, args.angRes, args.patch_size_for_test,
                              args.stride_for_test, label.size(-2) // args.angRes,
                              label.size(-1) // args.angRes)
        subbeta = LFintegrate(subbeta, args.angRes, args.patch_size_for_test,
                              args.stride_for_test, label.size(-2) // args.angRes,
                              label.size(-1) // args.angRes)
        subrefine = LFintegrate(subrefine, args.angRes, args.patch_size_for_test,
                              args.stride_for_test, label.size(-2) // args.angRes,
                              label.size(-1) // args.angRes)

        Rec_4D = rearrange(Rec_4D, 'a1 a2 c h w -> 1 c (a1 h) (a2 w)')
        subalpha = rearrange(subalpha, 'a1 a2 c h w -> 1 c (a1 h) (a2 w)')
        subbeta = rearrange(subbeta, 'a1 a2 c h w -> 1 c (a1 h) (a2 w)')
        subrefine = rearrange(subrefine, 'a1 a2 c h w -> 1 c (a1 h) (a2 w)')

        psnr, ssim = cal_metrics(args, label, Rec_4D)
        psnr_mean = psnr.sum() / np.sum(psnr > 0)
        ssim_mean = ssim.sum() / np.sum(ssim > 0)

        psnr_iter_test.append(psnr_mean)
        ssim_iter_test.append(ssim_mean)
        LF_iter_test.append(LF_name[0])
        psnr_iter_test_allviews.append(psnr_mean)
        ssim_iter_test_allviews.append(ssim_mean)

        LF_scence_name = f"{int(LF_name[0]):03}" if LF_name[0].isdigit() else LF_name[0]


        # Save results
        # Save results
        if args.save_output:
            if epoch_dir is not None:
                save_dir_ = epoch_dir.joinpath(LF_scence_name)
                save_dir_.mkdir(exist_ok=True)

                # -------------------------
                # 1) SR 输出（保持你的逻辑：转 uint8 + 拆分保存）
                # -------------------------
                sr = Rec_4D[0].detach().cpu().numpy()  # ✅ 原来 squeeze() 改成取 batch=0，避免维度被挤坏
                refine = subrefine[0].detach().cpu().numpy()  # [c,H,W]
                sr = np.clip(sr, 0, 1)
                refine = np.clip(refine, 0, 1)

                if sr.ndim == 2:  # [H,W]
                    sr = np.stack([sr, sr, sr], axis=0)  # [3,H,W]
                elif sr.ndim == 3 and sr.shape[0] == 1:
                    sr = np.repeat(sr, 3, axis=0)

                Sr_SAI_rgb = (sr * 255).astype(np.uint8)  # [3,H,W]
                refine_rgb = (refine * 255).astype(np.uint8)  # [3,H,W]
                # -------------------------
                # 2) alpha/beta 伪彩色（保持你的逻辑：viridis）
                #    ✅ 只加“压通道成单通道”这一行，保证 alpha/beta 是 [H,W]
                # -------------------------
                cmap = cm.get_cmap('viridis')

                alpha = subalpha[0].detach().cpu().numpy()  # [c,H,W]
                beta = subbeta[0].detach().cpu().numpy()  # [c,H,W]

                # ✅ 最小修改：如果 c>1（比如3），压成 [H,W]，不改后续整体流程
                if alpha.ndim == 3:
                    alpha = alpha.mean(axis=0)  # 或者 alpha = alpha[0]
                if beta.ndim == 3:
                    beta = beta.mean(axis=0)  # 或者 beta = beta[0]

                alpha = alpha.clip(0, 1)  # [H,W]
                beta = beta.clip(0, 1)  # [H,W]

                alpha_color = (cmap(alpha)[..., :3] * 255).astype(np.uint8)  # [H,W,3]
                beta_color = (cmap(beta)[..., :3] * 255).astype(np.uint8)  # [H,W,3]

                # -------------------------
                # 3) 从“整张拼图”拆成 [U,V,h,w,3]（保持你的逻辑不变）
                # -------------------------
                Sr_4D_rgb = rearrange(
                    Sr_SAI_rgb, 'c (a1 h) (a2 w) -> a1 a2 h w c',
                    a1=args.angRes, a2=args.angRes
                )  # [U,V,h,w,3]

                subalpha_4D = rearrange(
                    alpha_color, '(a1 h) (a2 w) c -> a1 a2 h w c',
                    a1=args.angRes, a2=args.angRes
                )  # [U,V,h,w,3]

                subbeta_4D = rearrange(
                    beta_color, '(a1 h) (a2 w) c -> a1 a2 h w c',
                    a1=args.angRes, a2=args.angRes
                )  # [U,V,h,w,3]

                subrefine_4D = rearrange(
                    refine_rgb, 'c (a1 h) (a2 w) -> a1 a2 h w c',
                    a1=args.angRes, a2=args.angRes
                )  # [U,V,h,w,3]
                # -------------------------
                # 4) 保存中心视角（保持你的逻辑不变）
                # -------------------------
                img_center = Sr_4D_rgb[args.angRes // 2, args.angRes // 2]  # [h,w,3]
                path_center = str(CenterView_dir) + '/' + LF_scence_name + '_CenterView.png'
                imageio.imwrite(path_center, img_center)

                # -------------------------
                # 5) 保存每个子视角 + alpha/beta（保持你的逻辑不变）
                # -------------------------
                for i in range(args.angRes):
                    for j in range(args.angRes):
                        img = Sr_4D_rgb[i, j]  # [h,w,3]
                        img_alpha = subalpha_4D[i, j]  # [h,w,3]
                        img_beta = subbeta_4D[i, j]  # [h,w,3]

                        path = str(save_dir_) + '/' + f'{i + 1}_{j + 1}.png'
                        imageio.imwrite(path, img)

                        path_alpha = str(save_dir_) + '/' + f'{i + 1}_{j + 1}_alpha.png'
                        imageio.imwrite(path_alpha, img_alpha)

                        path_beta = str(save_dir_) + '/' + f'{i + 1}_{j + 1}_beta.png'
                        imageio.imwrite(path_beta, img_beta)

                        path_refine = str(save_dir_) + '/' + f'{i + 1}_{j + 1}_refine.png'
                        imageio.imwrite(path_refine, subrefine_4D[i, j])

    return psnr_iter_test, ssim_iter_test, LF_iter_test, psnr_iter_test_allviews, ssim_iter_test_allviews, single_sence_time_avg


import time
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

import time
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

def test_with_time(test_loader, device, net, args,
                   epoch_dir=None, CenterView_dir=None,
                   mat_file=None, aligned_image_dir=None):
    LF_iter_test = []
    psnr_iter_test = []
    ssim_iter_test = []
    psnr_iter_test_allviews = []
    ssim_iter_test_allviews = []
    single_sence_time_avg = []   # 每个场景的“推理+划块+还原”总时间（秒）

    net = net.to(device)
    net.eval()

    for idx_iter, (data, label, LF, LF_name) in tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            ncols=70):

        # ------------------ 基本预处理 ------------------
        data = data.squeeze()   # [C, H, W] 或 [1, C, H, W] -> [C, H, W]
        LF   = LF.squeeze()

        # =====================================================
        # ① 场景级计时开始：
        #    从“划块”开始，到 LFintegrate 还原完 Rec_4D 为止。
        #    不包含：cal_metrics、保存 PNG、写 mat 等操作。
        # =====================================================
        t_scene_start = time.time()

        # ---- LF 分 patch（划块时间也计入）----
        subLFin = LFdivide(data, args.angRes,
                           args.patch_size_for_test,
                           args.stride_for_test)
        subRef_LFin = LFdivide(LF, args.angRes,
                               args.patch_size_for_test,
                               args.stride_for_test)

        numU, numV, c, H, W = subLFin.size()
        subLFin     = rearrange(subLFin,     'n1 n2 c a1h a2w -> (n1 n2) c a1h a2w')
        subRef_LFin = rearrange(subRef_LFin, 'n1 n2 c a1h a2w -> (n1 n2) c a1h a2w')

        subOut = torch.zeros_like(subLFin)

        # ---- 网络前向（同样计入时间）----
        with torch.no_grad():
            for i in range(0, subLFin.size(0), args.minibatch_for_test):
                data_patch = subLFin[i:i + args.minibatch_for_test].to(device)
                ref_patch  = subRef_LFin[i:i + args.minibatch_for_test].to(device)

                _, pred = net(data_patch, ref_patch)


                subOut[i:i + args.minibatch_for_test] = pred.cpu()

        # ---- patch 还原 LF（也计入时间）----
        subOut = rearrange(subOut,
                           '(n1 n2) c a1h a2w -> n1 n2 c a1h a2w',
                           n1=numU, n2=numV)
        Rec_4D = LFintegrate(subOut, args.angRes,
                             args.patch_size_for_test,
                             args.stride_for_test,
                             label.size(-2) // args.angRes,
                             label.size(-1) // args.angRes)
        Rec_4D = rearrange(Rec_4D, 'a1 a2 c h w -> 1 c (a1 h) (a2 w)')

        t_scene_end = time.time()

        # 该场景的“划块 + 前向 + 还原”总时间
        infer_time_scene = t_scene_end - t_scene_start
        single_sence_time_avg.append(infer_time_scene)

        # =====================================================
        # ② 指标计算（不计入时间）
        # =====================================================
        psnr, ssim = cal_metrics(args, label, Rec_4D)
        psnr_mean = psnr.sum() / np.sum(psnr > 0)
        ssim_mean = ssim.sum() / np.sum(ssim > 0)

        psnr_iter_test.append(psnr_mean)
        ssim_iter_test.append(ssim_mean)
        LF_iter_test.append(LF_name[0])
        psnr_iter_test_allviews.append(psnr_mean)
        ssim_iter_test_allviews.append(ssim_mean)

        # ---- 如需保存图像，在这里打开（不计入时间）----
        # if args.save_output:
        #     LF_scence_name = f"{int(LF_name[0]):03}" if LF_name[0].isdigit() else LF_name[0]
        #     save_dir_ = epoch_dir.joinpath(LF_scence_name)
        #     save_dir_.mkdir(exist_ok=True)
        #
        #     Sr_SAI_y = ((Rec_4D.squeeze().cpu().numpy()).clip(0, 1) * 255).astype('uint8')
        #     Sr_4D_rgb = rearrange(Sr_SAI_y,
        #                           'c (a1 h) (a2 w) -> a1 a2 h w c',
        #                           a1=args.angRes, a2=args.angRes)
        #
        #     # Save center view
        #     img = Sr_4D_rgb[args.angRes // 2, args.angRes // 2, :, :, :]
        #     imageio.imwrite(str(CenterView_dir / f"{LF_scence_name}_CenterView.png"), img)
        #
        #     for i in range(args.angRes):
        #         for j in range(args.angRes):
        #             imageio.imwrite(str(save_dir_ / f"{i+1}_{j+1}.png"),
        #                             Sr_4D_rgb[i, j])

    # =====================================================
    # ③ 总结：场景数 + 平均每个场景耗时
    # =====================================================
    num_scenes = len(LF_iter_test)
    avg_scene_time = float(np.mean(single_sence_time_avg)) if num_scenes > 0 else 0.0

    print(f"\nNumber of test scenes (by LF_name): {num_scenes}")
    print(f"Average time per scene (divide + forward + integrate): {avg_scene_time:.6f} seconds")

    return (
        psnr_iter_test,
        ssim_iter_test,
        LF_iter_test,
        psnr_iter_test_allviews,
        ssim_iter_test_allviews,
        single_sence_time_avg
    )


'''Set Set the random number seed'''
def setup_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed has been set to: {seed}")

if __name__ == '__main__':
    from config import args

    # import torch, platform, numpy, h5py, sys
    #
    # print(platform.platform())
    # print('PyTorch', torch.__version__);
    # print(torch.__config__.show())
    # print('Python', sys.version)
    # print('NumPy', numpy.__version__, 'h5py', h5py.__version__)

    main(args)
