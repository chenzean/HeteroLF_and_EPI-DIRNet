import importlib
import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.utils_datasets import TestSetDataLoader, TestSetDataLoader
from collections import OrderedDict
from train import test
import random


def main(args):
    ''' Create Dir for Save '''
    _, result_dir, _  = create_dir(args)
    result_dir = result_dir.joinpath('TEST')
    result_dir.mkdir(exist_ok=True)

    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    ''' DATA TEST LOADING '''
    print('\nLoad Test Dataset ...')
    test_Dataset = TestSetDataLoader(args)
    test_loader = torch.utils.data.DataLoader(dataset=test_Dataset, num_workers=args.num_workers,
                                             batch_size=1, shuffle=False)
    print("The number of test data is: %d" % len(test_Dataset))

    test_name = 'Test'

    ''' MODEL LOADING '''
    print('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)

    ''' Load Pre-Trained PTH '''
    if args.use_pre_ckpt:
        try:

            # 加载预训练的检查点
            checkpoint = torch.load(args.path_pre_pth, map_location='cuda:0')

            # 仅加载模型的权重
            net.load_state_dict(checkpoint['state_dict'])
            print('Using pre-trained model: %s' % args.path_pre_pth)

        except Exception as e:
            print(f'[Error loading pretrain]: {str(e)}')
            # 如果加载失败，则使用模型的初始化方法
            net.apply(MODEL.weights_init)
    else:
        print('No pre-trained model used!')
        # 如果没有使用预训练模型，直接初始化模型
        net.apply(MODEL.weights_init)

    # net = net.to(device)
    cudnn.benchmark = True

    ''' Print Parameters '''
    print('PARAMETER ...')
    print(args)
    total_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f" - Net Parameters   : {total_params:.4f}M")

    ''' TEST on every dataset '''
    print('\nStart test...')
    with torch.no_grad():
        ''' Create Excel for PSNR/SSIM '''
        excel_file = ExcelFile()
        excel_file_allviews = ExcelFile_allviews()

        save_dir = result_dir.joinpath(test_name)
        save_dir.mkdir(exist_ok=True)

        CenterView_dir = save_dir.joinpath('CenterView/')
        CenterView_dir.mkdir(exist_ok=True)

        mat_file = save_dir.joinpath('mat_file/')
        mat_file.mkdir(exist_ok=True)

        LF_epoch_save_dir = save_dir.joinpath('Results/')
        LF_epoch_save_dir.mkdir(exist_ok=True)

        aligned_image_dir = save_dir.joinpath('aligned_image')
        aligned_image_dir.mkdir(exist_ok=True)

        psnr_iter_test, ssim_iter_test, LF_name, psnr_iter_test_allviews,ssim_iter_test_allviews, single_sence_time_avg = test(test_loader, device, net, args, LF_epoch_save_dir,
                                                                        CenterView_dir, mat_file, aligned_image_dir)

        psnr_iter_test_avg = sum(psnr_iter_test) / len(psnr_iter_test) if psnr_iter_test else 0
        ssim_iter_test_avg = sum(ssim_iter_test) / len(ssim_iter_test) if ssim_iter_test else 0

        print('Test psnr/ssim is %.2f/%.3f' % (psnr_iter_test_avg, ssim_iter_test_avg))

        excel_file.xlsx_file.save(str(result_dir) + '/evaluation.xls')
        excel_file_allviews.xlsx_file.save(str(result_dir) + '/evaluation_allviews.xls')

    pass







def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    from config import args

    main(args)
