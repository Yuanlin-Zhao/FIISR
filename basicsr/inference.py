import logging
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def inference_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':
    import sys
    import os.path as osp

    # 更加强劲的注入方式
    if len(sys.argv) == 1:

        sys.argv.extend(['-opt', r'D:\zyl\textsr\ours\seemoredetails-main\options\super_resolution\test\test_FIISR_S_X4.yml'])
    # 自动定位项目根目录 (通常是 train.py 的上两级)
    # 确保你的 options 文件夹在 root_path 下面
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))

    # 打印一下，方便你在控制台看到当前的运行路径，防止找不到 yml 文件
    print(f"Current root path: {root_path}")
    print(f"Arguments: {sys.argv}")

    inference_pipeline(root_path)