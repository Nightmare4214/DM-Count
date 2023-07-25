import argparse
import os

import torch

from train_helper import Trainer
from utils.pytorch_utils import setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--save_dir', default='/home/icml007/Nightmare4214/PyTorch_model/DM-Count',
                        help='directory to save models.')
    parser.add_argument('--data_dir', default='/home/icml007/Nightmare4214/datasets/UCF-Train-Val-Test',
                        help='data path')
    parser.add_argument('--dataset', default='qnrf', help='dataset name: qnrf, nwpu, sha, shb')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='the initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='the weight decay')
    parser.add_argument('--resume', default='', type=str,
                        help='the path of resume training model')
    parser.add_argument('--max_epoch', type=int, default=1000,
                        help='max training epoch')
    parser.add_argument('--val_epoch', type=int, default=5,
                        help='the num of steps to log training information')
    parser.add_argument('--val_start', type=int, default=50,
                        help='the epoch start to val')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num_workers', type=int, default=3,
                        help='the num of training process')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='the crop size of the train image')
    parser.add_argument('--wot', type=float, default=0.1, help='weight on OT loss')
    parser.add_argument('--wtv', type=float, default=0.01, help='weight on TV loss')
    parser.add_argument('--reg', type=float, default=10.0,
                        help='entropy regularization in sinkhorn')
    parser.add_argument('--num_of_iter_in_ot', type=int, default=100,
                        help='sinkhorn iterations')
    parser.add_argument('--norm_cood', type=int, default=0, help='whether to norm cood when computing distance')

    args = parser.parse_args()

    if args.dataset.lower() == 'qnrf':
        args.crop_size = 512
    elif args.dataset.lower() == 'nwpu':
        args.crop_size = 384
        args.val_epoch = 50
    elif args.dataset.lower() == 'sha':
        args.crop_size = 256
    elif args.dataset.lower() == 'shb':
        args.crop_size = 512
    else:
        raise NotImplementedError
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    # setup_seed(42)
    trainer = Trainer(args)
    trainer.setup()
    trainer.train()
    trainer.test()
