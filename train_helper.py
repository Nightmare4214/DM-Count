import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from timm.utils import AverageMeter
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb
from test import do_test, get_dataloader_by_args

import utils.log_utils as log_utils
from datasets.crowd import Crowd_qnrf, Crowd_nwpu, Crowd_sh
from losses.ot_loss import OT_Loss
from models import vgg19
from utils.pytorch_utils import Save_Handle, seed_worker, setup_seed


include_keys=['max_epoch', 'crop_size', 'extra_aug', 'lr', 'wot', 'wtv', 'reg', 'num_of_iter_in_ot', 'norm_cood', 'batch_size']


def get_run_name_by_args(args, include_keys=None, exclude_keys=None):
    data = args.__dict__
    result = []
    if include_keys:
        for k in include_keys:
            result.append(f'{k}_{data[k]}')
    else:
        for k, v in data.items():
            if exclude_keys and k in exclude_keys:
                continue
            result.append(f'{k}_{v}')
    return '_'.join(result)


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    gt_discretes = torch.stack(transposed_batch[2], 0)
    return images, points, gt_discretes


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def setup(self):
        args = self.args
        if args.randomless:
            seed = args.seed
            g = torch.Generator()
            g.manual_seed(seed)
            setup_seed(seed)
        else:
            torch.backends.cudnn.benchmark = True

        if os.path.exists(args.resume):
            self.save_dir = os.path.dirname(args.resume)
        else:
            self.save_dir = os.path.join(args.save_dir, get_run_name_by_args(args, include_keys) + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S'))
        args.save_dir = self.save_dir
        self.args = args
        os.makedirs(self.save_dir, exist_ok=True)
        # os.environ["WANDB_MODE"] = "offline"
        time_str = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
        self.logger = log_utils.get_logger(os.path.join(self.save_dir, 'train-{:s}.log'.format(time_str)))
        log_utils.print_config(vars(args), self.logger)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1
            self.logger.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        downsample_ratio = 8
        if args.dataset.lower() == 'qnrf':
            self.datasets = {x: Crowd_qnrf(os.path.join(args.data_dir, x),
                                           args.crop_size, downsample_ratio, x, extra_aug=args.extra_aug) for x in ['train', 'val']}
        elif args.dataset.lower() == 'nwpu':
            self.datasets = {x: Crowd_nwpu(os.path.join(args.data_dir, x),
                                           args.crop_size, downsample_ratio, x, extra_aug=args.extra_aug) for x in ['train', 'val']}
        elif args.dataset.lower() == 'sha' or args.dataset.lower() == 'shb':
            self.datasets = {'train': Crowd_sh(os.path.join(args.data_dir, 'train'),
                                               args.crop_size, downsample_ratio, 'train', extra_aug=args.extra_aug),
                             'val': Crowd_sh(os.path.join(args.data_dir, 'val'),
                                             args.crop_size, downsample_ratio, 'val', extra_aug=args.extra_aug),
                             }
        else:
            raise NotImplementedError

        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                                      if x == 'train' else 1),
                                          shuffle=(x == 'train'),
                                          num_workers=args.num_workers * self.device_count if x == 'train' else 0,
                                          pin_memory=(x == 'train'),
                                          worker_init_fn=seed_worker if args.randomless else None, generator=g if args.randomless else None
                                          )
                            for x in ['train', 'val']}
        self.model = vgg19().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0
        self.wandb_id = None
        if args.resume:
            self.logger.info('loading pretrained model from ' + args.resume)
            suf = os.path.splitext(args.resume)[-1]
            if suf == '.tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_mae = checkpoint['best_mae']
                self.best_mse = checkpoint['best_mse']
                self.best_count = checkpoint['best_count']
                if 'wandb_id' in checkpoint:
                    self.wandb_id = checkpoint['wandb_id']
                if args.randomless:
                    random.setstate(checkpoint['random_state'])
                    np.random.set_state(checkpoint['np_random_state'])
                    torch.random.set_rng_state(checkpoint['torch_random_state'].cpu())
            elif suf == '.pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))
        else:
            self.logger.info('random initialization')

        self.ot_loss = OT_Loss(args.crop_size, downsample_ratio, args.norm_cood, self.device, args.num_of_iter_in_ot,
                               args.reg)
        self.tv_loss = nn.L1Loss(reduction='none').to(self.device)
        self.mse = nn.MSELoss().to(self.device)
        self.mae = nn.L1Loss().to(self.device)

        self.log_dir = os.path.join(self.save_dir, 'runs')
        # self.writer = SummaryWriter(self.log_dir)
        self.save_list = Save_Handle(max_num=1)
        wandb.init(
            # set the wandb project where this run will be logged
            project="DM-Count",
            id = self.wandb_id,
            name = os.path.basename(self.args.save_dir),
            # track hyperparameters and run metadata
            config=args,
            resume=True if args.resume else None,
            # sync_tensorboard=True
        )
        self.wandb_id = wandb.run.id

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch + 1):
            self.logger.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)
            self.epoch = epoch
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()
        self.val_epoch()

    def train_eopch(self):
        epoch_ot_loss = AverageMeter()
        epoch_ot_obj_value = AverageMeter()
        epoch_wd = AverageMeter()
        epoch_count_loss = AverageMeter()
        epoch_tv_loss = AverageMeter()
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        for step, (inputs, points, gt_discrete) in enumerate(tqdm(self.dataloaders['train'])):
            inputs = inputs.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)  # (B,)
            count_tensor = torch.from_numpy(gd_count).float().to(self.device)  # (B,)
            points = [p.to(self.device) for p in points]  # (B, N, 2)
            gt_discrete = gt_discrete.to(self.device)
            N = inputs.size(0)

            outputs, outputs_normed = self.model(inputs)  # (B, 1, 64, 64)
            # Compute OT loss. ot_loss=<im_grad, predicted density> ,wd=<C,P>,  ot_obj_value=<z_hat / ||z_hat||1, beta>
            ot_loss, wd, ot_obj_value = self.ot_loss(outputs_normed, outputs, points)
            ot_loss = ot_loss * self.args.wot
            ot_obj_value = ot_obj_value * self.args.wot
            epoch_ot_loss.update(ot_loss.item(), N)
            epoch_ot_obj_value.update(ot_obj_value.item(), N)
            epoch_wd.update(wd, N)
            # Compute counting loss.
            count_loss = self.mae(torch.sum(outputs, dim=(1, 2, 3)),
                                  count_tensor)
            epoch_count_loss.update(count_loss.item(), N)

            # Compute TV loss.
            gd_count_tensor = count_tensor[..., None, None, None]  # (B, 1, 1, 1)
            gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)
            tv_loss = (torch.sum(self.tv_loss(outputs_normed, gt_discrete_normed), dim=(1, 2, 3)) * count_tensor).mean(
                0) * self.args.wtv
            epoch_tv_loss.update(tv_loss.item(), N)

            loss = ot_loss + count_loss + tv_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
            pred_err = pred_count - gd_count
            epoch_loss.update(loss.item(), N)
            epoch_mse.update(np.mean(pred_err * pred_err), N)
            epoch_mae.update(np.mean(abs(pred_err)), N)
        
        wandb.log({
            'train/loss': epoch_loss.avg,
            'train/ot': epoch_ot_loss.avg,
            'train/wd': epoch_wd.avg,
            'train/ot_obj': epoch_ot_obj_value.avg,
            'train/count': epoch_count_loss.avg,
            'train/tv': epoch_tv_loss.avg,
            'train/mse': np.sqrt(epoch_mse.avg),
            'train/mae': epoch_mae.avg,
        }, step=self.epoch)
        # self.writer.add_scalar('train/loss', epoch_loss.avg, self.epoch)
        # self.writer.add_scalar('train/ot', epoch_ot_loss.avg, self.epoch)
        # self.writer.add_scalar('train/wd', epoch_wd.avg, self.epoch)
        # self.writer.add_scalar('train/ot_obj', epoch_ot_obj_value.avg, self.epoch)
        # self.writer.add_scalar('train/count', epoch_count_loss.avg, self.epoch)
        # self.writer.add_scalar('train/tv', epoch_tv_loss.avg, self.epoch)
        # self.writer.add_scalar('train/mse', np.sqrt(epoch_mse.avg), self.epoch)
        # self.writer.add_scalar('train/mae', epoch_mae.avg, self.epoch)

        self.logger.info(
            'Epoch {} Train, Loss: {:.2f}, OT Loss: {:.2e}, Wass Distance: {:.2f}, OT obj value: {:.2f}, '
            'Count Loss: {:.2f}, TV Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
            .format(self.epoch, epoch_loss.avg, epoch_ot_loss.avg, epoch_wd.avg,
                    epoch_ot_obj_value.avg, epoch_count_loss.avg, epoch_tv_loss.avg,
                    np.sqrt(epoch_mse.avg), epoch_mae.avg,
                    time.time() - epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic,
            'best_mae': self.best_mae,
            'best_mse': self.best_mse,
            'best_count': self.best_count,
            'wandb_id': self.wandb_id,
            'random_state': random.getstate(),
            'np_random_state': np.random.get_state(),
            'torch_random_state': torch.random.get_rng_state()
        }, save_path)
        self.save_list.append(save_path)

    def val_epoch(self):
        # args = self.args
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        with torch.no_grad():
            for inputs, count, name in tqdm(self.dataloaders['val']):
                inputs = inputs.to(self.device)
                assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'

                outputs, _ = self.model(inputs)
                res = count.shape[1] - torch.sum(outputs).item()
                del inputs
                del outputs
                torch.cuda.empty_cache()
                epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        wandb.log({
            'val/mae': mae,
            'val/mse': mse,
        }, step=self.epoch)
        # self.writer.add_scalar('val/mae', mae, self.epoch)
        # self.writer.add_scalar('val/mse', mse, self.epoch)
        self.logger.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, mse, mae, time.time() - epoch_start))

        model_state_dic = self.model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            if mae < self.best_mae or (mae == self.best_mae and mse < self.best_mse):
                torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_mae.pth'))
            self.best_mse = mse
            self.best_mae = mae
            self.logger.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                     self.best_mae,
                                                                                     self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
            self.best_count += 1
    
    def test(self):
        dataloader = get_dataloader_by_args(self.args)
        model_path = os.path.join(self.save_dir, 'best_model_mae.pth')
        self.model.load_state_dict(torch.load(model_path, self.device))
        mae, mse = do_test(self.model, self.device, dataloader, model_path, pred_density_map=True)
        wandb.summary['test_mae'] = mae
        wandb.summary['test_mse'] = mse
