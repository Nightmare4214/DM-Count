import os
import random

import numpy as np
import torch


def adjust_learning_rate(optimizer, epoch, initial_lr=0.001, decay_epoch=10):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = max(initial_lr * (0.1 ** (epoch // decay_epoch)), 1e-6)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Save_Handle(object):
    """handle the number of """

    def __init__(self, max_num):
        self.save_list = []
        self.max_num = max_num

    def append(self, save_path):
        if len(self.save_list) < self.max_num:
            self.save_list.append(save_path)
        else:
            remove_path = self.save_list[0]
            del self.save_list[0]
            self.save_list.append(save_path)
            if os.path.exists(remove_path):
                os.remove(remove_path)


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed=42):
    """
    set random seed

    :param seed: seed num
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # LSTM(cuda>10.2)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.use_deterministic_algorithms(True, warn_only=True)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
