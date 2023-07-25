import argparse
import os

import numpy as np
import torch
from cv2 import cv2
from tqdm import tqdm

import datasets.crowd as crowd
from models import vgg19
from utils.pytorch_utils import setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='the crop size of the train image')
    parser.add_argument('--model_path', type=str, default='pretrained_models/model_qnrf.pth',
                        help='saved model path')
    parser.add_argument('--data_dir', type=str,
                        default='/home/icml007/Nightmare4214/datasets/UCF-Train-Val-Test',
                        help='saved model path')
    parser.add_argument('--dataset', type=str, default='qnrf',
                        help='dataset name: qnrf, nwpu, sha, shb')
    parser.add_argument('--pred_density_map', default=False, required=False, action='store_true',
                        help='save predicted density')
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


def get_dataloader_by_args(args):
    if args.dataset.lower() == 'qnrf':
        dataset = crowd.Crowd_qnrf(os.path.join(args.data_dir, 'test'), args.crop_size, 8, method='val')
    elif args.dataset.lower() == 'nwpu':
        dataset = crowd.Crowd_nwpu(os.path.join(args.data_dir, 'val'), args.crop_size, 8, method='val')
    elif args.dataset.lower() in ['sha', 'shb']:
        dataset = crowd.Crowd_sh(os.path.join(args.data_dir, 'test'), args.crop_size, 8, method='val')
    else:
        raise NotImplementedError
    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False,
                                             num_workers=1, pin_memory=True)
    return dataloader


def do_test(model, device, dataloader, model_path, pred_density_map=True, **kwargs):

    if pred_density_map:
        pred_density_map_path = os.path.join(os.path.dirname(model_path), 'pred_density_map')
        os.makedirs(pred_density_map_path, exist_ok=True)

    image_errs = []
    with torch.no_grad():
        for inputs, count, name in tqdm(dataloader):
            inputs = inputs.to(device)
            assert inputs.size(0) == 1, 'the batch size should equal to 1'

            outputs, _ = model(inputs)
            img_err = count.shape[1] - torch.sum(outputs).item()

            # print(name, img_err, count[0].item(), torch.sum(outputs).item())
            image_errs.append(img_err)

            if pred_density_map:
                vis_img = outputs[0, 0].cpu().numpy()
                # normalize density map values from 0 to 1, then map it to 0-255.
                vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
                vis_img = (vis_img * 255).astype(np.uint8)
                vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(pred_density_map_path, str(name[0]) + '.png'), vis_img)

    image_errs = np.array(image_errs)
    mse = np.sqrt(np.mean(np.square(image_errs)))
    mae = np.mean(np.abs(image_errs))
    print('{}: mae {}, mse {}\n'.format(model_path, mae, mse))
    with open(os.path.join(os.path.dirname(model_path),
                           os.path.splitext(os.path.basename(model_path))[0] + '_predict.log'), 'w') as f:
        f.write('{}: mae {}, mse {}\n'.format(os.path.basename(model_path), mae, mse))
    return mae, mse


if __name__ == '__main__':
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    # setup_seed(42)
    device = torch.device('cuda')

    model_path = args.model_path
    dataloader = get_dataloader_by_args(args)

    model = vgg19().to(device)
    model.load_state_dict(torch.load(model_path, device))
    model.eval()
    
    do_test(model, device, dataloader, model_path)
