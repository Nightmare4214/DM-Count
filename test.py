import argparse
import os

import numpy as np
import torch
from cv2 import cv2
from tqdm import tqdm

import datasets.crowd as crowd
from models import vgg19
from utils.pytorch_utils import setup_seed

parser = argparse.ArgumentParser(description='Test ')
parser.add_argument('--device', default='0', help='assign device')
parser.add_argument('--crop_size', type=int, default=512,
                    help='the crop size of the train image')
parser.add_argument('--model_path', type=str, default='pretrained_models/model_qnrf.pth',
                    help='saved model path')
parser.add_argument('--data_path', type=str,
                    default='/home/icml007/Nightmare4214/datasets/UCF-Train-Val-Test',
                    help='saved model path')
parser.add_argument('--dataset', type=str, default='qnrf',
                    help='dataset name: qnrf, nwpu, sha, shb')
parser.add_argument('--pred_density_map_path', type=str, default='',
                    help='save predicted density maps when pred-density-map-path is not empty.')

if __name__ == '__main__':
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    # setup_seed(42)
    device = torch.device('cuda')

    model_path = args.model_path
    crop_size = args.crop_size
    data_path = args.data_path
    if args.dataset.lower() == 'qnrf':
        dataset = crowd.Crowd_qnrf(os.path.join(data_path, 'test'), crop_size, 8, method='val')
    elif args.dataset.lower() == 'nwpu':
        dataset = crowd.Crowd_nwpu(os.path.join(data_path, 'val'), crop_size, 8, method='val')
    elif args.dataset.lower() == 'sha' or args.dataset.lower() == 'shb':
        dataset = crowd.Crowd_sh(os.path.join(data_path, 'test_data'), crop_size, 8, method='val')
    else:
        raise NotImplementedError
    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False,
                                             num_workers=1, pin_memory=True)

    if not args.pred_density_map_path:
        args.pred_density_map_path = os.path.join(os.path.dirname(model_path), 'pred_density_map')
    os.makedirs(args.pred_density_map_path, exist_ok=True)

    model = vgg19().to(device)
    model.load_state_dict(torch.load(model_path, device))
    model.eval()
    image_errs = []
    with torch.no_grad():
        for inputs, count, name in tqdm(dataloader):
            inputs = inputs.to(device)
            assert inputs.size(0) == 1, 'the batch size should equal to 1'

            outputs, _ = model(inputs)
            img_err = count[0].item() - torch.sum(outputs).item()

            # print(name, img_err, count[0].item(), torch.sum(outputs).item())
            image_errs.append(img_err)

            if args.pred_density_map_path:
                vis_img = outputs[0, 0].cpu().numpy()
                # normalize density map values from 0 to 1, then map it to 0-255.
                vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
                vis_img = (vis_img * 255).astype(np.uint8)
                vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + '.png'), vis_img)

    image_errs = np.array(image_errs)
    mse = np.sqrt(np.mean(np.square(image_errs)))
    mae = np.mean(np.abs(image_errs))
    print('{}: mae {}, mse {}\n'.format(model_path, mae, mse))
    with open(os.path.join(os.path.dirname(model_path),
                           os.path.splitext(os.path.basename(model_path))[0] + '_predict.log'), 'w') as f:
        f.write('{}: mae {}, mse {}\n'.format(os.path.basename(model_path), mae, mse))
