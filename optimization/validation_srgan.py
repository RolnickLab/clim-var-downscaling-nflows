import numpy as np
import torch
import random
import timeit
import PIL
import os
from torch.autograd import Variable
import torch
from torch import nn
import torchvision
from torchvision import transforms
from utils import metrics
import sys
sys.path.append("../../")

from os.path import exists, join
import matplotlib.pyplot as plt
import pdb

def inv_scaler(x, args):
    min_value = 0 if args.trainset == 'era5-TCW' else 315.91873
    max_value = 100 if args.trainset == 'era5-TCW' else 241.22385
    x = x * (max_value - min_value) + min_value
    return x

def minmax_scaler(x, args):
    values_range = (-1, 1)
    min_value = 0 if args.trainset == 'era5-TCW' else 315.91873
    max_value = 100 if args.trainset == 'era5-TCW' else 241.22385
    x = (x - min_value) / (max_value - min_value)
    return x * (values_range[1] - values_range[0]) + values_range[0]

def validate(discriminator, generator, val_loader, metric_dict, exp_name, logstep, args):

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cmap = 'viridis' if args.trainset == 'era5-TCW' else 'inferno'

    discriminator.eval()
    generator.eval()

    mse_loss_list = []
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    with torch.no_grad():
        for batch_idx, item in enumerate(val_loader):

            y = item[0].to(args.device)
            x = item[1].to(args.device)

            fake_img=generator(x)
            fake_out = discriminator(fake_img).mean()
            real_out = discriminator(y).mean()

            g_loss = mse_loss(fake_img, y)
            
            # Generative loss
            mse_loss_list.append(g_loss.mean().detach().cpu().numpy())

            if batch_idx == 10:
                break

            viz_dir = "{}/snapshots/validationset/".format(exp_name)
            os.makedirs(viz_dir, exist_ok=True)

            # Visualize low resolution GT
            grid_low_res = torchvision.utils.make_grid(x[0:9, :, :, :].cpu(), nrow=3)
            plt.figure()
            plt.imshow(grid_low_res.permute(1, 2, 0)[:,:,0], cmap=cmap)
            plt.axis('off')
            plt.title("Low-Res GT (train)")
            # plt.show()
            plt.savefig(viz_dir + '/low_res_gt{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
            plt.close()

            # Visualize High-Res GT
            grid_high_res_gt = torchvision.utils.make_grid(y[0:9, :, :, :].cpu(), nrow=3)
            plt.figure()
            plt.imshow(grid_high_res_gt.permute(1, 2, 0)[:,:,0], cmap=cmap)
            plt.axis('off')
            plt.title("High-Res GT")
            # plt.show()
            plt.savefig(viz_dir + '/high_res_gt_{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
            plt.close()

            # Super-Resolving low-res
            start = timeit.default_timer()
            y_hat=generator(x)
            stop = timeit.default_timer()
            print("Time Fwd pass:", stop-start)
            print(y_hat.max(), y_hat.min(), y.max(), y.min())
            grid_y_hat = torchvision.utils.make_grid(y_hat[0:9, :, :, :].cpu(), nrow=3)
            plt.figure()
            plt.imshow(grid_y_hat.permute(1, 2, 0)[:,:,0], cmap=cmap)
            plt.axis('off')
            plt.title("Y hat")
            plt.savefig(viz_dir + '/y_hat_{}.png'.format(batch_idx), dpi=300,bbox_inches='tight')
            # plt.show()
            plt.close()

            abs_err = torch.abs(y_hat - y)
            grid_abs_error = torchvision.utils.make_grid(abs_err[0:9,:,:,:].cpu(), nrow=3)
            plt.figure()
            plt.imshow(grid_abs_error.permute(1, 2, 0)[:,:,0], cmap=cmap)
            plt.axis('off')
            plt.title("Abs Err")
            plt.savefig(viz_dir + '/abs_err_{}.png'.format(batch_idx), dpi=300,bbox_inches='tight')
            # plt.show()
            plt.close()

            metric_dict['MSE'].append(metrics.MSE(inv_scaler(y_hat, args), y).mean())
            metric_dict['MAE'].append(metrics.MAE(inv_scaler(y_hat, args), y).mean())
            metric_dict['RMSE'].append(metrics.RMSE(inv_scaler(y_hat, args), y).mean())

            with open(viz_dir + '/metric_dict.txt', 'w') as f:
                for key, value in metric_dict.items():
                    f.write('%s:%s\n' % (key, value))

    return np.mean(mse_loss_list), metric_dict 
