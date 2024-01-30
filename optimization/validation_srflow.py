import numpy as np
import torch
import random

import PIL
import os
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

def validate(model, val_loader, metric_dict, exp_name, logstep, args):

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cmap = 'viridis' if args.trainset == 'era5-TCW' else 'inferno'

    nll_list=[]
    model.eval()

    with torch.no_grad():
        for batch_idx, item in enumerate(val_loader):

            y = item[0].to(args.device)
            x = item[1].to(args.device)

            z, nll = model.forward(x_hr=y, xlr=x)

            # Generative loss
            nll_list.append(nll.mean().detach().cpu().numpy())

            if batch_idx == 20:
                break

        # evalutae for different temperatures
        mu0, _, _ = model(xlr=x, reverse=True, eps=0)
        mu05, _, _ = model(xlr=x, reverse=True, eps=0.5)
        mu08, _, _ = model(xlr=x, reverse=True, eps=0.8)
        mu1, _, _ = model(xlr=x, reverse=True, eps=1.0)

        savedir = "{}/snapshots/validationset/".format(exp_name)

        os.makedirs(savedir, exist_ok=True)

        # Visualize low resolution GT
        grid_low_res = torchvision.utils.make_grid(x[0:9, :, :, :].cpu(), nrow=3)
        plt.figure()
        plt.imshow(grid_low_res.permute(1, 2, 0)[:,:,0], cmap=cmap)
        plt.axis('off')
        plt.title("Low-Res GT (train)")
        # plt.show()
        plt.savefig(savedir + '/low_res_gt{}.png'.format(logstep), dpi=300, bbox_inches='tight')
        plt.close()

        # Visualize High-Res GT
        grid_high_res_gt = torchvision.utils.make_grid(y[0:9, :, :, :].cpu(), nrow=3)
        plt.figure()
        plt.imshow(grid_high_res_gt.permute(1, 2, 0)[:,:,0], cmap=cmap)
        plt.axis('off')
        plt.title("High-Res GT")
        # plt.show()
        plt.savefig(savedir + '/high_res_gt_{}.png'.format(logstep), dpi=300, bbox_inches='tight')
        plt.close()

        grid_mu0 = torchvision.utils.make_grid(mu0[0:9,:,:,:].cpu(), normalize=True, nrow=3)
        plt.figure()
        plt.imshow(grid_mu0.permute(1, 2, 0)[:,:,0].contiguous(), cmap=cmap)
        plt.axis('off')
        plt.title("Prediction at t (valid), mu=0")
        plt.savefig(savedir + "mu_0_logstep_{}_valid.png".format(logstep), dpi=300)
        plt.close()

        grid_mu05 = torchvision.utils.make_grid(mu05[0:9,:,:,:].cpu(), normalize=True, nrow=3)
        plt.figure()
        plt.imshow(grid_mu0.permute(1, 2, 0)[:,:,0].contiguous(), cmap=cmap)
        plt.axis('off')
        plt.title("Prediction at t (valid), mu=0.5")
        plt.savefig(savedir + "mu_0.5_logstep_{}_valid.png".format(logstep), dpi=300)
        plt.close()

        grid_mu08 = torchvision.utils.make_grid(mu08[0:9,:,:,:].cpu(), normalize=True, nrow=3)
        plt.figure()
        plt.imshow(grid_mu08.permute(1, 2, 0)[:,:,0].contiguous(), cmap=cmap)
        plt.axis('off')
        plt.title("Prediction at t (valid), mu=0.8")
        plt.savefig(savedir + "mu_0.8_logstep_{}_valid.png".format(logstep), dpi=300)
        plt.close()

        grid_mu1 = torchvision.utils.make_grid(mu1[0:9,:,:,:].cpu(), normalize=True, nrow=3)
        plt.figure()
        plt.imshow(grid_mu1.permute(1, 2, 0)[:,:,0].contiguous(), cmap=cmap)
        plt.axis('off')
        plt.title("Prediction at t (valid), mu=1.0")
        plt.savefig(savedir + "mu_1_logstep_{}_valid.png".format(logstep), dpi=300)
        plt.close()

        abs_err = torch.abs(mu08 - y)
        grid_abs_error = torchvision.utils.make_grid(abs_err[0:9,:,:,:].cpu(), normalize=True, nrow=3)
        plt.figure()
        plt.imshow(grid_abs_error.permute(1, 2, 0)[:,:,0], cmap=cmap)
        plt.axis('off')
        plt.title("Abs Err")
        plt.savefig(savedir + '/abs_err_{}.png'.format(logstep), dpi=300,bbox_inches='tight')
        plt.close()

        metric_dict['MSE'].append(metrics.MSE(inv_scaler(mu08, args), y).mean())
        metric_dict['MAE'].append(metrics.MAE(inv_scaler(mu08, args), y).mean())
        metric_dict['RMSE'].append(metrics.RMSE(inv_scaler(mu08, args), y).mean())

        with open(savedir + '/metric_dict.txt', 'w') as f:
            for key, value in metric_dict.items():
                f.write('%s:%s\n' % (key, value))

    return metric_dict, np.mean(nll_list)
