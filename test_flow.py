import numpy as np
import torch
import random
from models.architectures import srgan, srgan2, srgan2_stochastic
import PIL
import os
import torchvision
from torchvision import transforms
from ignite.metrics import PSNR
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import sys
sys.path.append("../../")

# seeding only for debugging
# random.seed(0)
# torch.manual_seed(0)
# np.random.seed(0)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# Dataset loading
from data import dataloading
from data.era5_temp_dataset import InverseMinMaxScaler

from os.path import exists, join
import matplotlib.pyplot as plt
from matplotlib import transforms
import argparse
import timeit
import pdb
import seaborn as sns

from models.architectures import srflow
from utils import metrics, wasserstein
from geomloss import SamplesLoss
from operator import add
from scipy import ndimage
parser = argparse.ArgumentParser()

# train configs
parser.add_argument("--model", type=str, default="srflow",
                    help="Model you want to train.")
parser.add_argument("--modeltype", type=str, default="srflow",
                    help="Specify modeltype you would like to train [srflow, cdiff, srgan].")
parser.add_argument("--model_path", type=str, default="runs/",
                    help="Directory where models are saved.")
parser.add_argument("--modelname", type=str, default=None,
                    help="Sepcify modelname to be tested.")
parser.add_argument("--epochs", type=int, default=10000,
                    help="number of epochs")
parser.add_argument("--max_steps", type=int, default=2000000,
                    help="For training on a large dataset.")
parser.add_argument("--log_interval", type=int, default=100,
                    help="Interval in which results should be logged.")

# runtime configs
parser.add_argument("--visual", action="store_true",
                    help="Visualizing the samples at test time.")
parser.add_argument("--noscaletest", action="store_true",
                    help="Disable scale in coupling layers only at test time.")
parser.add_argument("--noscale", action="store_true",
                    help="Disable scale in coupling layers.")
parser.add_argument("--testmode", action="store_true",
                    help="Model run on test set.")
parser.add_argument("--train", action="store_true",
                    help="If model should be trained.")
parser.add_argument("--resume_training", action="store_true",
                    help="If training should be resumed.")
parser.add_argument("--constraint", type=str, default='scaddDS',
                    help="Physical Constraint to be applied during training.")                   

# hyperparameters
parser.add_argument("--nbits", type=int, default=8,
                    help="Images converted to n-bit representations.")
parser.add_argument("--s", type=int, default=2, help="Upscaling factor.")
parser.add_argument("--crop_size", type=int, default=500,
                    help="Crop size when random cropping is applied.")
parser.add_argument("--patch_size", type=int, default=500,
                    help="Training patch size.")
parser.add_argument("--bsz", type=int, default=16, help="batch size")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="learning rate")
parser.add_argument("--filter_size", type=int, default=512,
                    help="filter size NN in Affine Coupling Layer")
parser.add_argument("--L", type=int, default=3, help="# of levels")
parser.add_argument("--K", type=int, default=2,
                    help="# of flow steps, i.e. model depth")
parser.add_argument("--nb", type=int, default=16,
                    help="# of residual-in-residual blocks LR network.")
parser.add_argument("--condch", type=int, default=128//8,
                    help="# of residual-in-residual blocks in LR network.")

# data
parser.add_argument("--datadir", type=str, default="/home/mila/c/christina.winkler/scratch/data",
                    help="Dataset to train the model on.")
parser.add_argument("--trainset", type=str, default="era5-TCW", help='[era5-TCW, era5-T2M]')
parser.add_argument("--testset", type=str, default="era5-TCW",
                    help="Specify test dataset")

args = parser.parse_args()

def inv_scaler(x, min_value=None, max_value=None):
    # min_value = 0 if args.trainset == 'era5-TCW' else 315.91873
    # max_value = 100 if args.trainset == 'era5-TCW' else 241.22385
    x = x * (max_value - min_value) + min_value
    # return
    return x

def plot_std(model, test_loader, exp_name, modelname, args):
    """
    For this experiment we visualize the super-resolution space for a single
    low-resolution image and its possible HR target predictions. We visualize
    the standard deviation of these predictions from the mean of the model.
    """
    color = 'plasma'
    savedir_viz = "experiments/{}_{}_{}_{}x/snapshots/population_std/".format(exp_name, modelname, args.trainset, args.s)
    os.makedirs(savedir_viz, exist_ok=True)
    model.eval()
    cmap = 'viridis' if args.trainset == 'era5-TCW' else 'inferno'
    with torch.no_grad():
        for batch_idx, item in enumerate(test_loader):

            y = item[0].to(args.device)
            x = item[1].to(args.device)

            y_unorm = item[2].to(args.device)
            x_unnorm = item[3].to(args.device)

            mu0,_,_ = model(xlr=x, reverse=True, eps=0.0000000000000000001)

            samples = []
            n = 20
            sq_diff = torch.zeros_like(mu0)
            for n in range(n):
                mu1, _, _ = model(xlr=x, reverse=True, eps=0.8)
                samples.append(mu0)
                sq_diff += (mu1 - mu0)**2

            # compute population standard deviation
            sigma = torch.sqrt(sq_diff / n)

            # create plot
            plt.figure()
            plt.imshow(sigma[0,...].permute(1,2,0).cpu().numpy(), cmap=color)
            plt.axis('off')
            # plt.show()
            plt.savefig(savedir_viz + '/sigma_{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
            plt.close()

            plt.figure()
            plt.imshow(mu0[0,...].permute(1,2,0).cpu().numpy(), cmap=cmap)
            plt.axis('off')
            # plt.show()
            plt.savefig(savedir_viz + '/mu0_{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
            plt.close()

            fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(1,7)
            # fig.suptitle('Y, Y_hat, mu, sigma')
            ax1.imshow(y[0,...].permute(1,2,0).cpu().numpy(), cmap=cmap)
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cax.set_axis_off()
            ax1.set_title('Ground Truth', fontsize=5)
            ax1.axis('off')
            ax2.imshow(mu0[0,...].permute(1,2,0).cpu().numpy(), cmap=cmap)
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cax.set_axis_off()
            ax2.set_title('Mean', fontsize=5)
            ax2.axis('off')
            ax3.imshow(samples[1][0,...].permute(1,2,0).cpu().numpy(), cmap=cmap)
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cax.set_axis_off()
            ax3.set_title('Sample 1', fontsize=5)
            ax3.axis('off')
            ax4.imshow(samples[2][0,...].permute(1,2,0).cpu().numpy(), cmap=cmap)
            divider = make_axes_locatable(ax4)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cax.set_axis_off()
            ax4.set_title('Sample 2', fontsize=5)
            ax4.axis('off')
            ax5.imshow(samples[2][0,...].permute(1,2,0).cpu().numpy(), cmap=cmap)
            divider = make_axes_locatable(ax5)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cax.set_axis_off()
            ax5.set_title('Sample 3', fontsize=5)
            ax5.axis('off')
            ax6.imshow(samples[2][0,...].permute(1,2,0).cpu().numpy(), cmap=cmap)
            divider = make_axes_locatable(ax6)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cax.set_axis_off()
            ax6.set_title('Sample 4', fontsize=5)
            ax6.axis('off')
            divider = make_axes_locatable(ax7)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            im7 = ax7.imshow(sigma[0,...].permute(1,2,0).cpu().numpy(), cmap='magma')
            cbar = fig.colorbar(im7,cmap='magma', cax=cax)
            cbar.ax.tick_params(labelsize=5)
            ax7.set_title('Std. Dev.', fontsize=5)
            ax7.axis('off')
            plt.tight_layout()
            plt.savefig(savedir_viz + '/std_multiplot_{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
            # plt.show()
            plt.close()

    return None

def compute_probs(model, generator, test_loader, exp_name, modelname, args):
    """
    For this experiment we compute the probability of the prediction
    compared to the ground truth under the estimated density. 
    """
    color = 'plasma'
    savedir_viz = "experiments/{}_{}_{}_{}x/snapshots/population_std/".format(exp_name, modelname, args.trainset, args.s)
    os.makedirs(savedir_viz, exist_ok=True)
    model.eval()
    cmap = 'viridis' if args.trainset == 'era5-TCW' else 'inferno'

    log_probs_cnf = []
    log_probs_gan = []
    with torch.no_grad():
        for batch_idx, item in enumerate(test_loader):

            y = item[0].to(args.device)
            x = item[1].to(args.device)

            y_unorm = item[2].to(args.device)
            x_unnorm = item[3].to(args.device)

            # CNF results
            mu05,_,_ = model(xlr=x, reverse=True, eps=1.0)

            z_hat, yhat_nll = model(x_hr=mu05, xlr=x)
            z, y_nll = model(x_hr=y, xlr=x)

            # print('Logprob y CNF:', yhat_nll.mean())
            # print('Logprob y_hat CNF', y_nll.mean())
            # print('Diff NLLs', (y_nll-yhat_nll).mean())

            # GAN results
            y_hat_gen = generator(x) # treat y_hat as the mean
            samples = []
            n = 50
            for n in range(n):
                sample = generator(x)
                samples.append(sample)

            stack = torch.stack(samples, dim=0).squeeze(1)
            std_gen = stack.std(dim=0)
            mean_gen = stack.mean(dim=0)

            # Compute probability. Use PyTorch library normal distribution for this. 
            normal = torch.distributions.normal.Normal(loc=y_hat_gen, scale=std_gen)
            log_p_y_gen = normal.log_prob(y)
            prob_y_gen = log_p_y_gen.mean().exp()

            log_p_yhat_gen = normal.log_prob(y_hat_gen)
            prob_yhat_gen = log_p_yhat_gen.mean().exp()

            # print('Logprob y GAN:', log_p_y_gen.mean())
            # print('Logprob y_hat GAN', log_p_yhat_gen.mean())
            # print('Diff Logprobs', (log_p_y_gen-log_p_yhat_gen).mean())

            # pdb.set_trace()

            log_probs_cnf.append(y_nll.mean().item())
            log_probs_gan.append(log_p_y_gen.mean().item())

            print('CNF NLL:', np.mean(log_probs_cnf))
            print('GAN NLL:', np.mean(log_probs_gan))

            # pdb.set_trace()

            # # create plot
            # plt.figure()
            # plt.imshow(sigma[0,...].permute(1,2,0).cpu().numpy(), cmap=color)
            # plt.axis('off')
            # # plt.show()
            # plt.savefig(savedir_viz + '/sigma_{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
            # plt.close()

            # plt.figure()
            # plt.imshow(mu0[0,...].permute(1,2,0).cpu().numpy(), cmap=cmap)
            # plt.axis('off')
            # # plt.show()
            # plt.savefig(savedir_viz + '/mu0_{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
            # plt.close()

            # fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(1,7)
            # # fig.suptitle('Y, Y_hat, mu, sigma')
            # ax1.imshow(y[0,...].permute(1,2,0).cpu().numpy(), cmap=cmap)
            # divider = make_axes_locatable(ax1)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # cax.set_axis_off()
            # ax1.set_title('Ground Truth', fontsize=5)
            # ax1.axis('off')
            # ax2.imshow(mu0[0,...].permute(1,2,0).cpu().numpy(), cmap=cmap)
            # divider = make_axes_locatable(ax2)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # cax.set_axis_off()
            # ax2.set_title('Mean', fontsize=5)
            # ax2.axis('off')
            # ax3.imshow(samples[1][0,...].permute(1,2,0).cpu().numpy(), cmap=cmap)
            # divider = make_axes_locatable(ax3)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # cax.set_axis_off()
            # ax3.set_title('Sample 1', fontsize=5)
            # ax3.axis('off')
            # ax4.imshow(samples[2][0,...].permute(1,2,0).cpu().numpy(), cmap=cmap)
            # divider = make_axes_locatable(ax4)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # cax.set_axis_off()
            # ax4.set_title('Sample 2', fontsize=5)
            # ax4.axis('off')
            # ax5.imshow(samples[2][0,...].permute(1,2,0).cpu().numpy(), cmap=cmap)
            # divider = make_axes_locatable(ax5)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # cax.set_axis_off()
            # ax5.set_title('Sample 3', fontsize=5)
            # ax5.axis('off')
            # ax6.imshow(samples[2][0,...].permute(1,2,0).cpu().numpy(), cmap=cmap)
            # divider = make_axes_locatable(ax6)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # cax.set_axis_off()
            # ax6.set_title('Sample 4', fontsize=5)
            # ax6.axis('off')
            # divider = make_axes_locatable(ax7)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # im7 = ax7.imshow(sigma[0,...].permute(1,2,0).cpu().numpy(), cmap='magma')
            # cbar = fig.colorbar(im7,cmap='magma', cax=cax)
            # cbar.ax.tick_params(labelsize=5)
            # ax7.set_title('Std. Dev.', fontsize=5)
            # ax7.axis('off')
            # plt.tight_layout()
            # plt.savefig(savedir_viz + '/std_multiplot_{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
            # # plt.show()
            # plt.close()

    return None


def test(model, test_loader, exp_name, modelname, logstep, args):

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    nll_list=[]

    avrg_fwd_time = []
    avrg_bw_time = []

    # storing metrics
    ssim0 = []#  [0] * args.bsz
    ssim05 =[]# [0] * args.bsz
    ssim08 =[]# [0] * args.bsz
    ssim1 = []# [0] * args.bsz

    psnr0 = []# [0] * args.bsz
    psnr05 = []# [0] * args.bsz
    psnr08 =[]# [0] * args.bsz
    psnr1 = []# [0] * args.bsz

    mse0 = [0] * args.bsz
    mse05 = [0] * args.bsz
    mse08 = [0] * args.bsz
    mse1 = [0] * args.bsz

    mae0 = []
    mae05 = [] # [0] * args.bsz
    mae08 = []  # [0] * args.bsz
    mae1 = [] # [0] * args.bsz

    rmse0 = [] 
    rmse05 = [] 
    rmse08 = []
    rmse1 = [] 

    mmd0 = [0] * args.bsz
    mmd0 = [0] * args.bsz
    mmd05 = [0] * args.bsz
    mmd08 = [0] * args.bsz
    mmd1 = [0] * args.bsz

    emd = [0] * args.bsz
    rmse = [0] * args.bsz

    crps = [] 

    mae0_plot = []
    ssim0_dens = []
    psnr0_dens = []
    rmse0_dens = []

    color = 'inferno' if args.trainset == 'era5-T2M' else 'viridis'
    savedir_viz = "experiments/{}_{}_{}_{}_{}x/snapshots/test/".format(exp_name, modelname, args.trainset, args.constraint, args.s)
    savedir_txt = 'experiments/{}_{}_{}/'.format(exp_name, modelname, args.trainset, args.constraint)

    os.makedirs(savedir_viz, exist_ok=True)
    os.makedirs(savedir_txt, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for batch_idx, item in enumerate(test_loader):

            y = item[0].to(args.device)
            x = item[1].to(args.device)

            y_unorm = item[2].to(args.device)
            x_unorm = item[3].unsqueeze(1).to(args.device)

            z, nll = model.forward(x_hr=y, xlr=x)

            # Generative loss
            nll_list.append(nll.mean().detach().cpu().numpy())

            # evaluate for different temperatures
            mu0, _, _ = model(xlr=x, reverse=True, eps=0)
            mu05, _, _ = model(xlr=x, reverse=True, eps=0.5)
            mu08, _, _ = model(xlr=x, reverse=True, eps=0.8)
            mu1, _, _ = model(xlr=x, reverse=True, eps=1.0)

            # Visual Metrics
            print("Evaluate Predictions on visual metrics... ")

            # # SSIM
            # current_ssim_mu0 = metrics.ssim(inv_scaler(mu0), y_unorm)
            # print('Current SSIM', current_ssim_mu0.item())
            # ssim0.append(current_ssim_mu0.cpu().numpy())
            # pd.Series(ssim0).hist()
            # plt.xlabel('SSIM')
            # plt.ylabel('nr samples')
            # plt.savefig(savedir_viz + '/ssim0_density.png', dpi=300, bbox_inches='tight')
            # plt.close()
            #
            # current_ssim_mu05 = metrics.ssim(inv_scaler(mu05), y_unorm)
            # ssim05.append(current_ssim_mu05.cpu())
            #
            # current_ssim_mu08 = metrics.ssim(inv_scaler(mu08), y_unorm)
            # ssim08.append(current_ssim_mu08.cpu())# = list(map(add, current_ssim_mu08, ssim08))
            #
            # current_ssim_mu1 = metrics.ssim(inv_scaler(mu1), y_unorm)
            # ssim1.append(current_ssim_mu1.cpu()) # = list(map(add, current_ssim_mu1, ssim1))
            #
            # # PSNR
            # current_psnr_mu0 = metrics.psnr(inv_scaler(mu0), y_unorm)
            # # current_psnr_mu0 = metrics.psnr(mu0, y)
            # psnr0.append(current_psnr_mu0) # list(map(add, current_psnr_mu0, psnr0))
            # print('Current PSNR', current_psnr_mu0)
            # # psnr0_dens.extend(current_psnr_mu0)
            # pd.Series(psnr0).hist()
            # plt.xlabel('PSNR')
            # plt.ylabel('nr samples')
            # plt.savefig(savedir_viz + '/psnr0_density.png', dpi=300, bbox_inches='tight')
            # plt.close()
            #
            # current_psnr_mu05 = metrics.psnr(inv_scaler(mu05), y_unorm)
            # # current_psnr_mu05 = metrics.psnr(mu05, y)
            # psnr05.append(current_psnr_mu05) #= list(map(add, current_psnr_mu05, psnr05))
            #
            # # current_psnr_mu08 = metrics.psnr(mu08, y)
            # current_psnr_mu08 = metrics.psnr(inv_scaler(mu08), y_unorm)
            # psnr08.append(current_psnr_mu08) # = list(map(add, current_psnr_mu08, psnr08))
            #
            # # current_psnr_mu1 = metrics.psnr(mu1, y)
            # current_psnr_mu1 = metrics.psnr(inv_scaler(mu1), y_unorm)
            # psnr1.append(current_psnr_mu1) # = list(map(add, current_psnr_mu1, psnr1))

            # MSE
            # current_mse0 = metrics.MSE(inv_scaler(mu0, min_value=y_unorm.min(), max_value=y_unorm.max()), y_unorm).detach().cpu().numpy()
            # # current_mse0 = metrics.MSE(mu0, y).detach().cpu().numpy()*100
            # mse0 = list(map(add, current_mse0, mse0))
            # print('Current MSE', current_mse0[0])
            #
            # current_mse05 = metrics.MSE(inv_scaler(mu05, min_value=y_unorm.min(), max_value=y_unorm.max()), y_unorm).detach().cpu().numpy()#*100
            # mse05 = list(map(add, current_mse05, mse05))
            #
            # current_mse08 = metrics.MSE(inv_scaler(mu08, min_value=y_unorm.min(), max_value=y_unorm.max()), y_unorm).detach().cpu().numpy()#*100
            # mse08 = list(map(add, current_mse08, mse08))
            #
            # current_mse1 = metrics.MSE(inv_scaler(mu1, min_value=y_unorm.min(), max_value=y_unorm.max()), y_unorm).detach().cpu().numpy()#*100
            # mse1 = list(map(add, current_mse1, mse1))

            # MAE
            mae0.append(metrics.MAE(inv_scaler(mu0, min_value=y_unorm.min(), max_value=y_unorm.max()), y_unorm).detach().cpu().numpy())
            print('Current MAE', np.mean(mae0))
            mae05.append(metrics.MAE(inv_scaler(mu05, min_value=y_unorm.min(), max_value=y_unorm.max()), y_unorm).detach().cpu().numpy())
            mae08.append(metrics.MAE(inv_scaler(mu08, min_value=y_unorm.min(), max_value=y_unorm.max()), y_unorm).detach().cpu().numpy())
            mae1.append(metrics.MAE(inv_scaler(mu1, min_value=y_unorm.min(), max_value=y_unorm.max()),y_unorm).detach().cpu().numpy())


            # RMSE
            rmse0.append(metrics.RMSE(inv_scaler(mu0, min_value=y_unorm.min(), max_value=y_unorm.max()), y_unorm).detach().cpu().numpy())
            print('Current RMSE', np.mean(rmse0))
            rmse05.append(metrics.RMSE(inv_scaler(mu05, min_value=y_unorm.min(), max_value=y_unorm.max()),y_unorm).detach().cpu().numpy())
            rmse08.append(metrics.RMSE(inv_scaler(mu08, min_value=y_unorm.min(), max_value=y_unorm.max()),y_unorm).detach().cpu().numpy())
            rmse1.append(metrics.RMSE(inv_scaler(mu1, min_value=y_unorm.min(), max_value=y_unorm.max()),y_unorm).detach().cpu().numpy())

            # MMD
            # current_mmd0 = metrics.MMD(inv_scaler(mu0,min_value=y_unorm.min(), max_value=y_unorm.max()),y_unorm)
            # mmd0 = list(map(add, current_mmd0.cpu().numpy(), mmd0))
            #
            # current_mmd05 = metrics.MMD(inv_scaler(mu05,min_value=y_unorm.min(), max_value=y_unorm.max()),y_unorm)
            # mmd05 = list(map(add, current_mmd05.cpu().numpy(), mmd05))
            #
            # current_mmd08 = metrics.MMD(inv_scaler(mu08,min_value=y_unorm.min(), max_value=y_unorm.max()),y_unorm)
            # mmd08 = list(map(add, current_mmd08.cpu().numpy(), mmd08))
            #
            # current_mmd1 = metrics.MMD(inv_scaler(mu1s,min_value=y_unorm.min(), max_value=y_unorm.max()),y_unorm)
            # mmd1 = list(map(add, current_mmd1.cpu().numpy(), mse1))


            imgs = []
            for i in range(8):
                img, _, _ = model(xlr=x, reverse=True, eps=0.8)
                imgs.append(inv_scaler(img))

            crps_stack = torch.stack(imgs, dim=1)
            crps.append(metrics.crps_ensemble(y_unorm, crps_stack))

            print('Current CRPS', np.mean(crps))

            # if batch_idx == 2:
            #     break

            print('Visualize results ...')

            # Visualize low resolution GT
            grid_low_res = torchvision.utils.make_grid(x[0:9, :, :, :].cpu(), normalize=True, nrow=3)
            plt.figure()
            plt.imshow(grid_low_res.permute(1, 2, 0)[:,:,0], cmap=color)
            plt.axis('off')
            # plt.title("Low-Res GT (train)")
            # plt.show()
            plt.savefig(savedir_viz + '/low_res_gt{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
            plt.close()

            # Visualize High-Res GT
            grid_high_res_gt = torchvision.utils.make_grid(y[0:9, :, :, :].cpu(), normalize=True, nrow=3)
            plt.figure()
            plt.imshow(grid_high_res_gt.permute(1, 2, 0)[:,:,0], cmap=color)
            plt.axis('off')
            # plt.title("High-Res GT")
            # plt.show()
            plt.savefig(savedir_viz + '/high_res_gt_{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
            plt.close()

            grid_mu0 = torchvision.utils.make_grid(mu0[0:9,:,:,:].cpu(), normalize=True, nrow=3)
            plt.figure()
            plt.imshow(grid_mu0.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
            plt.axis('off')
            # plt.title("Prediction at t (test), mu=0")
            plt.savefig(savedir_viz + "mu_0_logstep_{}_test.png".format(batch_idx), dpi=300,bbox_inches='tight')
            plt.close()

            grid_mu05 = torchvision.utils.make_grid(mu05[0:9,:,:,:].cpu(), normalize=True, nrow=3)
            plt.figure()
            plt.imshow(grid_mu0.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
            plt.axis('off')
            # plt.title("Prediction at t (test), mu=0.5")
            plt.savefig(savedir_viz + "mu_0.5_logstep_{}_test.png".format(batch_idx), dpi=300, bbox_inches='tight')
            plt.close()

            grid_mu08 = torchvision.utils.make_grid(mu08[0:9,:,:,:].cpu(), normalize=True, nrow=3)
            plt.figure()
            plt.imshow(grid_mu08.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
            plt.axis('off')
            # plt.title("Prediction at t (test), mu=0.8")
            plt.savefig(savedir_viz + "mu_0.8_logstep_{}_test.png".format(batch_idx), dpi=300,bbox_inches='tight')
            plt.close()

            grid_mu1 = torchvision.utils.make_grid(mu1[0:9,:,:,:].cpu(), normalize=True, nrow=3)
            plt.figure()
            plt.imshow(grid_mu1.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
            plt.axis('off')
            # plt.title("Prediction at t (test), mu=1.0")
            plt.savefig(savedir_viz + "mu_1_logstep_{}_test.png".format(batch_idx), dpi=300, bbox_inches='tight')
            plt.close()

            abs_err = torch.abs(mu08 - y)
            grid_abs_error = torchvision.utils.make_grid(abs_err[0:9,:,:,:].cpu(), normalize=True, nrow=3)
            plt.figure()
            plt.imshow(grid_abs_error.permute(1, 2, 0)[:,:,0], cmap=color)
            plt.axis('off')
            # plt.title("Abs Err")
            plt.savefig(savedir_viz + '/abs_err_{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
            plt.close()

            # write results to file:
            with open(savedir_txt + '/nll_runtimes.txt'.format(exp_name, modelname),'w') as f:
                f.write('Avrg NLL: %d \n'% np.mean(nll_list))
                f.write('Avrg fwd. runtime: %.2f \n'% np.mean(avrg_fwd_time))
                f.write('Avrg bw runtime: %.2f'% np.mean(avrg_bw_time))

    # compute average metric values over test set
    # avrg_ssim0 = np.mean(ssim0) # list(map(lambda x: x/len(test_loader), ssim0))
    # avrg_ssim05 = np.mean(ssim05) # list(map(lambda x: x/len(test_loader), ssim05))
    # avrg_ssim08 = np.mean(ssim08) # list(map(lambda x: x/len(test_loader), ssim08))
    # avrg_ssim1 = np.mean(ssim1) # list(map(lambda x: x/len(test_loader), ssim1))
    #
    # avrg_psnr0 = np.mean(psnr0) #list(map(lambda x: x/len(test_loader), psnr0))
    # avrg_psnr05 = np.mean(psnr05) #list(map(lambda x: x/len(test_loader), psnr05))
    # avrg_psnr08 = np.mean(psnr08) # list(map(lambda x: x/len(test_loader), psnr08))
    # avrg_psnr1 = np.mean(psnr1) # list(map(lambda x: x/len(test_loader), psnr1))

    # avrg_mse0 = list(map(lambda x: x/len(test_loader), mse0))
    # avrg_mse05 = list(map(lambda x: x/len(test_loader), mse05))
    # avrg_mse08 = list(map(lambda x: x/len(test_loader), mse08))
    # avrg_mse1 = list(map(lambda x: x/len(test_loader), mse1))

    avrg_mae0 = np.mean(mae0) # list(map(lambda x: x/len(test_loader), mae0))
    avrg_mae05 = np.mean(mae05) # list(map(lambda x: x/len(test_loader), mae05))
    avrg_mae08 = np.mean(mae08)# list(map(lambda x: x/len(test_loader), mae08))
    avrg_mae1 = np.mean(mae1) #list(map(lambda x: x/len(test_loader), mae1))

    avrg_rmse0 = np.mean(rmse0)
    avrg_rmse05 = np.mean(rmse05)
    avrg_rmse08 = np.mean(rmse08)
    avrg_rmse1 = np.mean(rmse1)

    # avrg_mmd0 = list(map(lambda x: x/len(test_loader), mmd0))
    # avrg_mmd05 = list(map(lambda x: x/len(test_loader), mmd05))
    # avrg_mmd08 = list(map(lambda x: x/len(test_loader), mmd08))
    # avrg_mmd1 = list(map(lambda x: x/len(test_loader),mmd1))

    # avrg_crps = list(map(lambda x: x/len(test_loader), crps0))

    # Write metric results to a file in case to recreate plots
    with open(savedir_txt + 'metric_results.txt','w') as f:


        # f.write('Avrg SSIM mu0:\n')
        # f.write("%f \n" % np.mean(avrg_ssim0))
        #
        # f.write('Avrg SSIM mu05:\n')
        # f.write("%f \n" %np.mean(avrg_ssim05))
        #
        # f.write('Avrg SSIM mu08:\n')
        # f.write("%f \n" %np.mean(avrg_ssim08))
        #
        # f.write('Avrg SSIM mu1:\n')
        # f.write("%f \n" %np.mean(avrg_ssim1))
        #
        #
        # f.write('Avrg PSNR mu0:\n')
        # f.write("%f \n" % np.mean(avrg_psnr0))
        #
        # f.write('Avrg PSNR mu05:\n')
        # f.write("%f \n" %np.mean(avrg_psnr05))
        #
        # f.write('Avrg PSNR mu08:\n')
        # f.write("%f \n" %np.mean(avrg_psnr08))
        #
        # f.write('Avrg PSNR mu1:\n')
        # f.write("%f \n" %np.mean(avrg_psnr1))


        # f.write('Avrg MSE mu0:\n')
        # f.write("%f \n" %np.mean(avrg_mse0))
        #
        # f.write('Avrg MSE mu05:\n')
        # f.write("%f \n" %np.mean(avrg_mse05))
        #
        # f.write('Avrg MSE mu08:\n')
        # f.write("%f \n" %np.mean(avrg_mse08))
        #
        # f.write('Avrg MSE mu1:\n')
        # f.write("%f \n" %np.mean(avrg_mse1))


        f.write('MAE mu0:\n')
        f.write("%f \n" %np.mean(mae0))
        f.write("%f \n" %np.std(mae0))

        f.write('Avrg MAE mu05:\n')
        f.write("%f \n" %np.mean(mae05))
        f.write("%f \n" %np.std(mae05))

        f.write('Avrg MAE mu08:\n')
        f.write("%f \n" %np.mean(mae08))
        f.write("%f \n" %np.std(mae08))

        f.write('Avrg MAE mu1:\n')
        f.write("%f \n" %np.mean(mae1))
        f.write("%f \n" %np.std(mae1))

        f.write('Avrg RMSE mu0:\n')
        f.write("%f \n" %np.mean(rmse0))
        f.write("%f \n" %np.std(rmse0))

        f.write('Avrg RMSE mu05:\n')
        f.write("%f \n" %np.mean(rmse05))
        f.write("%f \n" %np.std(rmse05))        

        f.write('Avrg RMSE mu08:\n')
        f.write("%f \n" %np.mean(rmse08))
        f.write("%f \n" %np.std(rmse08))

        f.write('Avrg RMSE mu1:\n')
        f.write("%f \n" %np.mean(rmse1))
        f.write("%f \n" %np.mean(rmse1))

        # f.write('Avrg MMD mu0:\n')
        # f.write("%f \n" %np.mean(avrg_mmd0))
        #
        # f.write('Avrg MMD mu05:\n')
        # f.write("%f \n" %np.mean(avrg_mmd05))
        #
        # f.write('Avrg MMD mu08:\n')
        # f.write("%f \n" %np.mean(avrg_mmd08))
        #
        # f.write('Avrg MMD mu1:\n')
        # f.write("%f \n" %np.mean(avrg_mmd1))
        #
        f.write('CRPS mu08:\n')
        f.write("%f \n" %np.mean(crps))
        f.write("%f \n" %np.std(crps))

    print("Average Test Neg. Log Probability Mass:", np.mean(nll_list))
    print("Average Fwd. runtime", np.mean(avrg_fwd_time))
    print("Average Bw runtime:", np.mean(avrg_bw_time))

    return None

def calibration_exp(model, test_loader, exp_name, modelname, logstep, args):
    """
    For this experiment we visualize the pixel distribution of the normalized
    and unnormalized counterpart images of the ground truth and the predicted
    super-resolved image to assess model calibration.
    """

    savedir_viz = "experiments/{}_{}_{}/snapshots/calibration_histograms/".format(exp_name, modelname, args.trainset)
    os.makedirs(savedir_viz, exist_ok=True)

    print("Generating Histograms ... ")

    model.eval()
    with torch.no_grad():
        for batch_idx, item in enumerate(test_loader):

            y = item[0].to(args.device)
            x = item[1].to(args.device)

            y_unorm = item[2].squeeze(1).to(args.device)
            x_unorm = item[3].squeeze(1).to(args.device)

            # super resolve image
            mu05, _, _ = model(xlr=x, reverse=True, eps=0.5)

            n_bins = int(y_unorm.max())
            fig, ((ax0,ax1)) = plt.subplots(nrows=1, ncols=2, figsize=(30, 10))

            colors = ['mediumorchid', 'coral']

            labelax0 = [r'Ground Truth',r'Prediction']
            y_fl = y.flatten().detach().cpu().numpy()
            mu05_fl = mu05.flatten().detach().cpu().numpy()
            values, bins, _ = ax0.hist(np.stack((y_fl, mu05_fl),axis=1), n_bins,
                                      density=True, histtype='step',color=colors,
                                      label=labelax0)
            area0 = sum(np.diff(bins)*values[0])
            print(area0)
            area1 = sum(np.diff(bins)*values[1])
            print(area1)
            ax0.set_xlabel('pixel values')
            ax0.set_ylabel('density')
            ax0.set_title('Normalized prediction vs. ground truth pixel distribution')
            ax0.legend(prop={'size': 10})

            labelax1 = ['Ground Truth', r'Prediction']
            mu05_unorm = inv_scaler(mu05, y_unorm.min(), y_unorm.max())
            # y_unorm = inv_scaler(y, y_unorm.min(), y_unorm.max())
            y_unorm_fl = y_unorm.flatten().detach().cpu().numpy()
            mu05_unorm_fl = mu05_unorm.flatten().detach().cpu().numpy()
            value, bins ,_= ax1.hist(np.stack((y_unorm_fl, mu05_unorm_fl),axis=1), n_bins, density=True, histtype='step',color=colors, label=labelax1)
            ax1.set_xlabel('pixel values')
            ax1.set_ylabel('density')
            ax1.set_title('Unormalized prediction vs. ground truth pixel distribution')
            ax1.legend(prop={'size': 10})

            # plt.show()

            # labelax2 = ['abs diff normalized']
            # newcolor = ['deepskyblue', 'palegreen']
            # diff = torch.abs(y-mu05).flatten().detach().cpu().numpy()
            # diff_unorm = torch.abs(y_unorm-mu05_unorm).flatten().detach().cpu().numpy()
            # ax2.hist(np.stack((diff, diff_unorm),axis=1), n_bins, density=True, histtype='barstacked',color=newcolor, label=labelax2)
            # ax2.set_xlabel('nr of bins')
            # ax2.set_ylabel('pixel values')
            # ax2.set_title('Absolute difference distributions')
            # ax2.legend(prop={'size': 10})
            # plt.show()

            fig.savefig(savedir_viz + '/histogram_multiplot_{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":

    print(torch.cuda.device_count())

    # NOTE: when executing code, make sure you enable the --testmode flag !!!

    # Load testset
    _, _, test_loader, args = dataloading.load_data(args)

    in_channels = next(iter(test_loader))[0].shape[1]
    args.height, args.width = next(iter(test_loader))[0].shape[2], next(iter(test_loader))[0].shape[3]

    args.device = "cuda"

    # Load Model
    # temperature 2x
    # modelname = 'model_epoch_2_step_7500'
    # modelpath = '/home/christina/Documents/clim-var-ds-cnf/experiments/srflow_era5-T2M_2023_11_07_12_13_13_2x/models/{}.tar'.format(modelname)

    # temperature 4x
    # modelname = 'model_epoch_5_step_19750'
    # modelpath = '/home/christina/Documents/clim-var-ds-cnf/experiments/flow-3-level-2-k_model_epoch_5_step_19750_era5-T2M_4x/models/{}.tar'.format(modelname)

    # watercontent 2x upsampling
    # modelname = 'model_epoch_5_step_6500'
    # modelpath = '/home/mila/c/christina.winkler/clim-var-ds-cnf/runs/srflow_era5-TCW_2023_11_03_07_15_17_2x/model_checkpoints/{}.tar'.format(modelname)

    # watercontent 2x upsampling + perc loss
    # modelname = 'model_epoch_1_step_5000'
    # modelpath = '/home/mila/c/christina.winkler/clim-var-ds-cnf/runs/srflow_era5-TCW_2023_11_09_09_30_46_2x/model_checkpoints/{}.tar'.format(modelname)

    # 2x upsampling water content + addDS
    # modelname = 'model_epoch_4_step_10500'
    # modelpath = '/home/mila/c/christina.winkler/clim-var-ds-cnf/runs/srflow_era5-TCW_addDS__2023_11_10_10_53_58_2x/model_checkpoints/{}.tar'.format(modelname)
    
    # 4x upsampling water content + addDS
    # modelname = 'model_epoch_8_step_10250'
    # modelpath =  '/home/mila/c/christina.winkler/clim-var-ds-cnf/runs/srflow_era5-TCW_addDS__2023_11_13_15_57_11_4x/model_checkpoints/{}.tar'.format(modelname)

    # 2x watercontent None
    modelname = 'model_epoch_8_step_10500'
    modelpath = '/home/mila/c/christina.winkler/clim-var-ds-cnf/runs/srflow_era5-TCW_None__2023_11_13_15_57_17_2x/model_checkpoints/{}.tar'.format(modelname)

    # modelname = 'model_epoch_4_step_29000'
    # modelpath = '/home/christina/Documents/clim-var-ds-cnf/runs/srflow_era5-TCW_2023_10_23_13_00_15/model_checkpoints/{}.tar'.format(modelname)

    # 4x upsampling watercontent None
    # modelname = 'model_epoch_18_step_23000'
    # modelpath = '/home/mila/c/christina.winkler/clim-var-ds-cnf/runs/srflow_era5-TCW_None__2023_11_10_11_27_23_4x/model_checkpoints/{}.tar'.format(modelname)

    # 4x upsampling water content + perc loss
    # modelname = 'model_epoch_6_step_8750'
    # modelpath = '/home/mila/c/christina.winkler/clim-var-ds-cnf/runs/srflow_era5-TCW_2023_11_10_06_38_01_4x/model_checkpoints/{}.tar'.format(modelname)

    # 4x upsampling water content + softmax
    # modelname = 'model_epoch_9_step_12500'
    # modelpath = '/home/mila/c/christina.winkler/clim-var-ds-cnf/runs/srflow_era5-TCW_softmax__2023_11_10_11_19_16_4x/model_checkpoints/{}.tar'.format(modelname)

    # 2x watercontent
    gen_modelname = 'generator_epoch_2_step_7250'
    gen_modelpath = '/home/mila/c/christina.winkler/clim-var-ds-cnf/runs/srgan_stoch_era5-TCW_None__2023_11_16_05_31_20_2x/model_checkpoints/{}.tar'.format(gen_modelname)

    model = srflow.SRFlow((in_channels, args.height, args.width), args.filter_size, args.L, args.K,
                           args.bsz, args.s, args.nb, args.condch, args.nbits, args.noscale, args.noscaletest)

    # init model
    generator = srgan2_stochastic.RRDBNet(in_channels, out_nc=1, nf=128, s=args.s, nb=5)
    ckpt = torch.load(gen_modelpath)
    generator.load_state_dict(ckpt['model_state_dict'])
    generator = generator.to(args.device)
    generator.eval()

    print(torch.cuda.device_count())
    ckpt = torch.load(modelpath, map_location='cuda:0')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    params = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print('Nr of Trainable Params {}:  '.format(args.device), params)
    model = model.to(args.device)

    exp_name = "flow-{}-level-{}-k".format(args.L, args.K)
    compute_probs(model, generator, test_loader, exp_name, modelname, args)
    # plot_std(model, test_loader, exp_name, modelname, args)
    # calibration_exp(model, test_loader, exp_name, modelname, -99999, args)

    print("Evaluate on test split ...")
    # test(model, test_loader, exp_name, modelname, -99999, args)
