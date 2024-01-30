import numpy as np
import torch
import torch.nn as nn
import random

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

from models.architectures import srgan, srgan2, srgan2_stochastic
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


if torch.cuda.is_available():
    args.device = torch.device("cuda")
    args.num_gpus = torch.cuda.device_count()
    args.parallel = False

else:
    args.device = "cpu"


def inv_scaler(x, min_value=0, max_value=100):
    x = x * (max_value - min_value) + min_value
    return x

def plot_std(model):
    """
    For this experiment we visualize the super-resolution space for a single
    low-resolution image and its possible HR target predictions. We visualize
    the standard deviation of these predictions from the mean of the model.
    """
    color = 'plasma'
    savedir_viz = "experiments/{}_{}_{}/snapshots/population_std/".format(exp_name, modelname, args.trainset)
    os.makedirs(savedir_viz, exist_ok=True)
    model.eval()
    cmap = 'viridis' if args.trainset == 'era5-TCW' else 'inferno'
    with torch.no_grad():
        for batch_idx, item in enumerate(test_loader):

            y = item[0].to(args.device)
            x = item[1].to(args.device)

            y_unorm = item[2].to(args.device)
            x_unnorm = item[3].to(args.device)

            mu0 = model(x)

            samples = []
            n = 20
            sq_diff = torch.zeros_like(mu0)
            for n in range(n):
                mu1 = model(x)
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

            fig, (ax1, ax3, ax4, ax5, ax6, ax7) = plt.subplots(1,6)
            # fig.suptitle('Y, Y_hat, mu, sigma')
            ax1.imshow(y[0,...].permute(1,2,0).cpu().numpy(), cmap=cmap)
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cax.set_axis_off()
            ax1.set_title('Ground Truth', fontsize=5)
            ax1.axis('off')
            # ax2.imshow(mu0[0,...].permute(1,2,0).cpu().numpy(), cmap=cmap)
            # divider = make_axes_locatable(ax2)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # cax.set_axis_off()
            # ax2.set_title('Mean', fontsize=5)
            # ax2.axis('off')
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

def test(model, test_loader, exp_name, modelname, args):
    
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        nll_list=[]

        avrg_fwd_time = []
        avrg_bw_time = []

        mse = []
        mae = []
        rmse = [] 
        crps = [] 

        color = 'inferno' if args.trainset == 'era5-T2M' else 'viridis'
        savedir_viz = "experiments/{}_{}_{}/snapshots/test/".format(exp_name, modelname, args.trainset)
        savedir_txt = 'experiments/{}_{}_{}/'.format(exp_name, modelname, args.trainset)

        os.makedirs(savedir_viz, exist_ok=True)
        os.makedirs(savedir_txt, exist_ok=True)

        generator.eval()

        mse_loss_list = []
        mse_loss = nn.MSELoss()

        with torch.no_grad():
            for batch_idx, item in enumerate(test_loader):

                y = item[0].to(args.device)
                x = item[1].to(args.device)

                y_unorm = item[2].to(args.device)
                x_unorm = item[3].to(args.device)

                fake_img = generator(x)

                g_loss = mse_loss(fake_img, y)
                    
                # Generative loss
                mse_loss_list.append(g_loss.mean().detach().cpu().numpy())


                print("Evaluate Predictions on visual metrics... ")

                # MAE
                mae.append(metrics.MAE(inv_scaler(fake_img, min_value=y_unorm.min(), max_value=y_unorm.max()), y_unorm).detach().cpu().numpy())
                    
                print('Current MAE', np.mean(mae),mae[-1])

                mse.append(metrics.MSE(inv_scaler(fake_img, min_value=y_unorm.min(), max_value=y_unorm.max()), y_unorm).mean().detach().cpu().numpy())
                print('Current MSE', np.mean(mse), mse[-1])
            
                # RMSE
                rmse.append(metrics.RMSE(inv_scaler(fake_img, min_value=y_unorm.min(), max_value=y_unorm.max()), y_unorm).detach().cpu().numpy())
                print('Current RMSE', np.mean(rmse), rmse[-1])

                # CRPS
                imgs = []
                for i in range(8):
                    img = generator(x)
                    imgs.append(inv_scaler(img, min_value=y_unorm.min(), max_value=y_unorm.max()))

                crps_stack = torch.stack(imgs, dim=1)

                crps.append(metrics.crps_ensemble(y_unorm, crps_stack))

                print('Current CRPS', np.mean(crps))

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

                grid_pred = torchvision.utils.make_grid(fake_img[0:9,:,:,:].cpu(), normalize=True, nrow=3)
                plt.figure()
                plt.imshow(grid_pred.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
                plt.axis('off')
                # plt.title("Prediction (test),")
                plt.savefig(savedir_viz + "y_hat_logstep_{}_test.png".format(batch_idx), dpi=300,bbox_inches='tight')
                plt.close()

                abs_err = torch.abs(fake_img - y)
                grid_abs_error = torchvision.utils.make_grid(abs_err[0:9,:,:,:].cpu(), normalize=True, nrow=3)
                plt.figure()
                plt.imshow(grid_abs_error.permute(1, 2, 0)[:,:,0], cmap=color)
                plt.axis('off')
                # plt.title("Abs Err")
                plt.savefig(savedir_viz + '/abs_err_{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
                plt.close()

        # Write metric results to a file in case to recreate plots
        with open(savedir_txt + 'metric_results.txt','w') as f:
            f.write('MAE:\n')
            f.write("%f \n" %np.mean(mae))
            f.write("%f \n" %np.std(mae))

            f.write('RMSE:\n')
            f.write("%f \n" %np.mean(rmse))
            f.write("%f \n" %np.std(rmse))

            f.write('CRPS:\n')
            f.write("%f \n" %np.mean(crps))
            f.write("%f \n" %np.std(crps))
        
        return None

def calibration_exp(model, test_loader, exp_name, modelname, args):
    """
    For this experiment we visualize the pixel distribution of the normalized
    and unnormalized counterpart images of the ground truth and the predicted
    super-resolved image to assess model calibration.
    """

    savedir_viz = "experiments/{}_{}_{}_x/snapshots/calibration_histograms/".format(exp_name, modelname, args.trainset, args.s)
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
            y_hat = model(x)

            mu05_unorm = inv_scaler(y_hat, y_unorm.min(), y_unorm.max())
            n_bins = int(y_unorm.max())
            fig, ((ax0,ax1)) = plt.subplots(nrows=1, ncols=2, figsize=(30, 10))

            colors = ['mediumorchid', 'coral']

            labelax0 = [r'Ground Truth',r'Prediction']
            y_fl = y.flatten().detach().cpu().numpy()
            mu05_fl = y_hat.flatten().detach().cpu().numpy()
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

    # Load testset
    _, _, test_loader, args = dataloading.load_data(args)
    in_channels = next(iter(test_loader))[0].shape[1]
    args.height, args.width = next(iter(test_loader))[0].shape[2], next(iter(test_loader))[0].shape[3]

    # init model
    generator = srgan2_stochastic.RRDBNet(in_channels, out_nc=1, nf=128, s=args.s, nb=5)
    # disc_net = srgan.Discriminator(in_channels)

    # Load model
    # 2x watercontent
    modelname = 'generator_epoch_2_step_7250'
    gen_modelpath = '/home/mila/c/christina.winkler/clim-var-ds-cnf/runs/srgan_stoch_era5-TCW_None__2023_11_16_05_31_20_2x/model_checkpoints/{}.tar'.format(modelname)

    # 4x watercontent
    # modelname = 'generator_epoch_5_step_12750'
    # gen_modelpath = '/home/mila/c/christina.winkler/clim-var-ds-cnf/runs/srgan_stoch_era5-TCW_None__2023_11_16_05_32_22_4x/model_checkpoints/{}.tar'.format(modelname)

    ckpt = torch.load(gen_modelpath)
    generator.load_state_dict(ckpt['model_state_dict'])
    generator.eval()

    params = sum(x.numel() for x in generator.parameters() if x.requires_grad)
    print('Nr of Trainable Params {}:  '.format(args.device), params)
    generator = generator.to(args.device)

    exp_name = "srgan-{}-{}x".format(args.trainset, args.s)
    # plot_std(generator)
    calibration_exp(generator, test_loader, exp_name, modelname, args)
    # test(generator, test_loader, exp_name, modelname, args)

