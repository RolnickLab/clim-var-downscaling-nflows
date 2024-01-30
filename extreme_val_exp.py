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

from models.architectures import srgan, srgan2
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

def extreme_val_exp(model, test_loader, exp_name, modelname, logstep, args):



    return None

if __name__ == "__main__":

    # Load testset
    _, _, test_loader, args = dataloading.load_data(args)
    in_channels = next(iter(test_loader))[0].shape[1]
    args.height, args.width = next(iter(test_loader))[0].shape[2], next(iter(test_loader))[0].shape[3]

    # init model
    generator = srgan2.RRDBNet(in_channels, out_nc=1, nf=128, s=args.s, nb=5)
    # disc_net = srgan.Discriminator(in_channels)

    # Load condNF model
    # 2x watercontent
    # 4x watercontent
    cnf = srflow.SRFlow((in_channels, args.height, args.width), args.filter_size, args.L, args.K,
                           args.bsz, args.s, args.nb, args.condch, args.nbits, args.noscale, args.noscaletest)

    print(torch.cuda.device_count())
    ckpt = torch.load(modelpath, map_location='cuda:0')
    cnf.load_state_dict(ckpt['model_state_dict'])
    cnf.eval()   


    # Load GAN model
    # 2x watercontent
    # modelname = 'generator_epoch_3_step_9500'
    # gen_modelpath = '/home/mila/c/christina.winkler/clim-var-ds-cnf/runs/srgan_era5-TCW_2023_11_08_11_16_25_2x/model_checkpoints/{}.tar'.format(modelname)

    # 4x watercontent
    modelname = 'generator_epoch_6_step_4000'
    gen_modelpath = '/home/mila/c/christina.winkler/clim-var-ds-cnf/runs/srgan_era5-TCW_2023_11_09_06_49_00_4x/model_checkpoints/{}.tar'.format(modelname)

    ckpt = torch.load(gen_modelpath)
    generator.load_state_dict(ckpt['model_state_dict'])
    generator.eval()


    exp_name = "srgan-{}-{}x".format(args.trainset, args.s)
    test(generator, test_loader, exp_name, modelname, args)
