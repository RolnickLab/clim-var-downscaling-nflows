from os.path import exists, join
from os import listdir
import torch.utils.data as data_utils
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from PIL import Image
import numpy as np
import math
import torch
import random
import sys
import pdb
import os
import xarray as xr

from data.era5_temp_dataset import ERA5T2MData
from data.era5_watercontent_dset import ERA5WTCData

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os
sys.path.append("../")

def load_era5_TCW(args):

    print("Loading ERA5 TCW ...")

    train_data = ERA5WTCData(data_path=args.datadir + '/era5_tcw/train', s=args.s)
    val_data = ERA5WTCData(data_path=args.datadir + '/era5_tcw/val', s=args.s)
    test_data = ERA5WTCData(data_path=args.datadir + '/era5_tcw/test', s=args.s)

    train_loader = data_utils.DataLoader(train_data, args.bsz, shuffle=True,
                                         drop_last=True)
    val_loader = data_utils.DataLoader(val_data, args.bsz, shuffle=True,
                                       drop_last=True)
    test_loader = data_utils.DataLoader(test_data, args.bsz, shuffle=False,
                                        drop_last=False)

    return train_loader, val_loader, test_loader, args

def load_era5_T2M(args):

    print("Loading ERA5 T2M ...")

    dpath = '/home/mila/c/christina.winkler/scratch/data/assets/ftp.bgc-jena.mpg.de/pub/outgoing/aschall/data.zarr'

    dataset = ERA5T2MData(data_path=dpath, s=args.s)

    n_train_samples = int(len(dataset) // (1/0.7))
    n_val_samples = int(len(dataset) // (1/0.2))
    n_test_samples = int(len(dataset) // (1/0.1))

    train_idcs = [i for i in range(0, n_train_samples)]
    val_idcs = [i for i in range(0, n_val_samples)]
    test_idcs = [i for i in range(0, n_test_samples)]

    trainset = torch.utils.data.Subset(dataset, train_idcs)
    valset = torch.utils.data.Subset(dataset, val_idcs)
    testset = torch.utils.data.Subset(dataset, test_idcs)

    train_loader = data_utils.DataLoader(trainset, args.bsz, shuffle=True,
                                         drop_last=True)
    val_loader = data_utils.DataLoader(valset, args.bsz, shuffle=True,
                                       drop_last=True)
    test_loader = data_utils.DataLoader(testset, args.bsz, shuffle=False,
                                        drop_last=True)

    return train_loader, val_loader, test_loader, args

def load_data(args):

    if args.trainset == "era5-T2M":
        return load_era5_T2M(args)

    elif args.trainset == "era5-TCW":
        return load_era5_TCW(args)

    else:
        raise ValueError("Dataset not available. Check for typos!")


# if __name__ == "__main__":
