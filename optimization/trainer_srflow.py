from datetime import datetime
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.optim as optim
import os
import json
from skimage.transform import resize

# Utils
from utils import utils
import numpy as np
import random
import pdb
import torchvision
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from models.architectures.conv_lstm import *
from optimization.validation_srflow import validate
from typing import Tuple, Callable

import wandb
os.environ["WANDB_SILENT"] = "true"
import sys
sys.path.append("../../")

# seeding only for debugging
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class MinMaxScaler:
    def __call__(self, x, max_value, min_value):
        values_range: Tuple[int, int] = (-1, 1)
        x = (x - min_value) / (max_value - min_value)
        return x * (values_range[1] - values_range[0]) + values_range[0]

def inv_scaler(x):
    min_value = 0
    max_value = 100
    return x * (max_value - min_value) + min_value

def trainer(args, train_loader, valid_loader, model,
            device='cpu', needs_init=True):

    cmap = 'viridis' if args.trainset == 'era5-TCW' else 'inferno'

    # wandb.init(project="arflow", config=config_dict)
    args.experiment_dir = os.path.join('runs',
                                        args.modeltype + '_' + args.trainset + '_' + datetime.now().strftime("_%Y_%m_%d_%H_%M_%S") +'_'+ str(args.s)+'x')

    os.makedirs(args.experiment_dir, exist_ok=True)
    config_dict = vars(args)
    with open(args.experiment_dir + '/configs.txt', 'w') as f:
        for key, value in config_dict.items():
            f.write('%s:%s\n' % (key, value))

    # set viz dir
    viz_dir = "{}/snapshots/trainset/".format(args.experiment_dir)
    os.makedirs(viz_dir, exist_ok=True)

    writer = SummaryWriter("{}".format(args.experiment_dir))
    prev_nll_epoch = np.inf
    logging_step = 0
    step = 0
    bpd_valid = 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=2 * 10 ** 5,
                                                gamma=0.5)
    state=None

    model.to(device)
    scaler = MinMaxScaler()
    l1 = nn.L1Loss()
    metric_dict = {'MSE': [], 'RMSE': [], 'MAE': []}
    params = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print('Nr of Trainable Params on {}:  '.format(device), params)

    # add hyperparameters to tensorboardX logger
    writer.add_hparams({'lr': args.lr, 'bsize':args.bsz, 'Flow Steps':args.K,
                        'Levels':args.L}, {'nll_train': - np.inf})


    if torch.cuda.device_count() > 1 and args.train:
        print("Running on {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
        args.parallel = True

    for epoch in range(args.epochs):
        for batch_idx, item in enumerate(train_loader):

            y = item[0].to(device)
            x = item[1].to(device)
            # y_unorm = item[2].to(device)
            # x_unorm = item[3].to(device)

            model.train()
            optimizer.zero_grad()
            
            # # We need to init the underlying module in the dataparallel object
            # For ActNorm layers.
            if needs_init and torch.cuda.device_count() > 1:
                bsz_p_gpu = args.bsz // torch.cuda.device_count()
                _, _ = model.module.forward(x_hr=y[:bsz_p_gpu],
                                            xlr=x[:bsz_p_gpu],
                                            logdet=0)

            # forward loss
            z, nll = model.forward(x_hr=y, xlr=x)

            writer.add_scalar("nll_train", nll.mean().item(), step)
            # wandb.log({"nll_train": nll.mean().item()}, step)

            # Compute gradients
            loss = nll
            loss.mean().backward()

            # Update model parameters using calculated gradients
            optimizer.step()
            scheduler.step()
            step = step + 1

            print("[{}] Epoch: {}, Train Step: {:01d}/{}, Bsz = {}, NLL {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    epoch, step,
                    args.max_steps,
                    args.bsz,
                    nll.mean().item()))

            if step % args.log_interval == 0:

                with torch.no_grad():

                    # if hasattr(model, "module"):
                    #     model_without_dataparallel = model.module
                    # else:
                    #     model_without_dataparallel = model

                    model.eval()

                    # Visualize low resolution GT
                    grid_low_res = torchvision.utils.make_grid(x[0:9, :, :, :].cpu(), normalize=True, nrow=3)
                    plt.figure()
                    plt.imshow(grid_low_res.permute(1, 2, 0)[:,:,0], cmap=cmap)
                    plt.axis('off')
                    plt.title("Low-Res GT (train)")
                    # plt.show()
                    plt.savefig(viz_dir + '/low_res_gt{}.png'.format(step), dpi=300, bbox_inches='tight')
                    plt.close()

                    # Visualize High-Res GT
                    grid_high_res_gt = torchvision.utils.make_grid(y[0:9, :, :, :].cpu(), normalize=True, nrow=3)
                    plt.figure()
                    plt.imshow(grid_high_res_gt.permute(1, 2, 0)[:,:,0], cmap=cmap)
                    plt.axis('off')
                    plt.title("High-Res GT")
                    # plt.show()
                    plt.savefig(viz_dir + '/high_res_gt_{}.png'.format(step), dpi=300, bbox_inches='tight')
                    plt.close()

                     # Super-Resolving low-res
                    y_hat, logdet, logpz = model(xlr=x, reverse=True, eps=0.8)
                    # print(y_hat.max(), y_hat.min(), y.max(), y.min())
                    grid_y_hat = torchvision.utils.make_grid(y_hat[0:9, :, :, :].cpu(), normalize=False, nrow=3)
                    plt.figure()
                    plt.imshow(grid_y_hat.permute(1, 2, 0)[:,:,0], cmap=cmap)
                    plt.axis('off')
                    plt.title("Y hat")
                    plt.savefig(viz_dir + '/y_hat_mu08_{}.png'.format(step), dpi=300,bbox_inches='tight')
                    # plt.show()
                    plt.close()

                    abs_err = torch.abs(y_hat - y)
                    grid_abs_error = torchvision.utils.make_grid(abs_err[0:9,:,:,:].cpu(), normalize=True, nrow=3)
                    plt.figure()
                    plt.imshow(grid_abs_error.permute(1, 2, 0)[:,:,0], cmap=cmap)
                    plt.axis('off')
                    plt.title("Abs Err")
                    plt.savefig(viz_dir + '/abs_err_{}.png'.format(step), dpi=300,bbox_inches='tight')
                    # plt.show()
                    plt.close()


            if step % args.val_interval == 0:
                print('Validating model ... ')
                metric_dict, nll_valid = validate(model,
                                     valid_loader,
                                     metric_dict,
                                     args.experiment_dir,
                                     "{}".format(step),
                                     args)

                writer.add_scalar("nll_valid",
                                  nll_valid.mean().item(),
                                  logging_step)

                # save checkpoint only when nll lower than previous model
                if nll_valid < prev_nll_epoch:
                    PATH = args.experiment_dir + '/model_checkpoints/'
                    os.makedirs(PATH, exist_ok=True)
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': nll_valid.mean()}, PATH+ f"model_epoch_{epoch}_step_{step}.tar")
                    prev_nll_epoch = nll_valid

            logging_step += 1

            if step == args.max_steps:
                break

        if step == args.max_steps:
        #     print("Done Training for {} mini-batch update steps!".format(args.max_steps)
        #     )
        #
        #     if hasattr(model, "module"):
        #         model_without_dataparallel = model.module
        #     else:
        #         model_without_dataparallel = model

            utils.save_model(model,
                             epoch, optimizer, args, time=True)

            print("Saved trained model :)")
            wandb.finish()
            break
