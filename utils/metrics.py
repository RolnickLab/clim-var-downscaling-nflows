from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from utils.metrics_esrgan import calculate_ssim, calculate_psnr, bgr2ycbcr
from torchmetrics.functional import structural_similarity_index_measure
import torchvision
import numpy as np
import pdb
import torch
import torch.nn as nn

def ssim(im1, im2):
    """
    Computes the similarity index between two images measuring
    the similarity between the two images. SSIM has a maximum value of 1, indicating that the two signals are perfectly structural similar
    while a value of 0 indicates no structural similarity.

    Args:
        im1 (tensor):
        im2 (tensor):
    Returns:
        ssim (value):
    """
    ssim = []
    # # Compute ssim over samples in mini-batch
    # for i in range(im1.shape[0]):
    #     ssim.append(calculate_ssim(im1[i, :, :, :], im2[i, :, :, :]))
    #
    # return np.mean(ssim)
    return structural_similarity_index_measure(im1,im2)

def psnr(img1, img2, max=100):
    img1 = img1.detach().cpu().numpy().astype(np.float64)
    img2 = img2.detach().cpu().numpy().astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max/np.sqrt(mse))

def RMSE(yhat,y):
    _,_,h,w=y.shape
    sq_diff = (yhat-y)**2
    sum = sq_diff.sum(dim=[1,2,3])
    return torch.sqrt(sum / (h*w)).mean()

def MSE(y_hat, y):
    _,_,h,w=y.shape
    diff = (y_hat - y)**2
    sum = (diff).sum(dim=[1,2,3])
    return sum/(h*w)

def MAE(y_hat, y):
    mae = nn.L1Loss()
    return mae(y_hat,y)

def nrmse(im1, im2):
    """

    Args:

    Returns:

    """
    im1 = im1.permute(0, 3, 2, 1).contiguous().cpu().detach().numpy()
    im2 = im2.permute(0, 3, 2, 1).contiguous().cpu().detach().numpy()
    nrmse = []
    # Compute ssim over samples in mini-batch
    for i in range(im1.shape[0]):
        nrmse.append(compare_nrmse(im1[i, :, :, :], im2[i, :, :, :]))
    return np.mean(nrmse)

def MMD(x, y, kernel= lambda x, y: rbf_kernel(x, y, gamma=1.0), sqrtMMD=False):
    """
    :param x:
    :param y:
    :param gamma:
    :param sqrtMMD:
    :return:
    """
    result = kernel(x, x).mean(dim=[1,2,3]) - 2 * kernel(x, y).mean(dim=[1,2,3]) + kernel(y, y).mean(dim=[1,2,3])
    if sqrtMMD == 2:
        result = torch.sqrt(result)
    elif sqrtMMD == 3:
        result = result ** (1 / 3)
    elif sqrtMMD == 4:
        result = result ** (1 / 4)
    return result

def rbf_kernel(x, y, gamma=None):
    """Radial basis (Gaussian) kernel between x and y. Exp(-gamma*|x-y|^2).

    Input:
        x: Tensor (n_samples_x, n_features).
        y: Tensor (n_samples_y, n_features).
        gamma: Default: 1.0 / n_features. Gamma can also be a list, then the cost function is
                                evaluated overall entries of those lists for gamma realizations
    """
    gamma = gamma or (1.0 / x.shape[1])
    if not isinstance(gamma, list):
        return torch.exp(-gamma * euclidean_distances(x, y, squared=True))
    else:
        reVal = torch.zeros((x.shape[0], y.shape[0]))
        euclDist = euclidean_distances(x, y, squared=True)
        for g in gamma:
            reVal = reVal + torch.exp(-g * euclDist)
        return reVal

def euclidean_distances(x, y, squared=False):
    """Euclidean distance.

    Input:
        x: Tensor.
        y: Tensor.
        squared: Compute squared distance? Default: False.

    Returns:
        Tensor (n_samples_x, n_samples_y).
    """
    # (a-b)^2 = -2ab + a^2 + b^2
    # distances = -2 * x @ y.mT

    # pdb.set_trace()
    # distances += torch.einsum('ij,ij->i', x, x)[:, None]
    # distances += torch.einsum('ij,ij->i', y, y)[None, :]
    # distances += torch.cdist(x,y,p=2)
    distances = ((x-y)**2)

    if squared:
        return distances
    else:
        return distances.sqrt()

def crps_ensemble(observation, forecasts):
    # explanation: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003120
    # forceasts contains n predicted frames
    _,_,h,w = observation.shape
    observation = observation.detach().cpu().numpy()
    forecasts = forecasts.detach().cpu().numpy()

    fc = forecasts.copy()
    fc.sort(axis=1)
    obs = observation
    fc_below = fc<obs[:,None,...]
    crps = np.zeros_like(obs)
    for i in range(fc.shape[1]):
        below = fc_below[:,i,...]
        weight = ((i+1)**2 - i**2) / fc.shape[-1]**2
        crps[below] += weight * (obs[below]-fc[:,i,...][below])

    for i in range(fc.shape[1]-1,-1,-1):
        above  = ~fc_below[:,i,...]
        k = fc.shape[1]-1-i
        weight = ((k+1)**2 - k**2) / fc.shape[1]**2
        crps[above] += weight * (fc[:,i,...][above]-obs[above])
    crps = np.sum(crps, axis=1)
    crps = np.sum(crps, axis=1)
    crps = np.sum(crps, axis=1)
    return crps / (h*w) # np.mean(crps)

# def EMD(x, y):
#     """Computes EMD / Wasserstein Distance"""
#     emd = neuralnet_pytorch.metrics.emd_loss(x,y)
#     return emd
