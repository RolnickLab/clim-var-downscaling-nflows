from typing import Tuple, Callable
import xarray as xr
from dataclasses import dataclass
import numpy as np
import pdb
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
from torchvision import transforms, datasets

class XRToTensor:
    def __call__(self, zarr_array):
        # unsqueeze to have a tensor of shape (CxWxH)
        return torch.from_numpy(zarr_array.values).unsqueeze(0)


@dataclass
class MinMaxScaler:
    max_value: float = 315.91873
    min_value: float = 241.22385
    values_range: Tuple[int, int] = (-1, 1)

    def __call__(self, x):
        x = (x - self.min_value) / (self.max_value - self.min_value)
        return x * (self.values_range[1] - self.values_range[0]) + self.values_range[0]

@dataclass
class InverseMinMaxScaler:
    max_value: float = 315.91873
    min_value: float = 241.22385
    values_range: Tuple[int, int] = (0, 1)

    def __call__(self, y):
        x = y * (self.max_value - self.min_value) + self.min_value
        return x

@dataclass
class StandardScaler:
    """
    Class which can be used to normalize PyTorch tensors to zero mean and unit
    variance.
    Similar as done in: https://gist.github.com/farahmand-m/8a416f33a27d73a149f92ce4708beb40
    """
    # TODO: implement this with running mean and std
    mean = 0
    std = 1
    epsilon = 0.00000000000000001

    values_range: Tuple[int, int] = (0, 1)

    def fit(self, values):
        dims = []
        for d in range(len(values.shape) - 1):  # -1 since we do not take into account time dimension yet
            dims.append(d + 1)
        self.mean = torch.mean(values, dim=dims) # for now taking over all channels is ok since all channels are same. but this may change later on.
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values-self.mean)/(self.std+self.epsilon)

    def __call__(self, y):
        self.fit(y)
        x = self.transform(y)
        return x


@dataclass
class ERA5T2MData(Dataset):
    """
    ERA5 Temperature Dataset containing 15341 samples of shape 1x128x128.
    Loads as X array file format (https://docs.xarray.dev/en/stable/user-guide/data-structures.html)
    and converts to PyTorch Tensors. Temperature is returned in Kelvin and
    can be converted to Celsius by subtracting 273.15.
    Description of data: https://apps.ecmwf.int/codes/grib/param-db/?id=167
    """

    data_path: str
    transform: Callable = None
    window_size: int = 2

    def __post_init__(self):
        self.data = xr.open_zarr(self.data_path)['t2m']

        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])  #XRToTensor(),
                                                 # MinMaxScaler(values_range=(0, 1))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        x = self.data[idx:idx+self.window_size+1]

        # x = self.data.isel(time=idx)
        # print(x.values.min()-273.15, x.values.max()-273.15), converting min and max temp to Celsius
        time = np.array(x.coords['time'])
        latitude = np.array(x.coords['latitude'])
        longitude = np.array(x.coords['longitude'])

        # resize frames
        x_resh = np.zeros((x.shape[0], x.shape[1]//2, x.shape[2]//2))

        for i in range(self.window_size+1):
            x_resh[i,...] = resize(x[i,...], (x.shape[1]//2, x.shape[2]//2),
                              anti_aliasing=True)

        x=x_resh
        # self.transform(x).permute(1,0,2).unsqueeze(1)
        return self.transform(x).permute(1,0,2).unsqueeze(1), str(time), latitude, longitude

# datashape = ERA5T2MData('/home/christina/Documents/research/auto-encoding-normalizing-flows/code/data/ftp.bgc-jena.mpg.de/pub/outgoing/aschall/data.zarr')[0][0].shape
# temperatures, time = ERA5T2MData('/home/christina/Documents/research/auto-encoding-normalizing-flows/code/data/ftp.bgc-jena.mpg.de/pub/outgoing/aschall/data.zarr')[0]

### visualize
# fig, ax = plt.subplots()
# ax.set_xlabel('longitude [degree]')
# ax.set_ylabel('latitude [degree]')
# # ax.set_xticklabels(['-10','-5','0','5','10','15','20'])
# # ax.set_yticklabels(['30','35','40','45','50','55','60'])
# temp = ax.imshow(temperatures[0,:,:], cmap='inferno')
# fig.colorbar(temp, ax=ax, location='right')
# ax.set_title('2m temperature - {}'.format(time))
# plt.show()

# TODO: fix x set_xticks and set_xticklabels
# TODO: fix y set_yticks and set_yticklabels
# TODO: fix temperature scale
