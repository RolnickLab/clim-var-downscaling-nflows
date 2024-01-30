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

def minmax_scaler(x):
    values_range = (0,1)
    for i in range(x.shape[0]):
        x[i,...] = (x[i,...] - x[i,...].max()) / (x[i,...].max() - x[i,...].min())
    return x * (values_range[1] - values_range[0]) + values_range[0]

@dataclass # TODO: needs to be adapted!
class InverseMinMaxScaler:
    max_value: float = 315.91873
    min_value: float = 241.22385
    values_range: Tuple[int, int] = (0, 1)

    def __call__(self, y):
        x = y * (self.max_value - self.min_value) + self.min_value
        return x

@dataclass
class WeatherBenchData(Dataset):
    """
    WeatherBench: A benchmark dataset for data-driven weather forecasting.
    Description of data: https://arxiv.org/pdf/2002.00469.pdf
    """

    data_path: str
    args: None
    transform: Callable = None
    window_size: int = 2
    s: int = 2

    def __post_init__(self):

        self.data = xr.open_mfdataset(self.data_path + '*.nc', combine='by_coords')['z'] #.sel(time=slice('2016', '2016')).mean('time').load()

        if self.transform is None:
            self.transform = transforms.Compose([XRToTensor()])
            # self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        x = self.data[idx:idx+self.window_size+1]
        x_unorm = x.values

        time = np.array(x.coords['time'])
        latitude = np.array(x.coords['lat'])
        longitude = np.array(x.coords['lon'])

        # normalize over each time-step, perhaps change this later
        x_ = minmax_scaler(x) 
        
        return self.transform(x_), torch.FloatTensor(x_unorm), str(time), latitude, longitude
