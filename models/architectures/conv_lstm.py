"""
Author: Christina Winkler, July 2022
This module implements a Convolutional LSTM based on a Gated Conv Net in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

import sys
sys.path.append("../../")

class StackedConvLSTM(nn.Module):
    def __init__(self, num_channels, num_kernels=64, kernel_size=(3,3), num_layers=3):
        """"""
        super(StackedConvLSTM, self).__init__()

        self.sequential = nn.Sequential()

        # first layer
        self.sequential.add_module("convlstm1", ConvLSTM(
        in_channels=num_channels, out_channels=num_kernels,
        hidden_channels=num_kernels))

        self.sequential.add_module("batchnorm1", nn.BatchNorm3d(num_features=num_kernels))

        # add remaining layers
        for l in range(2, num_layers+1):

            self.sequential.add_module(f'convlstm{l}', ConvLSTM(
                    in_channels=num_kernels, out_channels=num_kernels,
                    hidden_channels=kernel_size
            ))
            self.sequential.add_module(f"batchnorm{l}", nn.BatchNorm3d(num_features=num_kernels))

        # add last layer to predict output frame
        self.conv = nn.Conv2d(in_channels=num_kernels, out_channels=num_channels,
                              kernel_size=kernel_size, padding=(1,1))
    def forward(self, x):
        out = self.sequential(x)
        out = nn.Sigmoid()(self.conv(output[:,:,-1]))
        return out

class ConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        """

        """
        super(ConvLSTM, self).__init__()
        self.out_channels = out_channels
        self.convLSTMcell = ConvLSTMCell(in_channels, hidden_channels, out_channels)

    def forward(self, x):
        # x of shape [bsz, num_channels, seq_len, height, width]
        bsz, _, seq_len, height, width = x.size()

        output = torch.zeros(bsz, self.out_channels, seq_len, height, width).cuda()
        h = torch.zeros(bsz,self.out_channels,height,width).cuda()
        c = torch.zeros(bsz,self.out_channels,height,width).cuda()

        for time_step in range(seq_len):
            h,c = self.convLSTMcell(x[:,:,time_step],(h,c))
            output[:,:,time_step] = h
        return output

class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        """
        Implements convolutional LSTM cell.
        Args:

        Returns:
        """
        super(ConvLSTMCell, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.gated_conv_net = GatedConvNet(in_channels, hidden_channels,
                                           out_channels, num_layers)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_channels)
        for weight in self.parameters():
            if len(weight.data.shape) >= 2:
                torch.nn.init.kaiming_uniform_(weight.data)
            else:
                weight.data.uniform_(-stdv, stdv)

    def forward(self, x_t, state_t=None):

        h_t, c_t = state_t # h: [] c: []

        # concat
        X = torch.cat([x_t.cuda(), h_t.cuda()], dim=1)
        out = self.gated_conv_net(X).cuda()

        i_, f_, o_, g_ = torch.chunk(out, 4, dim=1)
        i = torch.sigmoid(i_)
        f = torch.sigmoid(f_)
        o = torch.sigmoid(o_)
        g = torch.sigmoid(g_) # TODO: could be tanh !!!

        c_next = f * c_t.cuda() + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class GatedConv(nn.Module):
# taken from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html
    def __init__(self, c_in, c_hidden):
        """
        This module applies a two-layer convolutional ResNet block with input gate
        Inputs:
            c_in - Number of channels of the input
            c_hidden - Number of hidden dimensions we want to model (usually similar to c_in)
        """
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(c_in, c_hidden, kernel_size=3, padding=1), ConcatRELU(),
                                 nn.Conv3d(2*c_hidden, 2*c_in, kernel_size=1))

    def forward(self, x):
        out = self.net(x)
        val, gate = torch.chunk(out, 2, dim=1)
        return x + val * torch.sigmoid(gate)

class GatedConvNet(nn.Module):
    # taken from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html
    # paper: https://arxiv.org/pdf/1612.08083v3.pdf
    def __init__(self, c_in, c_hidden=128, c_out=-1, num_layers=6):
        """
        Module that summarizes the previous blocks to a full convolutional neural network.
        Inputs:
            c_in - Number of input channels
            c_hidden - Number of hidden dimensions to use within the network
            c_out - Number of output channels. If -1, 2 times the input channels are used (affine coupling)
            num_layers - Number of gated ResNet blocks to apply
        """
        super().__init__()
        c_out = c_out if c_out > 0 else 2 * c_in
        layers = []
        layers += [nn.Conv3d(c_in, c_hidden, kernel_size=3, padding=1)]
        for layer_index in range(num_layers):
            layers += [GatedConv(c_hidden, c_hidden),
                       InstanceNorm3dChannel(c_hidden)]
        layers += [ConcatRELU(),
                   nn.Conv3d(2*c_hidden, c_out, kernel_size=3, padding=1)]
        self.nn = nn.Sequential(*layers)

        # change initalization
        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()

    def forward(self, x):
        out = self.nn(x.cuda()).cuda()
        return out

class ConcatRELU(nn.Module):
    """
    Activation function that applies ELU in both direction (inverted and plain).
    Allows non-linearity while providing strong gradients for any input (important for final convolution)
    """
# taken from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html
    def forward(self, x):
        return torch.cat([F.relu(x), F.relu(-x)], dim=1)

class InstanceNorm3dChannel(nn.Module):
    def __init__(self, c_in):
        """
        This module applies instance norm across channels in an image.
        Inputs:
            c_in - Number of channels of the input
        """
        super().__init__()
        self.instance_norm = nn.InstanceNorm3d(c_in)

    def forward(self, x):
        x = self.instance_norm(x)
        return x
