import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import random

import pdb

# random.seed(0)
# torch.manual_seed(0)
# np.random.seed(0)
#
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


class ActNorm(nn.Module):
    """
    Activation Normalization layer which normalizes the activation
    values of a batch by their mean and variance. The activations of
    each channel then should have zero mean and unit variance. This
    layer ensure more stable parameter updates during training as it reduces
    the variance over the samples in a mini batch.
    Adapted from: https://github.com/pclucas14/pytorch-glow/blob/master/invertible_layers.py
    Note: the initialization here is data dependent. Might need to know during
    inference phase. Therefore, at test time we need to set initialize to true.
    """

    def __init__(self, num_features, logscale_factor=1.0, scale=1.0, testmode=False):
        super(ActNorm, self).__init__()

        self.initialized = True if testmode else False
        # self.testmode = testmode

        self.logscale_factor = logscale_factor
        self.scale = scale

        self.register_parameter("b", nn.Parameter(torch.zeros(1, num_features, 1)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(1, num_features, 1)))

    def forward(self, input, logdet, reverse=False):

        if not reverse:

            input_shape = input.size()
            input = input.view(input_shape[0], input_shape[1], -1)

            if not self.initialized:
                self.initialized = True
                unsqueeze = lambda x: x.unsqueeze(0).unsqueeze(-1).detach()

                # compute the mean and variance
                sum_size = input.size(0) * input.size(-1)
                b = -torch.sum(input, dim=(0, -1)) / sum_size
                vars = unsqueeze(
                    torch.sum((input + unsqueeze(b)) ** 2, dim=(0, -1)) / sum_size
                )

                logs = (torch.log(self.scale / (torch.sqrt(vars) + 1e-6))
                    / self.logscale_factor)

                # print("Input", input.shape, "Bias", self.b.data.shape)
                self.b.data.copy_(unsqueeze(b).data)
                self.logs.data.copy_(logs.data)

            logs = self.logs * self.logscale_factor
            b = self.b

            output = (input + b) * torch.exp(logs)
            dlogdet = torch.sum(logs) * input.size(-1)  # c x h

            return output.view(input_shape), logdet.cuda() + dlogdet.cuda()

        elif reverse == True:
            assert self.initialized
            input_shape = input.size()
            input = input.view(input_shape[0], input_shape[1], -1)
            logs = self.logs * self.logscale_factor
            b = self.b
            output = input * torch.exp(-logs) - b
            dlogdet = torch.sum(logs) * input.size(-1)  # c x h

            return output.view(input_shape), logdet - dlogdet


class Invert1x1Conv(nn.Conv2d):
    """

    """
    # from https://github.com/pclucas14/pytorch-glow/blob/master/invertible_layers.py
    # TODO: could add decomposition of matrix.
    def __init__(self, num_channels):
        self.num_channels = num_channels
        nn.Conv2d.__init__(self, num_channels, num_channels, 1, bias=False)
        self.init = False

    def reset_parameters(self):
        # initialization done with rotation matrix
        w_init = np.linalg.qr(np.random.randn(self.num_channels, self.num_channels))[0]
        w_init = torch.from_numpy(w_init.astype("float32"))
        w_init = w_init.unsqueeze(-1).unsqueeze(-1)
        self.weight.data.copy_(w_init)

    def forward_(self, x, logdet, reverse=False):

        _, _, height, width = x.size()

        if not reverse:
            dlogdet = torch.slogdet(self.weight.squeeze())[1] * height * width
            logdet += dlogdet
            output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                              self.dilation, self.groups)
        else:
            dlogdet = torch.slogdet(self.weight.squeeze())[1] * height * width
            logdet = logdet - dlogdet
            weight_inv = (torch.inverse(self.weight.squeeze()).unsqueeze(-1).unsqueeze(-1))
            output = F.conv2d(x, weight_inv, self.bias, self.stride, self.padding,
                              self.dilation, self.groups)

        self.init = True
        return output, logdet


class Shuffle(nn.Module):
    # Shuffling on the channel axis
    def __init__(self, num_channels):
        super(Shuffle, self).__init__()
        indices = np.arange(num_channels)
        np.random.shuffle(indices)
        rev_indices = np.zeros_like(indices)
        for i in range(num_channels):
            rev_indices[indices[i]] = i

        indices = torch.from_numpy(indices).long()
        rev_indices = torch.from_numpy(rev_indices).long()
        self.register_buffer("indices", indices)
        self.register_buffer("rev_indices", rev_indices)
        # self.indices, self.rev_indices = indices.cuda(), rev_indices.cuda()

    def forward(self, x, logdet, reverse=False):
        if not reverse:
            return x[:, self.indices], logdet
        else:
            return x[:, self.rev_indices], logdet


class Squeeze(nn.Module):
    def __init__(self, factor=2):
        super(Squeeze, self).__init__()
        assert factor > 1 and isinstance(
            factor, int
        ), "no point of using this if factor <= 1"
        self.factor = factor

    def squeeze_bchw(self, x):
        bs, c, h, w = x.size()
        assert h % self.factor == 0 and w % self.factor == 0

        # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
        x = x.view(bs, c, h // self.factor, self.factor, w // self.factor, self.factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(bs, c * self.factor * self.factor,
                   h // self.factor, w // self.factor)
        return x

    def unsqueeze_bchw(self, x):
        bs, c, h, w = x.size()
        assert c >= 4 and c % 4 == 0

        # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
        x = x.view(bs, c // self.factor ** 2, self.factor, self.factor, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(bs, c // self.factor ** 2, h * self.factor, w * self.factor)
        return x

    def forward(self, x, reverse=False):
        if len(x.size()) != 4:
            raise NotImplementedError
        if not reverse:
            return self.squeeze_bchw(x)
        else:
            return self.unsqueeze_bchw(x)


class SqueezeTS(nn.Module):

    def __init__(self, factor=2):

        super(SqueezeTS, self).__init__()
        assert factor > 1 and isinstance(
            factor, int
        ), "no point of using this if factor <= 1"
        self.factor = factor

    def squeeze_bchw(self, x):
        bs, c, h, w = x.size()
        assert h % self.factor == 0 and w % self.factor == 0

        # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
        x = x.view(bs, c, h // self.factor, self.factor, w // self.factor, self.factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(bs, c * self.factor * self.factor,
                   h // self.factor, w // self.factor)
        return x

    def unsqueeze_bchw(self, x):
        bs, c, h, w = x.size()
        assert c >= 4 and c % 4 == 0

        # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
        x = x.view(bs, c // self.factor ** 2, self.factor, self.factor, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(bs, c // self.factor ** 2, h * self.factor, w * self.factor)
        return x

    def forward(self, x, reverse=False):
        if len(x.size()) != 4:
            raise NotImplementedError
        if not reverse:
            return self.squeeze_bchw(x)
        else:
            return self.unsqueeze_bchw(x)


class conv2d_actnorm(nn.Conv2d):
    def __init__(self, channels_in, channels_out, filter_size, stride=1, padding=None):
        super().__init__(
            channels_in, channels_out, filter_size, stride=stride, padding=padding
        )
        padding = (filter_size - 1) // 2 or padding
        self.conv = nn.Conv2d(channels_in, channels_out, filter_size,
                              stride=stride, padding=padding, bias=False)
        self.actnorm = ActNorm(channels_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.actnorm.forward(x, -1)[0]
        return x


class conv2d_zeros(nn.Conv2d):
    def __init__(self, channels_in, channels_out, filter_size=3, stride=1,
                 padding=0, logscale=3.0):

        super().__init__(channels_in, channels_out, filter_size, stride=stride,
                         padding=padding)

        self.register_parameter("logs", nn.Parameter(torch.zeros(channels_out, 1, 1)))
        self.logscale_factor = logscale

    def reset_parameters(self):
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        out = super().forward(input)
        return out * torch.exp(self.logs * self.logscale_factor)


class conv3d_zeros(nn.Conv3d):
    def __init__(self, channels_in, channels_out, filter_size=3, stride=1,
                 padding=0, logscale=3.0):
        super().__init__(channels_in, channels_out, filter_size, stride=stride,
                         padding=padding)

        self.register_parameter("logs", nn.Parameter(torch.zeros(channels_out, 1, 1, 1)))
        self.logscale_factor = logscale

    def reset_parameters(self):
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        out = super().forward(input)
        return out * torch.exp(self.logs * self.logscale_factor)


class Net(nn.Module):
    def __init__(self, level, s, in_channels, input_shape, cond_channels, lag_len,
                 noscale, noscaletest, intermediate_size=512):
        super().__init__()

        self.squeezer = Squeeze()
        self.s = s
        self.level = level
        self.lag_len = lag_len
        c, w, h = input_shape
        self.cond_channels = cond_channels
        d = 2 if noscale else 1

        self.upsample_conv = nn.Conv3d(in_channels//2, 2**(2*(level+1))*c, kernel_size=1)
        self.change_time_channel = nn.Conv3d(lag_len + 1, 1,
                                             (1,1,1), stride=(1,1,1),padding=(0,0,0))  # nn.Conv3d(3, 1, kernel_size=1) # conv = nn.Conv3d(4, 1, (3,3,3), stride=(1,1,1),padding=(0,0,0))

        # pdb.set_trace()
        self.Net = nn.Sequential(
                   nn.Conv3d(2**(2*(level+1))*c, intermediate_size,
                             kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.Conv3d(intermediate_size, intermediate_size,
                             kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.Conv3d(intermediate_size, in_channels, #  2*(2**(2*(level+1))//2 *c)
                             kernel_size=1),
                   nn.ReLU())

        # pdb.set_trace()
        self.Net[0].bias.data.zero_()
        self.Net[2].weight.data.normal_(0, 0.05)
        self.Net[2].bias.data.zero_()

    def forward(self, input, h):

        input = self.upsample_conv(input)
        h = torch.cat((input, h), 2)

        h = h.permute(0,2,1,3,4)
        h = self.change_time_channel(h)
        h = h.permute(0,2,1,3,4)

        out = self.Net(h)
        return out


class ConditionalCoupling(nn.Module):

    def __init__(self, level, s, in_channels, input_shape, cond_channels, lag_len,
                 filter_size, noscale, noscaletest):
        super().__init__()
        self.Net = Net(level, s, in_channels, input_shape, cond_channels, lag_len,
                       noscale, filter_size)
        self.noscale = noscale
        self.noscaletest = noscaletest
        self.scaling_factor = nn.Parameter(torch.zeros(in_channels//2))

    def forward(self, z, h=None, logdet=0, logpz=0, reverse=False):

        z1, z2 = torch.chunk(z, 2, dim=1)
        out = self.Net(z1, h)

        if self.noscale: # print("Scale disabled")
            t, scale = h, torch.ones_like(out)

        else:
            # print("Scale enabled")

            t, logs = cross(out)
            s_fac = self.scaling_factor.exp().view(1, -1, 1, 1, 1)
            logscale = torch.tanh(logs / s_fac) * s_fac

            if self.noscaletest:
                # print("Scale disabled for sampling")
                logscale = torch.ones_like(scale)

        if not reverse:
            y2 = (z2 + t) * logscale.exp()
            y1 = z1
            logdet if self.noscale else flatten_sum(logscale)

        else:
            y2 = (z2 * torch.exp(-logscale) - t)
            y1 = z1
            logdet if self.noscale else flatten_sum(logscale)

        y = torch.cat((y1, y2), dim=1)
        return y, logdet


class GaussianPrior(nn.Module):
    def __init__(self, C, s, level, cond_channels, lag_len, flow_var_shape, final=False):
        super(GaussianPrior, self).__init__()

        self.scaling_factor = nn.Parameter(torch.zeros(4))
        self.scaling_factor_final = nn.Parameter(torch.zeros(4))

        _, c, t, h, w = flow_var_shape
        self.lag_len = lag_len
        self.cond_channels = cond_channels
        self.final = final
        self.squeezer = Squeeze()
        self.s = s

        if final:

            if level == 1:
                self.conv1 = nn.Conv3d(2**(2*(level+1)), 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv3d(32, 2*2**(2*(level+1))//2, kernel_size=1)

            elif level==0:
                self.conv1 = nn.Conv3d(2**(2*(level+1)), 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv3d(32, 2*c, kernel_size=1)

            elif level==2:
                self.conv1 = nn.Conv3d(2**(2*(level+1)), 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv3d(32, 2*2**(2*(level+1))//4, kernel_size=1)

            # self.Net = nn.Sequential(
            #            nn.Conv3d(4, 64,
            #                      kernel_size=3, padding=1),
            #            nn.ReLU(),
            #            nn.Conv3d(64, 128,
            #                      kernel_size=3),
            #            nn.ReLU(),
            #            nn.Conv3d(128, 64,
            #                      kernel_size=3),
            #            nn.ReLU(),
            #            nn.Conv3d(64, 2,
            #                      kernel_size=1),
            #            nn.ReLU())

            self.change_time_channel = nn.Conv3d(lag_len, 1, (1,1,1), stride=(1,1,1),
                                                 padding=(0,0,0))

            # nn.Conv3d(2, 1, kernel_size=1)
            # self.conv1 = conv3d_zeros(9, 2, padding=1)
            # self.conv2 = conv3d_zeros(2, 4, filter_size=1)

        else:
            self.change_time_channel = nn.Conv3d((level+1) * C, (level+1) * C,
                                                 (1,1,1), stride=(2,1,1),
                                                 padding=(0,0,0))
            self.conv = conv3d_zeros((level+1) * C + C // 2, C, padding=1)

            # TODO: improve architecture here
            # self.Net = nn.Sequential(
            #            nn.Conv3d(4, 64,
            #                      kernel_size=3, padding=1),
            #            nn.ReLU(),
            #            nn.Conv3d(64, 128,
            #                      kernel_size=3),
            #            nn.ReLU(),
            #            nn.Conv3d(128, 64,
            #                      kernel_size=3),
            #            nn.ReLU(),
            #            nn.Conv3d(64, 2,
            #                      kernel_size=1),
            #            nn.ReLU())

    def split2d_prior(self, z, h):

        # decrease time-channel-axis to 1
        h = self.change_time_channel(h)

        x = torch.cat((z, h), 1)
        h = self.conv(x)
        mean, log_sigma = h[:, 0::2], nn.functional.softplus(h[:, 1::2])

        # s_fac = self.scaling_factor.exp().view(1, -1, 1, 1, 1)
        # log_sigma = torch.tanh(log_sigma / s_fac) * s_fac

        return mean, log_sigma

    def final_prior(self, h):

        # pdb.set_trace()

        h = self.conv1(h)
        h = self.conv2(h)

        # decrease time-channel-axis to 1 - depending on lead length but leaving 1 for now
        h = h.permute(0,2,1,3,4)
        h = self.change_time_channel(h)
        h = h.permute(0,2,1,3,4)

        # split along channel size
        mean, log_sigma = h[:, 0::2,:,:,:], nn.functional.softplus(h[:, 1::2,:,:,:])
        return mean, log_sigma

    def forward(self, x, h, reverse, eps=1.0, logpz=0,
                logdet=0, use_stored=False):

        if not reverse: # forward mode
            if not self.final:
                # print("Computing log probs intermediate")
                z, y = torch.chunk(x, 2, 1)
                self.y = y.detach()
                mean, log_sigma = self.split2d_prior(z, h)
                prior = torch.distributions.normal.Normal(loc=mean, scale=log_sigma)
                logpz += prior.log_prob(y).sum(dim=[1,2,3,4])

            else:
                # print("Computing log probs final")
                # final prior computation
                mean, log_sigma = self.final_prior(h)
                prior = torch.distributions.normal.Normal(loc=mean, scale=log_sigma)
                logpz += prior.log_prob(x).sum(dim=[1,2,3,4])
                self.y = x
                z = x

        else: # reverse mode
            if not self.final: # split prior
                mean, log_sigma = self.split2d_prior(x, h)
                prior = torch.distributions.normal.Normal(loc=mean, scale=log_sigma)
                z2 = self.y if use_stored else prior.sample()
                z = torch.cat((x, z2), 1)

            else:
                # sample from final prior
                mean, log_sigma = self.final_prior(h)
                prior = torch.distributions.normal.Normal(loc=mean, scale=log_sigma)
                # bsz, t, c, h, w = mean.shape
                # print(bsz, t*2, c, h, w)
                z = self.y if use_stored else prior.sample() # prior.sample(sample_shape=(bsz, 2*t, c, h, w))

        return z, logdet, logpz


class CouplingLayer(nn.Module):
# taken from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html
    def __init__(self, network, mask, c_in):
        """
        Coupling layer inside a normalizing flow.
        Inputs:
            network - A PyTorch nn.Module constituting the deep neural network for mu and sigma.
                      Output shape should be twice the channel size as the input.
            mask - Binary mask (0 or 1) where 0 denotes that the element should be transformed,
                   while 1 means the latent will be used as input to the NN.
            c_in - Number of input channels
        """
        super().__init__()
        self.network = network
        self.scaling_factor = nn.Parameter(torch.zeros(c_in))
        # Register mask as buffer as it is a tensor which is not a parameter,
        # but should be part of the modules state.
        self.register_buffer('mask', mask)

    def forward(self, z, ldj, reverse=False, orig_img=None):
        """
        Inputs:
            z - Latent input to the flow
            ldj - The current ldj of the previous flows.
                  The ldj of this layer will be added to this tensor.
            reverse - If True, we apply the inverse of the layer.
            orig_img (optional) - Only needed in VarDeq. Allows external
                                  input to condition the flow on (e.g. original image)
        """
        # Apply network to masked input
        z_in = z * self.mask
        if orig_img is None:
            nn_out = self.network(z_in)
        else:
            nn_out = self.network(torch.cat([z_in, orig_img], dim=1))

        s, t = nn_out.chunk(2, dim=1)

        # Stabilize scaling output
        s_fac = self.scaling_factor.exp().view(1, -1, 1, 1)
        s = torch.tanh(s / s_fac) * s_fac

        # Mask outputs (only transform the second part)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        # Affine transformation
        if not reverse:
            # whether we first shift and then scale, or the other way round,
            # is a design choice, and usually does not have a big impact
            z = (z + t) * torch.exp(s)
            ldj += s.sum(dim=[1,2,3])
        else:
            z = (z * torch.exp(-s)) - t
            ldj -= s.sum(dim=[1,2,3])

        return z, ldj


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
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1),
            ConcatRELU(),
            nn.Conv2d(2*c_hidden, 2*c_in, kernel_size=1)
        )

    def forward(self, x):
        out = self.net(x)
        val, gate = out.chunk(2, dim=1)
        return x + val * torch.sigmoid(gate)


class GatedConvNet(nn.Module):
# taken from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html
    def __init__(self, c_in, c_hidden=32, c_out=-1, num_layers=3):
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
        layers += [nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1)]
        for layer_index in range(num_layers):
            layers += [GatedConv(c_hidden, c_hidden),
                       LayerNormChannels(c_hidden)]
        layers += [ConcatRELU(),
                   nn.Conv2d(2*c_hidden, c_out, kernel_size=3, padding=1)]
        self.nn = nn.Sequential(*layers)

        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()

    def forward(self, x):
        return self.nn(x)


class conv2d_actnorm(nn.Conv2d):
    def __init__(self, channels_in, channels_out, filter_size, args=None, padding=None, stride=1):
        super().__init__(channels_in, channels_out, filter_size, args, padding,stride)
        padding = (filter_size - 1) // 2 or padding
        self.conv = nn.Conv2d(channels_in, channels_out, filter_size, stride=stride, padding=padding, bias=False)
        self.actnorm = ActNorm(channels_out, args)

    def forward(self, x):
        x = self.conv(x)
        x = self.actnorm.forward(x, -1)[0]
        return x


class conv2d_zeros(nn.Conv2d):
    def __init__(self, channels_in, channels_out, filter_size=3, stride=1, padding=0, logscale=3.):
        super().__init__(channels_in, channels_out, filter_size, stride=stride, padding=padding)
        self.register_parameter("logs", nn.Parameter(torch.zeros(channels_out, 1, 1)))
        self.logscale_factor = logscale

    def reset_parameters(self):
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        out = super().forward(input)
        return out * torch.exp(self.logs * self.logscale_factor)


def create_checkerboard_mask(h, w, invert=False):
    # taken from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html
    x, y = torch.arange(h, dtype=torch.int32), torch.arange(w, dtype=torch.int32)
    xx, yy = torch.meshgrid(x, y)
    mask = torch.fmod(xx + yy, 2)
    mask = mask.to(torch.float32).view(1, 1, h, w)
    if invert:
        mask = 1 - mask
    return mask


class ConcatRELU(nn.Module):
    """
    Activation function that applies ELU in both direction (inverted and plain).
    Allows non-linearity while providing strong gradients for any input (important for final convolution)
    """
# taken from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html
    def forward(self, x):
        return torch.cat([F.relu(x), F.relu(-x)], dim=1)


class LayerNormChannels(nn.Module):
# taken from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html
    def __init__(self, c_in):
        """
        This module applies layer norm across channels in an image. Has been shown to work well with ResNet connections.
        Inputs:
            c_in - Number of channels of the input
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(c_in)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
########################### UTILS ##############################################

def split(feature):
    """
    Splits the input feature tensor into two halves along the channel dimension.
    Channel-wise masking.
    Args:
        feature: Input tensor to be split.
    Returns:
        Two output tensors resulting from splitting the input tensor into half
        along the channel dimension.
    """
    C = feature.size(1)
    return feature[:, : C // 2, ...], feature[:, C // 2:, ...]

def cross(feature):
    """
    Performs two different slicing operations along the channel dimension.
    Args:
        feature: PyTorch Tensor.
    Returns:
        feature[:, 0::2, ...]: Selects every feature with even channel dimensions index.
        feature[:, 1::2, ...]: Selects every feature with uneven channel dimension index.
    """
    return feature[:, 0::2, ...], feature[:, 1::2, ...]

def concat_feature(tensor_a, tensor_b):
    """
    Concatenates features along the first dimension.
    """
    return torch.cat((tensor_a, tensor_b), dim=1)

def flatten_sum(logps):
    while len(logps.size()) > 1:
        logps = logps.sum(dim=-1)
    return logps
