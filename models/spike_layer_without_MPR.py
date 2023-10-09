import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from IPython import embed

class SpikeModule(nn.Module):

    def __init__(self):
        super().__init__()
        self._spiking = False

    def set_spike_state(self, use_spike=True):
        self._spiking = use_spike

    def forward(self, x):
        if self._spiking is not True and len(x.shape) == 4:
            x = x.mean([0])
        return x


def spike_activation(x, ste=False, temp=5.0):
    out_s = torch.gt(x, 0.5)
    if ste:
        out_bp = torch.clamp(x, 0, 1)
    else:
        out_bp = torch.clamp(x, 0, 1)
        out_bp = (torch.tanh(temp * (out_bp-0.5)) + np.tanh(temp * 0.5)) / (2 * (np.tanh(temp * 0.5)))
    return (out_s.float() - out_bp).detach() + out_bp


def gradient_scale(x, scale):
    yout = x
    ygrad = x * scale
    y = (yout - ygrad).detach() + ygrad
    return y


def mem_update(x_in, mem, V_th, decay, grad_scale=1., temp=5.0):
    mem = mem * decay + x_in
    spike = spike_activation(mem / V_th, temp=temp)
    mem = mem * (1 - spike)
    return mem, spike


class LIFAct(SpikeModule):
    """ Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """

    def __init__(self, step, temp):
        super(LIFAct, self).__init__()
        self.step = step
        self.V_th = 1.0
        self.temp = 5.0
        self.grad_scale = 0.1

    def forward(self, x):
        if self._spiking is not True:
            return F.relu(x)
        if self.grad_scale is None:
            self.grad_scale = 1 / math.sqrt(x[0].numel()*self.step)
        u = torch.rand_like(x[0]) * 0.5
        out = []
        for i in range(self.step):
            u, out_i = mem_update(x_in=x[i], mem=u, V_th=self.V_th,
                                  grad_scale=self.grad_scale, decay=0.25, temp=self.temp)
            out += [out_i]
        out = torch.stack(out)
        return out


class SpikeConv(SpikeModule):


    def __init__(self, conv, step=2):
        super(SpikeConv, self).__init__()
        self.conv = conv
        self.step = step

    def forward(self, x):
        if self._spiking is not True:
            return self.conv(x)
        out = []
        for i in range(self.step):
            out += [self.conv(x[i])]
        out = torch.stack(out)
        return out


class SpikeLinear(SpikeModule):

    def __init__(self, linear, step=2):
        super().__init__()
        self.linear = linear
        self.step = step

    def forward(self, x):
        if self._spiking is not True:
            return self.linear(x)
        T, B, C= x.shape
        out = x.reshape(-1, C)
        out = self.linear(out)
        B_o, C_o = out.shape
        out = out.view(T, B, C_o).contiguous()
        return out
    
class SpikeBatchNorm(SpikeModule):

    def __init__(self, BN: nn.BatchNorm1d, step=2):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(BN.num_features)
        self.bn2 = nn.BatchNorm2d(BN.num_features)
        self.step = step
        
    def forward(self, x):
        if self._spiking is not True:
            return self.BN(x)
        if len(x.shape) == 3:
            out = x.permute(1, 2, 0)
            out = self.bn1(out)
            out = out.permute(2, 0, 1).contiguous() 
        else:
            out = x.permute(1, 2, 0, 3)
            out = self.bn2(out)
            out = out.permute(2, 0, 1, 3).contiguous()
        return out
