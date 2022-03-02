"""
Skilful precipitation nowcasting using deep generative models of radar, from DeepMind
https://arxiv.org/abs/2104.00954
"""

import torch
import torch.nn as nn

from utils.utils import Identity, Space2Depth

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, relu=True, downsample=True):
        super(DBlock, self).__init__()
        Scaling = (Space2Depth if downsample else Identity)
        ReLU = (nn.ReLU() if relu else Identity())

        self.conv1x1 = nn.Sequential(
                nn.utils.parametrizations.spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0)),
                Scaling()
                )
        self.conv3x3 = nn.Sequential(
                ReLU,
                nn.utils.parametrizations.spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
                nn.ReLU(inplace=True),
                nn.utils.parametrizations.spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1)),
                Scaling()
                )

    def forward(self, x):
        conv1x1 = self.conv1x1(x)
        conv3x3 = self.conv3x3(x)
        return conv1x1 + conv3x3

class D3Block(nn.Module):
    def __init__(self, in_channels, out_channels, relu=True, downsample=True):
        super(D3Block, self).__init__()
        Scaling = (Space2Depth if downsample else Identity)
        ReLU = (nn.ReLU() if relu else Identity())

        self.conv1x1 = nn.Sequential(
                nn.utils.parametrizations.spectral_norm(nn.Conv3d(in_channels, out_channels, 1, (2, 1, 1), (0, 0, 0))),
                Scaling()
                )
        self.conv3x3 = nn.Sequential(
                ReLU,
                nn.utils.parametrizations.spectral_norm(nn.Conv3d(in_channels, in_channels, 3, (1, 1, 1), (1, 1, 1))),
                nn.ReLU(inplace=True),
                nn.utils.parametrizations.spectral_norm(nn.Conv3d(in_channels, out_channels, 3, (2, 1, 1), (1, 1, 1))),
                Scaling()
                )

    def forward(self, x):
        conv1x1 = self.conv1x1(x)
        conv3x3 = self.conv3x3(x)
        return conv1x1 + conv3x3