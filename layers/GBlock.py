"""
Skilful precipitation nowcasting using deep generative models of radar, from DeepMind
https://arxiv.org/abs/2104.00954
"""

import torch
import torch.nn as nn

from utils.utils import Identity, depth2space

class GBlockCell(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True):
        super(GBlockCell, self).__init__()
        Scaling = (nn.Upsample if upsample else Identity)
        
        self.conv3x3 = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                Scaling(scale_factor=2),
                nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
                )
        self.conv1x1 = nn.Sequential(
            Scaling(scale_factor=2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0))
            )


    def forward(self, x):
        conv3x3 = self.conv3x3(x)
        conv1x1 = self.conv1x1(x)
        return conv3x3 + conv1x1

class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GBlock, self).__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 1, 1, 0))
        self.g_block = GBlockCell(in_channels, in_channels, upsample=False)
        self.g_block_up = GBlockCell(in_channels, out_channels, upsample=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.g_block(x)
        x = self.g_block_up(x)
        return x

class LastGBlock(nn.Module):
    def __init__(self, in_channels):
        super(LastGBlock, self).__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 1, 1, 0))
        self.g_block = GBlockCell(in_channels, in_channels, upsample=False)
        self.g_block_up = GBlockCell(in_channels, in_channels, upsample=True)
        self.conv_out = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(in_channels, 4, 1, 1, 0))
                )

    def forward(self, x):
        x = self.conv(x)
        x = self.g_block(x)
        x = self.g_block_up(x)
        x = self.conv_out(x)
        x = depth2space(x)
        return x