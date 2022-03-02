"""
Skilful precipitation nowcasting using deep generative models of radar, from DeepMind
https://arxiv.org/abs/2104.00954
"""

import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    """
    Spatial attention module: for latent conditioning stack
    """
    def __init__(self, in_channels=192, out_channels=192, ratio_kq=8, ratio_v=8, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.ratio_kq = ratio_kq
        self.ratio_v = ratio_v
        self.conv_q = nn.Conv2d(in_channels, out_channels//ratio_kq, 1, 1, 0, bias=False)
        self.conv_k = nn.Conv2d(in_channels, out_channels//ratio_kq, 1, 1, 0, bias=False)
        self.conv_v = nn.Conv2d(in_channels, out_channels//ratio_v, 1, 1, 0, bias=False)
        self.conv_out = nn.Conv2d(out_channels//ratio_v, out_channels, 1, 1, 0, bias=False)
    
    def einsum(self, q, k, v):
        # org shape = B, C, H, W
        k = k.view(k.shape[0], k.shape[1], -1) # B, C, H*W
        v = v.view(v.shape[0], v.shape[1], -1) # B, C, H*W
        beta = torch.einsum("bchw, bcL->bLhw", q, k)
        beta = torch.softmax(beta, dim=1)
        out = torch.einsum("bLhw, bcL->bchw", beta, v)
        return out

    def forward(self, x):
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)
        
        # the question is whether x should be preserved or just attn
        out = self.einsum(q, k, v)
        out = self.conv_out(out)
        return x + out

class LBlock(nn.Module):
    """
    L block: for latent conditioning stack
    """
    def __init__(self, in_channels, out_channels):
        super(LBlock, self).__init__()
        self.conv3x3 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                )
        self.conv1x1 = nn.Conv2d(in_channels, out_channels-in_channels, 1, 1, 0)

    def forward(self, x):
        conv3x3 = self.conv3x3(x)
        conv1x1 = self.conv1x1(x)
        x = torch.cat((x, conv1x1), dim=1)
        x += conv3x3
        return x