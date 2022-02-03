import torch
import torch.nn as nn

def Regularizer(pred, target, weight=None, alpha=20):
    B, N, H, W = pred.shape
    difference = pred - target
    if weight is not None:
        difference = difference * weight
    return alpha * torch.sum(torch.abs(difference * target)) / (H * W * N)

def space2depth(x, factor=2):
    B, C, H, W = x.shape
    x = nn.Unfold(factor, stride=factor)(x)
    return x.view(B, C * factor ** 2, H // factor, W // factor)

def depth2space(x, factor=2):
    B, C, H, W = x.shape
    x = x.view(B, C, H*W)
    x = nn.Fold((H*factor, W*factor), kernel_size=(factor, factor), stride=factor)(x)
    return x

