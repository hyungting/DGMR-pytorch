import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
def Regularizer(pred, target, alpha=20):
    B, N, H, W = pred.shape
    difference = torch.abs(pred - target)
    # 24mm/h -> 45dbz
    weight = torch.clip(target, min=0, max=45)
    loss = alpha * difference * weight
    return loss.mean()

def DiscriminatorLoss(pred, operator):
    # if y = true, y = 1
    # if y = false, y = -1
    loss = F.relu(1. - operator * pred)
    return Variable(loss.mean(), requires_grad=True)

def Normalizer(x):
    return 1 - torch.exp(-x/1.0)

def space2depth(x, factor=2):
    B, C, H, W = x.shape
    x = nn.Unfold(factor, stride=factor)(x)
    return x.view(B, C * factor ** 2, H // factor, W // factor)

def depth2space(x, factor=2):
    B, C, H, W = x.shape
    x = x.view(B, C, H*W)
    x = nn.Fold((H*factor, W*factor), kernel_size=(factor, factor), stride=factor)(x)
    return x

