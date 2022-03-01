"""
Skilful precipitation nowcasting using deep generative models of radar, from DeepMind
https://arxiv.org/abs/2104.00954
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def Regularizer(pred, target, alpha=20):
    B, N, H, W = pred.shape
    difference = torch.abs(pred - target)
    # 24mm/h -> 45dbz
    weight = torch.clip(target, min=0.5, max=45)
    loss = difference * weight
    return alpha * loss.mean()

def DiscriminatorLoss(pred, operator):
    # if y = true, y = 1
    # if y = false, y = -1
    loss = F.relu(1. - operator * pred)
    return Variable(loss.mean(), requires_grad=True)