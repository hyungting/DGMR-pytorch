"""
Skilful precipitation nowcasting using deep generative models of radar, from DeepMind
https://arxiv.org/abs/2104.00954
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def Regularizer(pred, target, alpha=20):
    B, N, H, W = pred.shape
    difference = torch.abs(pred - target)
    # 24mm/h -> 45dbz
    #map = 45 * torch.ones(target.shape, device=pred.device)
    #map = torch.cat((target, map), dim=1)
    #weight, _ = torch.max(map, keepdim=True, dim=1)
    weight = torch.clamp(target, min=0, max=45)
    loss = difference * weight
    return alpha * loss.mean()

def DiscriminatorLoss(pred, operator):
    # if y = true, y = 1
    # if y = false, y = -1
    loss = F.relu(1. - operator * pred)
    return Variable(loss.mean(), requires_grad=True)

class LaplacianPyramidLoss(nn.Module):
    """
    orginal: https://github.com/mtyka/laploss/blob/master/laploss.py
    """
    def __init__(self, max_level=4, kernel_size=5, sigma=1, stride=1, repeat=1):
        super(LaplacianPyramidLoss, self).__init__()
        self.max_level = max_level
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.stride = stride
        self.repeat = repeat

        self.conv = lambda input, weight: F.conv2d(input, weight, stride=self.stride, padding="same", groups=1)
        self.pool = nn.AvgPool2d(2)

    def make_gauss_kernel(self):
        grid = np.float32(np.mgrid[0:self.kernel_size, 0:self.kernel_size].T)
        gaussian = lambda x: np.exp((x - self.kernel_size//2) ** 2 / (-2 * self.sigma ** 2)) ** 2
        kernel = np.sum(gaussian(grid), axis=2)
        kernel /= np.sum(kernel)
        return torch.tensor(kernel)

    def conv_gauss(self, x):
        weights = self.make_gauss_kernel()
        weights = weights.view(1, 1, self.kernel_size, self.kernel_size)
        weights = weights.to(x.device)
        result = x
        for r in range(self.repeat):
            result = self.conv(x, weights)
        return result

    def make_laplacian_pyramid(self, x):
        current = x
        pyramid = []
        for _ in range(self.max_level):
            gauss = self.conv_gauss(current)
            diff = current - gauss
            pyramid.append(diff)
            current = self.pool(gauss)
        pyramid.append(current)
        return pyramid

    def forward(self, x, target):
        B, N, H, W = x.shape
        P = [_ for _ in range(self.max_level)]
        loss = torch.zeros(N)
        for n in range(N):
            x_pyramid = self.make_laplacian_pyramid(x[:, n, ...].unsqueeze(1))
            t_pyramid = self.make_laplacian_pyramid(target[:, n, ...].unsqueeze(1))
            n_loss = [torch.abs(a-b).sum() / H / W * (2 ** (2 * p)) for p, a, b in zip(P, x_pyramid, t_pyramid)]
            loss[n] = sum(n_loss)
        loss = torch.mean(Variable(loss, requires_grad=True))
        return loss

if __name__ == "__main__":
    loss = Regularizer
    x = torch.randn(16, 4, 256, 256)
    y = torch.randn(16, 4, 256, 256)
    out = loss(x, y)
    print(out)