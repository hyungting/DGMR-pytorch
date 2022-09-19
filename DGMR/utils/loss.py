import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class PixelWiseRegularizer(nn.Module):
    """
    Weighted loss function.
    Args:
        cfg: config format tranformed by config.py
        magnitude: int, weight of the summed loss value
        min_value: int, lower boundary of input tensor
        max_value: int, highr boundary of output tensor
    Return:
        torch.tensor
    """
    def __init__(
        self,
        cfg=None, 
        magnitude: int=20,
        min_value: int=-9,
        max_value: int=60
        ):
        super().__init__()
        if cfg is not None:
            self.magnitude = cfg.LAMBDA
            self.min_value = cfg.MIN_VALUE
            self.max_value = cfg.MAX_VALUE
        else:
            self.magnitude = magnitude
            self.min_value = min_value
            self.max_value = max_value
    
    def forward(self, pred, target):
        weight = torch.clamp(target, min=self.min_value, max=self.max_value)
        difference = torch.abs(pred - target)
        loss = difference * weight
        return self.magnitude * loss.mean()

class HingeLoss(nn.Module):
    """
    Args:
        cfg: config format tranformed by config.py
        margin: int
        label_smoothing: bool
    """
    def __init__(
        self,
        cfg=None,
        margin: int=1,
        label_smoothing: bool=False
        ):
        super().__init__()
        if cfg is not None:
            self.margin = cfg.MARGIN
            self.label_smoothing = cfg.LABEL_SMOOTHING
        else:
            self.margin = margin
            self.label_smoothing = label_smoothing

    def forward(self, pred, validity):
        if self.label_smoothing:
            prob = 0.1 * torch.randint(low=8, high=12, size=pred.shape, device=pred.device)
            pred = prob * pred
        if validity:
            loss = F.relu(self.margin - pred)
        else:
            loss = F.relu(self.margin + pred)
        return Variable(loss.mean(), requires_grad=True)

if __name__ == "__main__":
    pass