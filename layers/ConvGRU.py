"""
Skilful precipitation nowcasting using deep generative models of radar, from DeepMind
https://arxiv.org/abs/2104.00954
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

class ConvGRUCell(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=50):
        super(ConvGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.reset_gate = spectral_norm(nn.Conv2d(
                in_channels=(input_dim+hidden_dim),
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1
                ),
                eps=1e-4)
        self.update_gate = spectral_norm(nn.Conv2d(
                in_channels=(input_dim+hidden_dim),
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1
                ),
                eps=1e-4)
        self.out_gate = spectral_norm(nn.Conv2d(
                in_channels=input_dim+hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1
                ),
                eps=1e-4)
    
    def forward(self, x, h):
        stacked = torch.cat([x, h], dim=1)

        update = torch.sigmoid(self.update_gate(stacked)) #hidden=384
        reset = torch.sigmoid(self.reset_gate(stacked)) #hidden=384

        out = torch.tanh(self.out_gate(torch.cat([x, h*reset], dim=1))) #x=768, h=384, out=768+384
        h_next = h * (1-update) + out * update #h=384, update=384, out=384, update=384

        return h_next
