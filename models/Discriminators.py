import torch
import random
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from layers.DBlock import DBlock, D3Block
from utils.utils import random_crop, space2depth

class SpatialDiscriminator(nn.Module):
    def __init__(self, n_frame=8, debug=False):
        super(SpatialDiscriminator, self).__init__()
        self.debug = debug
        self.n_frame = n_frame

        self.avgpooling = nn.AvgPool2d(2)
        self.d_blocks = nn.ModuleList([
                DBlock(4, 48, relu=False, downsample=True), # 4 -> (3 * 4) * 4 = 48
                DBlock(48, 96, downsample=True), # 48 -> (6 * 4) * 4 = 96
                DBlock(96, 192, downsample=True), # 96 -> (12 * 4) * 4 = 192
                DBlock(192, 384, downsample=True), # 192 -> (24 * 4) * 4 = 384
                DBlock(384, 768, downsample=True), # 384 -> (48 * 4) * 4 = 768
                DBlock(768, 768, downsample=False) # 768 -> 768, no downsample no * 4
                ])
        self.linear = nn.Sequential(
                nn.BatchNorm1d(768),
                spectral_norm(nn.Linear(768, 1))
                )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(2)
        B, N, C, H, W = x.shape # batch_size, total_frames, channel=1, height, width
        indices = random.sample(range(N), self.n_frame)
        x = x[:, indices, :, :, :]
        if self.debug: print(f"Picked x: {x.shape}")
        x = x.view(B*self.n_frame, C, H, W)
        if self.debug: print(f"Reshaped: {x.shape}")
        x = self.avgpooling(x)
        if self.debug: print(f"Avg pool: {x.shape}")
        x = space2depth(x)
        if self.debug: print(f"S2Dshape: {x.shape}")

        for i, block in enumerate(self.d_blocks):
            x = block(x)
            if self.debug: print(f"D block{i}: {x.shape}")

        # sum pooling
        x = self.relu(x)
        x = torch.sum(x, dim=(-1, -2))
        if self.debug: print(f"Sum pool: {x.shape}")

        x = self.linear(x)
        if self.debug: print(f"Linear : {x.shape}")

        x = x.view(B, self.n_frame, -1)
        if self.debug: print(f"Reshaped: {x.shape}")

        x = torch.sum(x, dim=1)
        if self.debug: print(f"Sum up : {x.shape}")

        #x = self.activation(x)
        return x

class TemporalDiscriminator(nn.Module):
    def __init__(self, crop_size=128, debug=False):
        super(TemporalDiscriminator, self).__init__()
        self.debug = debug
        self.crop_size = crop_size

        self.avgpooling = nn.AvgPool3d(2)
        self.d3_blocks = nn.ModuleList([
                D3Block(4, 48, relu=False, downsample=True), # C: 4 -> 48, T -> T/2
                D3Block(48, 96, downsample=True) # C: 48 -> 96, T/2 -> T/4 (not exactly the same as DGMR)
                ])
        self.d_blocks = nn.ModuleList([
                DBlock(96, 192, downsample=True), # 96 -> (12 * 4) * 4 = 192
                DBlock(192, 384, downsample=True), # 192 -> (24 * 4) * 4 = 384
                DBlock(384, 768, downsample=True), # 384 -> (48 * 4) * 4 = 768
                DBlock(768, 768, downsample=False) # 768 -> 768, no downsample no * 4
                ])
        self.linear = nn.Sequential(
                nn.BatchNorm1d(768),
                spectral_norm(nn.Linear(768, 1))
                )
        self.relu = nn.ReLU()

    def forward(self, x):
        # n_frame = in_step + out_step
        x = random_crop(x, size=self.crop_size).to(x.device)
        x = x.unsqueeze(1)
        B, C, T, H, W = x.shape

        x = x.permute(0, 2, 1, 3, 4).view(B*T, C, H, W) # -> B, T, C, H, W
        if self.debug: print(f"Cropped : {x.shape}")

        x = space2depth(x) # B*T, C, H, W
        x = x.view(B, T, -1, x.shape[-2], x.shape[-1]).permute(0, 2, 1, 3, 4) # -> B, C, T, H, W
        if self.debug: print(f"S2Dshape: {x.shape}")

        for i, block3d in enumerate(self.d3_blocks):
            x = block3d(x)
            if self.debug: print(f"3D block: {x.shape}")

        B, C, T, H, W  = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        if self.debug: print(f"Reshaped: {x.shape}")

        for i, block in enumerate(self.d_blocks):
            x = block(x)
            if self.debug: print(f"D block{i}: {x.shape}")

        # sum pooling
        x = self.relu(x)
        x = torch.sum(x, dim=(-1, -2))
        if self.debug: print(f"Sum pool: {x.shape}")

        x = self.linear(x)
        if self.debug: print(f"Linear : {x.shape}")

        x = x.view(B, T, -1)
        if self.debug: print(f"Reshaped: {x.shape}")

        x = torch.sum(x, dim=1)
        if self.debug: print(f"Sum up : {x.shape}")

        return x

class Discriminators(nn.Module):
    def __init__(self, n_frame=6, crop_size=128, debug=False):
        super(Discriminators, self).__init__()
        self.spatial_discriminator = SpatialDiscriminator(n_frame=n_frame, debug=debug)
        self.temporal_discriminator = TemporalDiscriminator(crop_size=crop_size, debug=debug)

    def forward(self, x, y):
        s_score = self.spatial_discriminator(y)
        t_score = self.temporal_discriminator(torch.cat((x, y), dim=1))
        return torch.cat((s_score, t_score), dim=0)

if __name__ == "__main__":
    device = torch.device("cuda")
    model = TemporalDiscriminator()
    model.to(device)
    test = torch.randn(800, 1, 22, 256, 256).to(device)
    out = model(test)
    print(out.shape)