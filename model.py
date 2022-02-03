import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast
from torchvision import transforms

from module import *
prints = lambda tensor_list: print([x.shape for x in tensor_list])

class Generator(nn.Module):
    def __init__(self, in_step, out_step, debug=False):
        super(Generator, self).__init__()
        self.in_step = in_step
        self.out_step = out_step
        self.debug = debug
        
        self.d_block0 = nn.ModuleList([DBlock(self.in_step*4, self.in_step*2, relu=False, downsample=True) for _ in range(3)])
        self.d_block1 = nn.ModuleList([DBlock(self.in_step*8, self.in_step*4, relu=False, downsample=True) for _ in range(3)])
        self.d_block2 = nn.ModuleList([DBlock(self.in_step*16, self.in_step*8, relu=False, downsample=True) for _ in range(3)])
        self.d_block3 = nn.ModuleList([DBlock(self.in_step*32, self.in_step*16, relu=False, downsample=True) for _ in range(3)])

        self.conv0 = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(96, 48, 3, 1, 1)),
                nn.ReLU(inplace=True)
                )
        self.conv1 = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(192, 96, 3, 1, 1)),
                nn.ReLU(inplace=True)
                )
        self.conv2 = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(384, 192, 3, 1, 1)),
                nn.ReLU(inplace=True)
                )
        self.conv3 = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(768, 384, 3, 1, 1)),
                nn.ReLU(inplace=True)
                ) 
        self.latent_conv = nn.ModuleList(
                nn.Sequential(
                    nn.utils.spectral_norm(nn.Conv2d(8, 8, 3, 1, 1)),
                    LBlock(8, 24),
                    LBlock(24, 48),
                    LBlock(48, 192),
                    SpatialAttention(),
                    LBlock(192, 768)
                    ) for _ in range(self.out_step)
                )
        self.g_block0 = nn.ModuleList([GBlock(384, 384) for _ in range(self.out_step)])
        self.g_block1 = nn.ModuleList([GBlock(192, 192) for _ in range(self.out_step)])
        self.g_block2 = nn.ModuleList([GBlock(96, 96) for _ in range(self.out_step)])
        self.g_block3 = nn.ModuleList([LastGBlock(48) for _ in range(self.out_step)])
        self.gru_layer0 = nn.ModuleList([ConvGRUCell(768, 384) if _ > 0 else ConvGRUCell(768, 384) for _ in range(self.out_step)])
        self.gru_layer1 = nn.ModuleList([ConvGRUCell(384, 192) if _ > 0 else ConvGRUCell(384, 192) for _ in range(self.out_step)])
        self.gru_layer2 = nn.ModuleList([ConvGRUCell(192, 96) if _ > 0 else ConvGRUCell(192, 96) for _ in range(self.out_step)])
        self.gru_layer3 = nn.ModuleList([ConvGRUCell(96, 48) if _ > 0 else ConvGRUCell(96, 48) for _ in range(self.out_step)])
    
    def forward(self, x0, return_noise=False):
        B, C, H, W = x0.shape

        ##### conditioning stack #####
        x0 = space2depth(x0) # 256 -> 128
        if self.debug: print(f"s2d    : {x0.shape}")
        
        ### downsample with so many Ds ###
        ### We want x0, x1, x2, x3 for next step
        temp_x0, temp_x1, temp_x2, temp_x3 = [], [], [], []
        for block0, block1, block2, block3 in zip(self.d_block0, self.d_block1, self.d_block2, self.d_block3):
            d0 = block0(x0)
            d1 = block1(d0)
            d2 = block2(d1)
            d3 = block3(d2)
            temp_x0.append(d0)
            temp_x1.append(d1)
            temp_x2.append(d2)
            temp_x3.append(d3)

        x0 = torch.cat(temp_x0, dim=1)
        if self.debug: print(f"new x0 : {x0.shape}")
        x1 = torch.cat(temp_x1, dim=1)
        if self.debug: print(f"new x1 : {x1.shape}")
        x2 = torch.cat(temp_x2, dim=1)
        if self.debug: print(f"new x2 : {x2.shape}")
        x3 = torch.cat(temp_x3, dim=1)
        if self.debug: print(f"new x3 : {x3.shape}")

        del temp_x0, temp_x1, temp_x2, temp_x3

        x0 = self.conv0(x0)
        if self.debug: print(f"conv 0 : {x0.shape}")
        
        x1 = self.conv1(x1)
        if self.debug: print(f"conv 1 : {x1.shape}")
        
        x2 = self.conv2(x2)
        if self.debug: print(f"conv 2 : {x2.shape}")

        x3 = self.conv3(x3)
        if self.debug: print(f"conv 3 : {x3.shape}")
        
        torch.cuda.empty_cache()
        ##### sampler #####
        outputs = []
        noises = []
        for t in range(self.out_step):
            noise = self.latent_conv[t](torch.randn((B, 8, H//32, W//32)).to(x0.device))
            
            if self.debug: print(f"init x3: {x3.shape}")
            x3 = self.gru_layer0[t](noise, x3)
            if self.debug: print(f"1st GRU: {x3.shape}")
            g = self.g_block0[t](x3)
            x2 = self.gru_layer1[t](g, x2)
            if self.debug: print(f"2nd GRU: {x2.shape}")
            g = self.g_block1[t](x2)
            x1 = self.gru_layer2[t](g, x1)
            if self.debug: print(f"3rd GRU: {x1.shape}")
            g = self.g_block2[t](x1)
            x0 = self.gru_layer3[t](g, x0)
            if self.debug: print(f"4th GRU: {x0.shape}")
            g = self.g_block3[t](x0) 
            outputs.append(g)
            noises.append(noise.detach().cpu().numpy()[0])
        
        outputs = torch.cat(outputs, dim=1)
        if self.debug: print(f"outputs: {outputs.shape}")
        if return_noise:
            return outputs, noises
        else:
            return outputs

class SpatialDiscriminator(nn.Module):
    def __init__(self, n_frame=8, factor=2, debug=False):
        super(SpatialDiscriminator, self).__init__()
        self.debug = debug
        self.n_frame = n_frame
        self.factor = factor
        self.in_channels = n_frame * (factor**2) # 8 * 4

        self.avgpooling = nn.AvgPool2d(2)
        self.d_blocks = nn.ModuleList([
                DBlock(4, 3 * 4, relu=False, downsample=True), # 4 -> (3 * 4) * 4 = 48
                DBlock(12 * 4, 6 * 4, downsample=True), # 48 -> (6 * 4) * 4 = 96
                DBlock(24 * 4, 12 * 4, downsample=True), # 96 -> (12 * 4) * 4 = 192
                DBlock(48 * 4, 24 * 4, downsample=True), # 192 -> (24 * 4) * 4 = 384
                DBlock(96 * 4, 48 * 4, downsample=True), # 384 -> (48 * 4) * 4 = 768
                DBlock(192 * 4, 192 * 4, downsample=False) # 768 -> 768, no downsample no * 4
                ])
        self.linear = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(768, 1)),
                nn.BatchNorm1d(1)
                )
        self.sigmoid = nn.Sequential(
                #nn.ReLU(),
                nn.Sigmoid()
                )

    def forward(self, x):
        B, N, C, H, W = x.shape # batch_size, total_frames, channel=1, height, width
        indices = random.sample(range(N), self.n_frame)
        x = x[:, indices, :, :, :]
        if self.debug: print(f"Picked x: {x.shape}")
        x = x.view(B*self.n_frame, C, H, W)
        if self.debug: print(f"Reshaped: {x.shape}")
        x = self.avgpooling(x)
        if self.debug: print(f"Avg pool: {x.shape}")
        x = space2depth(x, self.factor)
        if self.debug: print(f"S2Dshape: {x.shape}")

        for i, block in enumerate(self.d_blocks):
            x = block(x)
            if self.debug: print(f"D block{i}: {x.shape}")

        # sum pooling
        x = torch.sum(x, dim=(-1, -2))
        if self.debug: print(f"Sum pool: {x.shape}")
        
        x = self.linear(x)
        if self.debug: print(f"Linear : {x.shape}")

        x = x.view(B, self.n_frame, -1)
        if self.debug: print(f"Reshaped: {x.shape}")

        x = torch.sum(x, dim=1)
        if self.debug: print(f"Sum up : {x.shape}")

        x = self.sigmoid(x)
        return x

class TemporalDiscriminator(nn.Module):
    def __init__(self, factor=2, size=128, debug=False):
        super(TemporalDiscriminator, self).__init__()
        self.debug = debug
        self.factor = factor
        self.size = size

        self.random_crop = nn.Sequential(
                transforms.RandomCrop(size)
                )
        self.d3_blocks = nn.ModuleList([
                D3Block(4, 3 * 4, relu=False, downsample=True), # C: 4 -> 48, T -> T/2
                D3Block(12 * 4, 6 * 4, downsample=True) # C: 48 -> 96, T/2 -> T/4 (not exactly the same as DGMR)
                ])
        self.d_blocks = nn.ModuleList([
                DBlock(24 * 4, 12 * 4, downsample=True), # 96 -> (12 * 4) * 4 = 192
                DBlock(48 * 4, 24 * 4, downsample=True), # 192 -> (24 * 4) * 4 = 384
                DBlock(96 * 4, 48 * 4, downsample=True), # 384 -> (48 * 4) * 4 = 768
                DBlock(192 * 4, 192 * 4, downsample=False) # 768 -> 768, no downsample no * 4
                ])
        self.linear = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(768, 1)),
                nn.BatchNorm1d(1)
                )
        self.sigmoid = nn.Sequential(
                #nn.ReLU(),
                nn.Sigmoid()
                )

    def forward(self, x):
        B, C, T, H, W = x.shape
        
        x = x.permute(0, 2, 1, 3, 4).view(B*T, C, H, W) # -> B, T, C, H, W
        
        x = self.random_crop(x)
        if self.debug: print(f"Cropped : {x.shape}")
        
        x = space2depth(x) # B*T, C, H, W
        x = x.view(B, T, -1, x.shape[-2], x.shape[-1]).permute(0, 2, 1, 3, 4) # -> B, C, T, H, W
        if self.debug: print(f"S2Dshape: {x.shape}")

        for i, block3d in enumerate(self.d3_blocks):
            x = block3d(x)
            if self.debug: print(f"3D block: {x.shape}")
        
        B, C, T, H, W  = x.shape
        x = x.permute(0, 2, 1, 3, 4).view(B*T, C, H, W)
        if self.debug: print(f"Reshaped: {x.shape}")
        
        for i, block in enumerate(self.d_blocks):
            x = block(x)
            if self.debug: print(f"D block{i}: {x.shape}")
        
        # sum pooling
        x = torch.sum(x, dim=(-1, -2))
        if self.debug: print(f"Sum pool: {x.shape}")

        x = self.linear(x)
        if self.debug: print(f"Linear : {x.shape}")

        x = x.view(B, T, -1)
        if self.debug: print(f"Reshaped: {x.shape}")

        x = torch.sum(x, dim=1)
        if self.debug: print(f"Sum up : {x.shape}")

        x = self.sigmoid(x)

        return x


if __name__ == "__main__":
    device = torch.device("cpu")
    model = TemporalDiscriminator()
    model.to(device)
    test = torch.randn(16, 1, 22, 256, 256).to(device)
    out = model(test)
    print(out.shape)

