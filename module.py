import torch
import torch.nn as nn

from utils import space2depth, depth2space

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class Space2Depth(nn.Module):
    def __init__(self):
        super(Space2Depth, self).__init__()
    def forward(self, x):
        if len(x.shape) == 4:
            return space2depth(x)
        if len(x.shape) == 5:
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4) # B, T, C, H, W
            x = x.reshape(B*T, C, H, W)
            x = space2depth(x)
            x = x.view(B,  T, -1, x.shape[-2], x.shape[-1])
            x = x.permute(0, 2, 1, 3, 4) # B, C, T, H, W
            return x

class ConvGRUCell(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=50):
        super(ConvGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.reset_gate = nn.utils.spectral_norm(nn.Conv2d(
                in_channels=(input_dim+hidden_dim),
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1
                ))
        self.update_gate = nn.utils.spectral_norm(nn.Conv2d(
                in_channels=(input_dim+hidden_dim),
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1
                ))
        self.out_gate = nn.utils.spectral_norm(nn.Conv2d(
                in_channels=input_dim+hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1
                ))
    
    def forward(self, x, h):
        stacked = torch.cat([x, h], dim=1)

        update = torch.sigmoid(self.update_gate(stacked)) #hidden=384
        reset = torch.sigmoid(self.reset_gate(stacked)) #hidden=384

        out = torch.tanh(self.out_gate(torch.cat([x, h*reset], dim=1))) #x=768, h=384, out=768+384
        h_next = h * (1-update) + out * update #h=384, update=384, out=384, update=384

        return h_next

class SpatialAttention(nn.Module):
    """
    DGMR from DeepMind
    Spatial attention module: for latent conditioning stack, the structure is questionable
    """
    def __init__(self, in_channels=192, out_channels=192, ratio_kq=8, ratio_v=8, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.ratio_kq = ratio_kq
        self.ratio_v = ratio_v
        self.conv_q = nn.Conv2d(in_channels, out_channels//ratio_kq, 1, 1, 0)
        self.conv_k = nn.Conv2d(in_channels, out_channels//ratio_kq, 1, 1, 0)
        self.conv_v = nn.Conv2d(in_channels, out_channels//ratio_v, 1, 1, 0)
        self.conv_out = nn.Conv2d(out_channels//ratio_v, out_channels, 1, 1, 0)
    
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
    DGMR from DeepMind
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

class GBlockCell(nn.Module):
    def __init__(self, in_channels):
        super(GBlockCell, self).__init__()
        self.conv3x3 = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1))
                )
        self.conv1x1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 1, 1, 0))


    def forward(self, x):
        conv3x3 = self.conv3x3(x)
        conv1x1 = self.conv1x1(x)
        return conv3x3 + conv1x1

class GBlockUpCell(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GBlockUpCell, self).__init__()
        self.conv3x3 = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2),
                nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
                )
        self.conv1x1 = nn.Sequential(
                nn.Upsample(scale_factor=2),
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
        self.g_block = GBlockCell(in_channels)
        self.g_block_up = GBlockUpCell(in_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.g_block(x)
        x = self.g_block_up(x)
        return x

class LastGBlock(nn.Module):
    def __init__(self, in_channels):
        super(LastGBlock, self).__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 1, 1, 0))
        self.g_block = GBlockCell(in_channels)
        self.g_block_up = GBlockUpCell(in_channels, in_channels)
        self.conv_out = nn.Sequential(
                #nn.Upsample(scale_factor=2),
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

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, relu=True, downsample=True):
        super(DBlock, self).__init__()
        Scaling = (Space2Depth if downsample else Identity)
        ReLU = (nn.ReLU() if relu else Identity())

        self.conv1x1 = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0)),
                Scaling()
                )
        self.conv3x3 = nn.Sequential(
                ReLU,
                nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
                nn.ReLU(inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1)),
                Scaling()
                )

    def forward(self, x):
        conv1x1 = self.conv1x1(x)
        conv3x3 = self.conv3x3(x)
        return conv1x1 + conv3x3

class D3Block(nn.Module):
    def __init__(self, in_channels, out_channels, relu=True, downsample=True):
        super(D3Block, self).__init__()
        Scaling = (Space2Depth if downsample else Identity)
        ReLU = (nn.ReLU() if relu else Identity())

        self.conv1x1 = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv3d(in_channels, out_channels, 1, (2, 1, 1), (0, 0, 0))),
                Scaling()
                )
        self.conv3x3 = nn.Sequential(
                ReLU,
                nn.utils.spectral_norm(nn.Conv3d(in_channels, in_channels, 3, (1, 1, 1), (1, 1, 1))),
                nn.ReLU(inplace=True),
                nn.utils.spectral_norm(nn.Conv3d(in_channels, out_channels, 3, (2, 1, 1), (1, 1, 1))),
                Scaling()
                )

    def forward(self, x):
        conv1x1 = self.conv1x1(x)
        conv3x3 = self.conv3x3(x)
        return conv1x1 + conv3x3
