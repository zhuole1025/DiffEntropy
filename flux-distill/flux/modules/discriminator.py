from typing import Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
        

class ResidualBlock(nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.fn(x) + x) / np.sqrt(2)


    
def make_block(channels: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(
            channels,
            channels,
            kernel_size = kernel_size,
            padding = kernel_size//2,
            padding_mode = 'circular',
        ),
        BatchNormLocal(channels),
        nn.LeakyReLU(0.2, True),
    )


class BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 8, eps: float = 1e-5):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
    
        # Reshape batch into groups.
        G = np.ceil(x.size(0)/self.virtual_bs).astype(int)
        x = x.view(G, -1, x.size(-2), x.size(-1))
    
        # Calculate stats.
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))
    
        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]
    
        return x.view(shape)


    
class DiscriminatorHead(nn.Module):
    def __init__(self, channels: int, c_dim: int = 0, cmap_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.c_dim = c_dim
        self.cmap_dim = cmap_dim

        self.conv_1 = nn.Conv1d(channels, channels, kernel_size=1, padding=0)
        self.norm_1 = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.act_1 = nn.SiLU()
        
        self.conv_2 = nn.Conv1d(channels, channels, kernel_size=9, padding=4, padding_mode='circular')
        self.norm_2 = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.act_2 = nn.SiLU()
        
        self.conv_3 = nn.Conv1d(channels, 1, kernel_size=1, padding=0)
        nn.init.kaiming_normal_(self.conv_3.weight)
        nn.init.zeros_(self.conv_3.bias)
        
        # self.main = nn.Sequential(
        #     make_block(channels, kernel_size=1),
        #     ResidualBlock(make_block(channels, kernel_size=9))
        # )
    
        # if self.c_dim > 0:
        #     self.cmapper = FullyConnectedLayer(self.c_dim, cmap_dim)
        #     self.cls = spectral_norm(
        #         nn.Conv1d(channels, cmap_dim, kernel_size=1, padding=0)
        #     )
        # else:
        #     self.cls = spectral_norm(
        #         nn.Conv1d(channels, 1, kernel_size=1, padding=0)
        #     )
    
    def forward(self, x: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        # Apply first conv block
        x = x.transpose(1, 2)  # [B, L, C] -> [B, C, L] for conv1d
        x = self.conv_1(x)
        x = x.transpose(1, 2)  # Back to [B, L, C]
        x = self.norm_1(x)
        x = self.act_1(x)
        
        # Apply second conv block with residual connection
        x_skip = x
        x = x.transpose(1, 2)
        x = self.conv_2(x) 
        x = x.transpose(1, 2)
        x = self.norm_2(x)
        x = self.act_2(x)
        x = x + x_skip
        
        # Final conv to get logits
        x = x.transpose(1, 2)
        x = self.conv_3(x)
        out = x.transpose(1, 2)
        
        # x = self.norm(x).transpose(1, 2)
        # h = self.main(x)
        # out = self.cls(h)
    
        # if self.c_dim > 0:
        #     cmap = self.cmapper(c).unsqueeze(-1)
        #     out = (out * cmap).sum(1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
    
        return out