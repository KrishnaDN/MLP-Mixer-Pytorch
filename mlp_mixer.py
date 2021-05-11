#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 09:38:47 2021

@author: krishna
"""

from torch import nn
from einops.layers.torch import Rearrange
import torch


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches):
        super().__init__()
        self.pre_layer_norm = nn.LayerNorm(dim)
        self.post_layer_norm = nn.LayerNorm(dim)
        
        self.token_mixer = nn.Sequential(nn.Linear(num_patches, num_patches),
                            nn.GELU(),
                            nn.Dropout(0.1))
        self.channel_mixer = nn.Sequential(nn.Linear(dim, dim),
                            nn.GELU(),
                            nn.Dropout(0.1))
    def forward(self, x):
        pre_ln =self.pre_layer_norm(x)
        tm_out = self.token_mixer(pre_ln.transpose(1,2)).transpose(1,2)
        tm_out = tm_out + x
        post_ln = self.post_layer_norm(tm_out)
        cm_out = self.channel_mixer(post_ln)
        return cm_out
    
    
class MLPMixer(nn.Module):
    def __init__(self,input_size, patch_size, dim = 512, img_channel=3, layers = 12, num_classes=12):
        super().__init__()
        assert (input_size[0] % patch_size[0]) == 0, 'H must be divisible by patch size'
        assert (input_size[1] % patch_size[1]) == 0, 'W must be divisible by patch size'
        num_patches = int(input_size[0]/patch_size[0] * input_size[1]/patch_size[1])
        patch_dim = img_channel * patch_size[0] * patch_size[1]
        self.to_patch_embedding = nn.Sequential(
                    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size[0], p2 = patch_size[1]),
                    nn.Linear(patch_dim, dim))
         
        self.network = nn.Sequential(*[nn.Sequential(MixerBlock(dim,num_patches)) for _ in range(layers)])
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(dim,num_classes)
    
    def forward(self,x):
        x = self.to_patch_embedding(x)
        x = self.network(x)
        return self.classifier(self.pool(x.transpose(1,2)).squeeze(2))
    
    
    
if __name__=='__main__':  
    model = MLPMixer(
        input_size = (256,256),
        patch_size = (16,16),
        dim = 512,
        layers = 12,
        num_classes = 12,   
        )
    
    img = torch.randn(10, 3, 256, 256)
    pred = model(img)
