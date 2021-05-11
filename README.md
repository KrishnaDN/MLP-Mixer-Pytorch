# MLP-Mixer-Pytorch
Implementation of the the paper "MLP-Mixer: An all-MLP Architecture for Vision"
```
from mlp_mixer import MLPMixer
model = MLPMixer(
        input_size = (256,256),
        patch_size = (16,16),
        dim = 512,
        layers = 12,
        num_classes = 12)
    
img = torch.randn(10, 3, 256, 256)
pred = model(img) # (1, 1000)
```
