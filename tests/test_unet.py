import torch

from src.model import UNet


def test_forward_unet():
    n_channels = 2
    n_classes = 3
    x = torch.rand((1, n_channels, 224, 224))
    model = UNet(n_channels, n_classes)
    model.forward(x)