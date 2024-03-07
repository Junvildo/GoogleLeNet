from torch import nn as nn
import torch
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

seq = nn.Sequential

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass

def conv2d(ch_in, ch_out, kz, s=1, p=0):
    return spectral_norm(nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=kz, stride=s, padding=p))

class Inception(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.reduce1 = seq(
            conv2d(ch_in=192, ch_out=64, kz=1, s=1), nn.ReLU()
        )
        self.reduce3 = seq(
            conv2d(ch_in=192, ch_out=96, kz=1, s=1), nn.ReLU(),
            conv2d(ch_in=96, ch_out=192, kz=3, s=1), nn.ReLU()
        )
        self.reduce5 = seq(
            conv2d(ch_in=192, ch_out=16, kz=1, s=1), nn.ReLU(),
            conv2d(ch_in=16, ch_out=32, kz=5, s=1), nn.ReLU()
        )
        self.pool = seq(
            nn.MaxPool2d(kernel_size=3, stride=1),
            conv2d(ch_in=192, ch_out=32, kz=1, s=1), nn.ReLU()
        )