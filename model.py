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
            conv2d(ch_in=96, ch_out=128, kz=3, s=1, p=1), nn.ReLU()
        )
        self.reduce5 = seq(
            conv2d(ch_in=192, ch_out=16, kz=1, s=1), nn.ReLU(),
            conv2d(ch_in=16, ch_out=32, kz=5, s=1, p=2), nn.ReLU()
        )
        self.pool = seq(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv2d(ch_in=192, ch_out=32, kz=1, s=1), nn.ReLU()
        )

    def forward(self, x):
        y1 = self.reduce1(x)
        y2 = self.reduce3(x)
        y3 = self.reduce5(x)
        y4 = self.pool(x)
        return torch.cat((y1, y2, y3, y4), dim=1)
    

class GoogLeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_bl_1 = Inception()
        self.block = seq(
            conv2d(ch_in=3, ch_out=64, kz=7, s=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), #nn.LocalResponseNorm(),
            conv2d(ch_in=64, ch_out=192, kz=1, s=1), nn.ReLU(),
            conv2d(ch_in=192, ch_out=192, kz=3, s=1), nn.ReLU(),
            #nn.LocalResponseNorm(),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )

    def forward(self, x):
        return self.in_bl_1(self.block(x))