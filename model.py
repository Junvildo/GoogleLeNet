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
    def __init__(self, ch_in, ch_1x1, ch_3x3_r, ch_3x3, ch_5x5_r, ch_5x5, ch_pool) -> None:
        super().__init__()

        self.only_1x1 = seq(
            conv2d(ch_in=ch_in, ch_out=ch_1x1, kz=1, s=1), nn.ReLU()
        )
        self.reduce3 = seq(
            conv2d(ch_in=ch_in, ch_out=ch_3x3_r, kz=1, s=1), nn.ReLU(),
            conv2d(ch_in=ch_3x3_r, ch_out=ch_3x3, kz=3, s=1, p=1), nn.ReLU()
        )
        self.reduce5 = seq(
            conv2d(ch_in=ch_in, ch_out=ch_5x5_r, kz=1, s=1), nn.ReLU(),
            conv2d(ch_in=ch_5x5_r, ch_out=ch_5x5, kz=5, s=1, p=2), nn.ReLU()
        )
        self.pool = seq(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv2d(ch_in=ch_in, ch_out=ch_pool, kz=1, s=1), nn.ReLU()
        )

    def forward(self, x):
        y1 = self.only_1x1(x)
        y2 = self.reduce3(x)
        y3 = self.reduce5(x)
        y4 = self.pool(x)
        return torch.cat((y1, y2, y3, y4), dim=1)
    

class GoogLeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state_train = False
        self.in_3a = Inception(192,64,96,128,16,32,32)
        self.in_3b = Inception(256,128,128,192,32,96,64)
        self.in_4a = Inception(480,192,96,208,16,48,64)
        self.in_4b = Inception(512,160,112,224,24,64,64)
        self.in_4c = Inception(512,128,128,256,24,64,64)
        self.in_4d = Inception(512,112,144,288,32,64,64)
        self.in_4e = Inception(528,256,160,320,32,128,128)
        self.in_5a = Inception(832,256,160,320,32,128,128)
        self.in_5b = Inception(832,384,192,384,48,128,128)
        self.max_pool3x3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.max_pool7x7 = nn.MaxPool2d(kernel_size=7, stride=1)
        self.softmax_0 = seq(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Flatten(),
            nn.Linear(in_features=4*4*512, out_features=1024),
            nn.Linear(in_features=1024, out_features=1000),
            nn.Softmax(dim=1)
        )
        self.softmax_1 = seq(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Flatten(),
            nn.Linear(in_features=4*4*528, out_features=1024),
            nn.Linear(in_features=1024, out_features=1000),
            nn.Softmax(dim=1)
        )
        self.InitBlock = seq(
            conv2d(ch_in=3, ch_out=64, kz=7, s=2, p=3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv2d(ch_in=64, ch_out=192, kz=1, s=1), nn.ReLU(),
            conv2d(ch_in=192, ch_out=192, kz=3, s=1, p=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        self.EndBlock = seq(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self.in_5a,
            self.in_5b,
            nn.AvgPool2d(kernel_size=7, stride=1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(in_features=1024, out_features=1000),
            nn.Softmax(dim=1)
        )
        #     self.in_3a,
        #     self.in_3b,
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     self.in_4a,
        #     self.in_4b,
        #     self.in_4c,
        #     self.in_4d,
        #     self.in_4e,
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     self.in_5a,
        #     self.in_5b,
        #     nn.AvgPool2d(kernel_size=7, stride=1),
        #     nn.Flatten(),
        #     nn.Dropout(0.4),
        #     nn.Linear(in_features=1024, out_features=1000),
        #     nn.Softmax(dim=1)
        # )

    def forward(self, x):
        init = self.InitBlock(x)
        in_3 = self.in_3b(self.in_3a(init))
        middle = self.max_pool3x3(in_3)
        in_4_a = self.in_4a(middle)
        in_4_d = self.in_4d(self.in_4c(self.in_4b(in_4_a)))
        in_4 = self.in_4e(in_4_d)
        softmax_2 = self.EndBlock(in_4)
        if self.state_train == True:
            softmax_0 = self.softmax_0(in_4_a)
            softmax_1 = self.softmax_1(in_4_d)
            return (softmax_0,softmax_1,softmax_2)
        return softmax_2