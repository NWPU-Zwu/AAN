import os
import sys
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from kaiming import init_weights_model_kaiming
import random

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.a = 0.3


    def forward(self, input):
        imgmask = torch.ones_like(input)

        if random.uniform(0,1)>0.5:

            h = w = 70
            flag = 0
            while flag == 0:
                Se = random.randrange(0, int(h * w * 0.4))
                he = int(pow(Se * self.a, 0.5))
                we = int(pow(Se // self.a, 0.5))
                xe = random.randrange(0, w)
                ye = random.randrange(0, h)
                if xe + we <= w and ye + he <= h:
                    flag = 1
                    imgmask[:,:, xe:xe+we, ye:ye+he] = torch.randn_like(imgmask[:,:, xe:xe+we, ye:ye+he])

        return input * imgmask

class AAN(nn.Module):
    def __init__(self, channel, reduction=16):
        super(AAN, self).__init__()
        self.Lc = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel//reduction, out_channels=1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.AE = AE()
        self.fc = nn.Linear(384, 512)

    def forward(self, x):
        res = []
        for i in range(3):
            self.Lc.apply(init_weights_model_kaiming)
            map = self.Lc(x)
            map = self.AE(map)
            map = x * map.expand_as(x)
            map = F.adaptive_avg_pool2d(map, (1,1))
            map = map.view(map.size(0), -1)
            res.append(map)
        x = torch.cat(res, dim=-1)
        x = self.fc(x)
        return x
