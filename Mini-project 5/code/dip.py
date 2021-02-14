import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F


class EncDec(nn.Module):

    def __init__(self):
        super(EncDec, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.deConv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deConv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.deConv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.deConv4 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.batchnorm8 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = F.relu(self.batchnorm4(self.conv4(x)))

        x = F.relu(self.batchnorm3(self.deConv1(self.Upsample(x))))
        x = F.relu(self.batchnorm2(self.deConv2(self.Upsample(x))))
#        x = data.ConcatDataset((x1, x2))
#        print(x.size())
        x = F.relu(self.batchnorm1(self.deConv3(self.Upsample(x))))
        x = F.relu(self.batchnorm8(self.deConv4(self.Upsample(x))))
        return x


