import torch
import torch.nn.functional as F
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, ch_in, ch_out, strides):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=strides, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=strides),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )
        self.block1 = BasicBlock(64, 64, strides=2)
        self.block2 = BasicBlock(64, 128, strides=2)
        self.block3 = BasicBlock(128, 256, strides=2)
        self.block4 = BasicBlock(256, 512, strides=2)

        self.outlayer = nn.Linear(512*1*1, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x




