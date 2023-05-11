"""MobileNet in PyTorch.
See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
"""
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """Depthwise conv + Pointwise conv"""

    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNetV1(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1

    def __init__(self, factor=1, num_classes=10, channels=3, initializer=None):
        super(MobileNetV1, self).__init__()
        self.factor = factor
        self.cfg = [int(64*self.factor), (int(128*self.factor), 2), int(128*self.factor),
                    (int(256*self.factor), 2), int(256*self.factor),
                    (int(512*self.factor), 2), int(512*self.factor), int(512*self.factor), int(512*self.factor),
                    int(512*self.factor), int(512*self.factor),
                    (int(1024*self.factor), 2), int(1024*self.factor)]

        self.conv1 = nn.Conv2d(channels, int(32*self.factor), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(32*self.factor))
        self.layers = self._make_layers(in_planes=int(32*self.factor))
        self.linear = nn.Linear(int(1024*self.factor), num_classes)
        self.apply(initializer)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        x_out = out.view(out.size(0), -1)
        out = self.linear(x_out)
        return out, x_out
