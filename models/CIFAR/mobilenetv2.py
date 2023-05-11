"""MobileNetV2 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
"""
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """expand + depthwise + pointwise"""

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(nn.Module):

    def __init__(self, factor=1, num_classes=10, channels=3, initializer=None):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.factor = factor
        # (expansion, out_planes, num_blocks, stride)
        self.cfg = [(1, int(16 * self.factor), 1, 1),
                    (6, int(24 * self.factor), 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
                    (6, int(32 * self.factor), 3, 2),
                    (6, int(64 * self.factor), 4, 2),
                    (6, int(96 * self.factor), 3, 1),
                    (6, int(160 * self.factor), 3, 2),
                    (6, int(320 * self.factor), 1, 1)]
        self.conv1 = nn.Conv2d(channels, int(32 * self.factor), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(32 * self.factor))
        self.layers = self._make_layers(in_planes=int(32 * self.factor))
        self.conv2 = nn.Conv2d(int(320 * self.factor), int(1280 * self.factor),
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(int(1280 * self.factor))
        self.linear = nn.Linear(int(1280 * self.factor), num_classes)
        self.apply(initializer)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride_ in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride_))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        x_out = out.view(out.size(0), -1)
        out = self.linear(x_out)
        return out, x_out
