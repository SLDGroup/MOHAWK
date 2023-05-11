"""
Conv5 in PyTorch.
See the paper "FedMAX: Mitigating Activation Divergence for Accurate and Communication-Efficient Federated Learning"
for more details.
Reference: https://github.com/weichennone/FedMAX/blob/master/digit_object_recognition/models/Nets.py
"""

from torch import nn
import torch.nn.functional as F


class Conv5(nn.Module):

    def __init__(self, num_classes=10, channels=3, initializer=None):
        super(Conv5, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.apply(initializer)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool3(F.relu(self.conv5(x)))
        x = x.view(-1, 64 * 4 * 4)
        x_out = F.relu(self.fc1(x))
        x = self.fc2(x_out)
        return x, x_out
