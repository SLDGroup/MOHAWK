"""VGG11/13/16/19 in Pytorch."""
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, batch_norm=True, channels=3, initializer=None):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], batch_norm=batch_norm, channels=channels)
        # self.classifier = nn.Linear(512, num_classes) # Kuangliu version
        self.classifier = nn.Sequential(  # FedNova version
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        self.apply(initializer)

    def forward(self, x):
        out = self.features(x)
        x_out = out.view(out.size(0), -1)
        out = self.classifier(x_out)
        return out, x_out

    @staticmethod
    def _make_layers(cfg_, batch_norm=True, channels=3):
        layers = []
        in_channels = channels
        for x in cfg_:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if batch_norm:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11bn(num_classes=10, channels=3, initializer=None):
    return VGG(vgg_name='VGG11', num_classes=num_classes, batch_norm=True, channels=channels, initializer=initializer)


def VGG13bn(num_classes=10, channels=3, initializer=None):
    return VGG(vgg_name='VGG13', num_classes=num_classes, batch_norm=True, channels=channels, initializer=initializer)


def VGG16bn(num_classes=10, channels=3, initializer=None):
    return VGG(vgg_name='VGG16', num_classes=num_classes, batch_norm=True, channels=channels, initializer=initializer)


def VGG19bn(num_classes=10, channels=3, initializer=None):
    return VGG(vgg_name='VGG19', num_classes=num_classes, batch_norm=True, channels=channels, initializer=initializer)


def VGG11(num_classes=10, channels=3, initializer=None):
    return VGG(vgg_name='VGG11', num_classes=num_classes, batch_norm=False, channels=channels, initializer=initializer)


def VGG13(num_classes=10, channels=3, initializer=None):
    return VGG(vgg_name='VGG13', num_classes=num_classes, batch_norm=False, channels=channels, initializer=initializer)


def VGG16(num_classes=10, channels=3, initializer=None):
    return VGG(vgg_name='VGG16', num_classes=num_classes, batch_norm=False, channels=channels, initializer=initializer)


def VGG19(num_classes=10, channels=3, initializer=None):
    return VGG(vgg_name='VGG19', num_classes=num_classes, batch_norm=False, channels=channels, initializer=initializer)
