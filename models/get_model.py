from models.CIFAR.efficientnetb0 import EfficientNetB0
from models.CIFAR.fedmax_cnn import Conv5
from models.CIFAR.mobilenetv1 import MobileNetV1
from models.CIFAR.mobilenetv2 import MobileNetV2
from models.CIFAR.resnet import ResNet14, ResNet20, ResNet32, ResNet44, ResNet56, ResNet110
from models.CIFAR.resnet_big import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.CIFAR.vgg import VGG11, VGG13, VGG16, VGG19, VGG11bn, VGG13bn, VGG16bn, VGG19bn
from torch import nn

def kaiming_normal(w):
    if isinstance(w, nn.Linear) or isinstance(w, nn.Conv2d):
        nn.init.kaiming_normal_(w.weight)

def get_model(model_name):
    dataset_name, model_name = model_name.split('_')
    num_classes = 10
    if dataset_name == "cifar100":
        num_classes = 100
    elif dataset_name == "emnist":
        num_classes = 62

    if dataset_name == "mnist" or dataset_name == "emnist":
        num_channels = 1
    elif dataset_name == "cifar10" or dataset_name == "cifar100":
        num_channels = 3
    else:
        print(f"Dataset name: {dataset_name} is not okay.")
        exit(-1)
    if model_name == "conv5":
        return Conv5(num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "efficientnetb0":
        return EfficientNetB0(num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "mbnv1":
        return MobileNetV1(factor=1, num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "mbnv1075":
        return MobileNetV1(factor=0.75, num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "mbnv105":
        return MobileNetV1(factor=0.75, num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "mbnv1025":
        return MobileNetV1(factor=0.25, num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "mbnv2":
        return MobileNetV2(factor=1, num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "mbnv2075":
        return MobileNetV2(factor=0.75, num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "mbnv205":
        return MobileNetV2(factor=0.75, num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "mbnv2025":
        return MobileNetV2(factor=0.25, num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "resnet14":
        return ResNet14(num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "resnet20":
        return ResNet20(num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "resnet32":
        return ResNet32(num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "resnet44":
        return ResNet44(num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "resnet56":
        return ResNet56(num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "resnet110":
        return ResNet110(num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "resnet18":
        return ResNet18(num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "resnet34":
        return ResNet34(num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "resnet50":
        return ResNet50(num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "resnet101":
        return ResNet101(num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "resnet152":
        return ResNet152(num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "vgg11":
        return VGG11(num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "vgg11bn":
        return VGG11bn(num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "vgg13":
        return VGG13(num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "vgg13bn":
        return VGG13bn(num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "vgg16":
        return VGG16(num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "vgg16bn":
        return VGG16bn(num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "vgg19":
        return VGG19(num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    elif model_name == "vgg19bn":
        return VGG19bn(num_classes=num_classes, channels=num_channels, initializer=kaiming_normal)
    else:
        raise NotImplementedError(f'Model {dataset_name}_{model_name} not implemented yet')
