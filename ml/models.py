"""
LungCancerDX – Model Definitions
All 5 architectures used in the ensemble
"""
import torch
import torch.nn as nn
from torchvision import models


def _build_resnet50(num_classes: int) -> nn.Module:
    net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    net.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(net.fc.in_features, num_classes),
    )
    return net


def _build_efficientnet_b0(num_classes: int) -> nn.Module:
    net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return net


def _build_densenet121(num_classes: int) -> nn.Module:
    net = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(net.classifier.in_features, num_classes),
    )
    return net


def _build_mobilenet_v3(num_classes: int) -> nn.Module:
    net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = net.classifier[3].in_features
    net.classifier[3] = nn.Linear(in_features, num_classes)
    return net


def _build_vgg16(num_classes: int) -> nn.Module:
    net = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    net.classifier[6] = nn.Linear(4096, num_classes)
    return net


FACTORY = {
    "ResNet50":       _build_resnet50,
    "EfficientNetB0": _build_efficientnet_b0,
    "DenseNet121":    _build_densenet121,
    "MobileNetV3":    _build_mobilenet_v3,
    "VGG16":          _build_vgg16,
}


def create_model(arch: str, num_classes: int = 3) -> nn.Module:
    if arch not in FACTORY:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from {list(FACTORY)}")
    return FACTORY[arch](num_classes)


def create_all_models(num_classes: int = 3, device: torch.device = None) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {name: fn(num_classes).to(device) for name, fn in FACTORY.items()}
