import torch
import torch.nn as nn
import torchvision


class ResNetPixelBackbone(nn.Module):
    def __init__(self, in_channels=30, feature_dim=512):
        super().__init__()
        _ = feature_dim
        base = torchvision.models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        self.base = base
        self.feature_dim = base.fc.in_features

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        x = self.base.avgpool(x)
        return torch.flatten(x, 1)
