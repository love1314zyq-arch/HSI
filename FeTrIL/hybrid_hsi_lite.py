import torch
import torch.nn as nn


class HybridHSILiteBackbone(nn.Module):
    def __init__(self, in_channels=30, feature_dim=512):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=(3, 1, 1), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.conv2d = nn.Sequential(
            nn.LazyConv2d(128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        b, c3, d, h, w = x.shape
        x = x.view(b, c3 * d, h, w)
        x = self.conv2d(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return x.view(x.size(0), -1)
