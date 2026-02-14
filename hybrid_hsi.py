import torch
import torch.nn as nn


class HybridHSILite(nn.Module):
    """Lightweight 3D-2D hybrid backbone for HSI patches."""

    def __init__(self, in_channels: int = 30, feature_dim: int = 512):
        super().__init__()
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

        # Input channel count after 3D branch depends on spectral depth at runtime.
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
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = x.unsqueeze(1)  # [B, 1, C, H, W]
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        b, c3, d, h, w = x.shape
        x = x.view(b, c3 * d, h, w)
        x = self.conv2d(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


def hybrid_hsi_lite(in_channels: int = 30, feature_dim: int = 512):
    _ = in_channels
    return HybridHSILite(in_channels=in_channels, feature_dim=feature_dim)
