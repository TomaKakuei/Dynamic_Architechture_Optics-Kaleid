"""Shared multiscale spatial encoder used by the released models."""
from typing import Sequence
import torch
from torch import nn

class GroupFactorizedHead(nn.Module):
    """Independent MLP sub-heads per physical lens group (5 DOFs per group)."""

    def __init__(self, in_features: int, group_dofs: Sequence[int]):
        super().__init__()
        self.group_dofs = list(group_dofs)
        self.sub_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, 64),
                nn.GELU(),
                nn.Linear(64, dofs),
            )
            for dofs in self.group_dofs
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [head(x) for head in self.sub_heads]
        return torch.cat(outputs, dim=-1)


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise-separable convolution with BatchNorm and GELU."""

    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        p = kernel_size // 2
        self.dw = nn.Conv2d(in_c, in_c, kernel_size=kernel_size, stride=stride, padding=p, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class MultiScaleStem(nn.Module):
    """Multi-scale parallel convolution stem (3x3, 5x5, 7x7)."""

    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        branch_c = out_channels // 4
        self.b1 = nn.Conv2d(in_channels, branch_c, kernel_size=3, padding=1)
        self.b2 = nn.Conv2d(in_channels, branch_c, kernel_size=5, padding=2)
        self.b3 = nn.Conv2d(in_channels, branch_c, kernel_size=7, padding=3)
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(in_channels, branch_c, kernel_size=1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [self.b1(x), self.b2(x), self.b3(x), self.b4(x)]
        out = torch.cat(feats, dim=1)
        return self.act(self.bn(out))


class SpatialEncoder(nn.Module):
    """Arch-A: Multi-Scale Stem + 3-stage DW-Separable ResNet + Group-Factorized Head."""

    def __init__(self, in_fields: int, group_dofs: Sequence[int]):
        super().__init__()
        self.stem = MultiScaleStem(in_fields, out_channels=64)

        # 3 Stages of Depthwise-Separable ResNet
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv2d(64, 128, kernel_size=3, stride=2),
            DepthwiseSeparableConv2d(128, 128, kernel_size=3, stride=1),
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv2d(128, 256, kernel_size=3, stride=2),
            DepthwiseSeparableConv2d(256, 256, kernel_size=3, stride=1),
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv2d(256, 256, kernel_size=3, stride=2),
            DepthwiseSeparableConv2d(256, 256, kernel_size=3, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = GroupFactorizedHead(256, group_dofs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.pool(feat).flatten(1)
        return self.head(feat)
