"""
Standalone MiCrowdNet (paper Figure 7 reproduction), copia mínima de
geomcu-counting/student/models/baselines.py (commit del run
microwd_paper_original_..._seed7, 2026-05-13).

Mantenemos nombres y orden de submódulos EXACTOS para que load_state_dict
del best_mae.pt no requiera mapeo.

Arquitectura:
    4 ramas paralelas, cada una:
        MV2Block(k1, c1) -> MaxPool2 -> MV2Block(k2, c2) -> MaxPool2 -> MV2Block(k3, c3)
    Concat 30 ch -> 1x1 conv -> softplus -> density map

Input  (1, 3, H, W) float32 en [0, 1]
Output (1, 1, H/4, W/4), sum = predicted count.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=None, groups=1):
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else (k[0] // 2, k[1] // 2)
        self.conv = nn.Conv2d(cin, cout, k, stride=s, padding=p,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SamePadDepthwiseConv(nn.Module):
    """Depthwise conv con SAME padding explícito, soporta kernels pares (16)."""

    def __init__(self, channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                              stride=1, padding=0, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        k = self.kernel_size
        pad_total = k - 1
        pl, pr = pad_total // 2, pad_total - pad_total // 2
        pt, pb = pad_total // 2, pad_total - pad_total // 2
        x = F.pad(x, (pl, pr, pt, pb))
        return self.act(self.bn(self.conv(x)))


class MiCrowdMV2Block(nn.Module):
    """MV2 block: 1x1 expand (3k) -> kxk depthwise -> 1x1 project (k)."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 expansion=3, use_residual=True):
        super().__init__()
        hidden = out_channels * expansion
        self.use_residual = use_residual and in_channels == out_channels
        self.expand = ConvBNReLU(in_channels, hidden, k=1, p=0)
        self.depthwise = SamePadDepthwiseConv(hidden, kernel_size=kernel_size)
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.project(self.depthwise(self.expand(x)))
        if self.use_residual:
            y = y + x
        return self.out_act(y)


class MiCrowdBranch(nn.Module):
    """MV2 -> MaxPool -> MV2 -> MaxPool -> MV2 (stride 4 total)."""

    def __init__(self, in_channels, cfg, expansion=3):
        super().__init__()
        (k1, c1), (k2, c2), (k3, c3) = cfg
        self.block1 = MiCrowdMV2Block(in_channels, c1, k1, expansion,
                                      use_residual=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.block2 = MiCrowdMV2Block(c1, c2, k2, expansion,
                                      use_residual=(c1 == c2))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.block3 = MiCrowdMV2Block(c2, c3, k3, expansion,
                                      use_residual=(c2 == c3))

    def forward(self, x):
        x = self.block1(x); x = self.pool1(x)
        x = self.block2(x); x = self.pool2(x)
        return self.block3(x)


class MiCrowdNetPaperFullFrame(nn.Module):
    BRANCHES = [
        [(16, 12), (13, 12), (13, 6)],
        [(13, 24), (11, 24), (11, 6)],
        [(9, 16),  (7, 32),  (7, 8)],
        [(7, 20),  (5, 40),  (5, 10)],
    ]

    def __init__(self, in_channels=3, expansion=3, final_activation="softplus"):
        super().__init__()
        self.final_activation = final_activation
        self.branch1 = MiCrowdBranch(in_channels, self.BRANCHES[0], expansion)
        self.branch2 = MiCrowdBranch(in_channels, self.BRANCHES[1], expansion)
        self.branch3 = MiCrowdBranch(in_channels, self.BRANCHES[2], expansion)
        self.branch4 = MiCrowdBranch(in_channels, self.BRANCHES[3], expansion)
        self.out = nn.Conv2d(30, 1, kernel_size=1)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        x = torch.cat([b1, b2, b3, b4], dim=1)
        x = self.out(x)
        if self.final_activation == "softplus":
            x = F.softplus(x)
        elif self.final_activation == "relu":
            x = F.relu(x)
        elif self.final_activation == "none":
            pass
        else:
            raise ValueError(self.final_activation)
        return x


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = MiCrowdNetPaperFullFrame()
    print(f"params: {count_params(m):,}")
    with torch.no_grad():
        y = m(torch.randn(1, 3, 768, 1024))
    print(f"out: {tuple(y.shape)}  sum={y.sum().item():.3f}")
