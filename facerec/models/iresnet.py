"""IR-ResNet (ArcFace / InsightFace style) for 112x112 faces.

Differences from torchvision ResNet: 3x3 stride-1 stem (no maxpool), pre-activation
BN-Conv-BN-PReLU-Conv-BN residual units with the stride in the second conv, and an
output head BN -> Dropout -> Flatten -> FC(512*7*7 -> emb) -> BN.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv3x3(i, o, stride=1):
    return nn.Conv2d(i, o, 3, stride, 1, bias=False)


class IBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample = downsample

    def forward(self, x):
        idt = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            idt = self.downsample(x)
        return out + idt


class IResNet(nn.Module):
    def __init__(self, layers, emb_dim=512, dropout=0.0):
        super().__init__()
        self.inplanes = 64
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make(64, layers[0], 2)
        self.layer2 = self._make(128, layers[1], 2)
        self.layer3 = self._make(256, layers[2], 2)
        self.layer4 = self._make(512, layers[3], 2)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * 7 * 7, emb_dim)
        self.features = nn.BatchNorm1d(emb_dim, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():  # zero-init last BN of each residual branch
            if isinstance(m, IBasicBlock):
                nn.init.constant_(m.bn3.weight, 0)

    def _make(self, planes, blocks, stride):
        down = nn.Sequential(nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                             nn.BatchNorm2d(planes, eps=1e-5))
        layers = [IBasicBlock(self.inplanes, planes, stride, down)]
        self.inplanes = planes
        layers += [IBasicBlock(planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x.float() if not torch.is_autocast_enabled() else x)
        return self.features(x)


def build_backbone(name: str, emb_dim=512, dropout=0.0) -> IResNet:
    layers = {"r18": [2, 2, 2, 2], "r34": [3, 4, 6, 3], "r50": [3, 4, 14, 3]}[name]
    return IResNet(layers, emb_dim, dropout)
