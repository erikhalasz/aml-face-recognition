"""Classification heads used only during training (discarded at deployment)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """Additive angular margin: logit_y = s*cos(theta_y + m), others s*cos(theta_j).
    Computed in fp32. Uses the standard fallback cos(theta)-m*sin(m) when theta+m > pi."""

    def __init__(self, emb_dim, n_classes, s=64.0, m=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, emb_dim))
        nn.init.normal_(self.weight, std=0.01)
        self.s, self.m = s, m
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb, labels):
        cos = F.linear(F.normalize(emb.float()), F.normalize(self.weight.float())).clamp(-1 + 1e-7, 1 - 1e-7)
        cos_y = cos.gather(1, labels[:, None])
        sin_y = torch.sqrt(1.0 - cos_y * cos_y)
        phi = cos_y * self.cos_m - sin_y * self.sin_m
        phi = torch.where(cos_y > self.th, phi, cos_y - self.mm)
        logits = cos.scatter(1, labels[:, None], phi) * self.s
        return F.cross_entropy(logits, labels), cos.detach()


class SoftmaxHead(nn.Module):
    """Plain softmax baseline: linear layer with bias + cross-entropy, no normalisation or margin."""

    def __init__(self, emb_dim, n_classes, **_):
        super().__init__()
        self.fc = nn.Linear(emb_dim, n_classes)

    def forward(self, emb, labels):
        logits = self.fc(emb.float())
        return F.cross_entropy(logits, labels), logits.detach()


def build_head(name, emb_dim, n_classes, s=64.0, m=0.5):
    return {"arcface": ArcFaceHead, "softmax": SoftmaxHead}[name](emb_dim, n_classes, s=s, m=m)
