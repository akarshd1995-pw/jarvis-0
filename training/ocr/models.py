"""The two OCR models, both trained from scratch.

LineFinder: a small fully-convolutional encoder-decoder (~0.5M params). Grey image -> probability that each
            4x4 cell lies on a text line. Dilated convolutions give it the wide horizontal view text needs.
Reader:     CRNN (~6M params). Convolutions turn a 32-px-high line into a sequence of column features, a
            2-layer BiLSTM reads it, and CTC lets it learn characters without per-character positions.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .charset import N_CLASSES


def cbr(cin: int, cout: int, k: int = 3, s: int = 1, d: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, padding=d * (k // 2), dilation=d, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class LineFinder(nn.Module):
    STRIDE = 4

    def __init__(self, w: int = 24):
        super().__init__()
        self.e1 = nn.Sequential(cbr(1, w), cbr(w, w))                  # H
        self.e2 = nn.Sequential(cbr(w, 2 * w, s=2), cbr(2 * w, 2 * w))   # H/2
        self.e3 = nn.Sequential(cbr(2 * w, 3 * w, s=2), cbr(3 * w, 3 * w))   # H/4
        self.e4 = nn.Sequential(cbr(3 * w, 4 * w, s=2), cbr(4 * w, 4 * w))   # H/8
        self.e5 = nn.Sequential(cbr(4 * w, 5 * w, s=2), cbr(5 * w, 5 * w, d=2), cbr(5 * w, 5 * w, d=4))  # H/16
        self.d4 = cbr(5 * w + 4 * w, 4 * w)
        self.d3 = cbr(4 * w + 3 * w, 3 * w)
        self.head = nn.Sequential(cbr(3 * w, 2 * w), nn.Conv2d(2 * w, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N,1,H,W) in 0..1 with H, W multiples of 16 -> logits (N,1,H/4,W/4)."""
        x = x - 0.5
        f1 = self.e1(x)
        f2 = self.e2(f1)
        f3 = self.e3(f2)
        f4 = self.e4(f3)
        f5 = self.e5(f4)
        u4 = self.d4(torch.cat([F.interpolate(f5, size=f4.shape[-2:], mode="nearest"), f4], 1))
        u3 = self.d3(torch.cat([F.interpolate(u4, size=f3.shape[-2:], mode="nearest"), f3], 1))
        return self.head(u3)


class Reader(nn.Module):
    DOWN = 4                                   # output step = 4 input pixels

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            cbr(1, 64), nn.MaxPool2d(2, 2),                     # 16 x W/2
            cbr(64, 128), nn.MaxPool2d(2, 2),                   # 8 x W/4
            cbr(128, 256), cbr(256, 256), nn.MaxPool2d((2, 1), (2, 1)),   # 4 x W/4
            cbr(256, 384), cbr(384, 384), nn.MaxPool2d((2, 1), (2, 1)),   # 2 x W/4
            nn.Conv2d(384, 384, (2, 1), bias=False), nn.BatchNorm2d(384), nn.ReLU(inplace=True))  # 1 x W/4
        self.rnn = nn.LSTM(384, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.out = nn.Linear(2 * hidden, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N,1,32,W) in 0..1 -> log-probabilities (N, W/4, classes)."""
        f = self.cnn(x - 0.5).squeeze(2).transpose(1, 2)        # (N, T, 384)
        y, _ = self.rnn(f)
        return F.log_softmax(self.out(y), dim=-1)
