#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/21 13:53
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   model.py
# @Desc     :   

from torch import nn, Tensor


class ConvolutionalModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        """
        Calculation for Convolutional Layers
            - h_out = (h + 2×padding - kernel_size) / stride + 1
            - w_out = (w + 2×padding - kernel_size) / stride + 1
        Calculation for Pooling Layers
            - h_out = (h - kernel_size) / stride + 1
            - w_out = (w - kernel_size) / stride + 1
        """
        self._model = nn.Sequential(
            # Input layer - First conv layer
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2),  # (6, 28, 28)
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # (6, 14, 14)
            # Second conv layer
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),  # (16, 10, 10)
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # (16, 5, 5)
            # Flatten the conv layer
            nn.Flatten(),
            # First linear layer
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            # Second linear layer
            nn.Linear(120, 84),
            nn.Sigmoid(),
            # Output layer
            nn.Linear(84, out_channels),
        )

        # Initialise the parameters
        self._model.apply(self._init_params)

    def _init_params(self, layer):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, X: Tensor) -> Tensor:
        return self._model(X)
