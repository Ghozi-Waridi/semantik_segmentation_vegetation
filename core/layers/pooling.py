from core.utils.gpu_utils import get_array_module, xp
import numpy as np
import logging
from datetime import datetime
import json


class MaxPooling2D:
    def __init__(
        self, pool_size: tuple[int, int] = (2, 2), stride: tuple[int, int] = (2, 2)
    ) -> None:
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.mask = None

    def __call__(self, x):
        """Forward pass with GPU support"""
        xp_module = get_array_module(x)
        self.input = x.copy()
        batch_size, h, w, channels = x.shape

        out_h = (h - self.pool_size[0]) // self.stride[0] + 1
        out_w = (w - self.pool_size[1]) // self.stride[1] + 1

        output = xp_module.zeros((batch_size, out_h, out_w, channels))
        self.mask = xp_module.zeros_like(x)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride[0]
                h_end = h_start + self.pool_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.pool_size[1]

                x_slice = x[:, h_start:h_end, w_start:w_end, :]
                output[:, i, j, :] = xp_module.max(x_slice, axis=(1, 2))

                mask_slice = x_slice == output[:, i, j, :][:, None, None, :]
                self.mask[:, h_start:h_end, w_start:w_end, :] = mask_slice

        return output

    def backward(self, dout, learning_rate: float):
        """Backward pass with GPU support"""
        xp_module = get_array_module(dout)
        # Gradient must match input shape
        dx = xp_module.zeros_like(self.input)
        batch_size, out_h, out_w, channels = dout.shape

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride[0]
                h_end = h_start + self.pool_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.pool_size[1]

                dx[:, h_start:h_end, w_start:w_end, :] += (
                    self.mask[:, h_start:h_end, w_start:w_end, :]
                    * dout[:, i : i + 1, j : j + 1, :]
                )
        return dx
