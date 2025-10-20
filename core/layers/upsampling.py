from core.utils.gpu_utils import get_array_module, xp
import numpy as np


class UpSampling2D:
    def __init__(self, size: tuple[int, int] = (2, 2)) -> None:
        self.size = size
        self.input = None

    def __call__(self, x):
        """Forward pass with GPU support"""
        xp_module = get_array_module(x)
        self.input = x.copy()
        batch_size, h, w, channels = x.shape

        new_h = h * self.size[0]
        new_w = w * self.size[1]

        output = xp_module.zeros((batch_size, new_h, new_w, channels))
        for i in range(new_h):
            for j in range(new_w):
                output[
                    :,
                    i * self.size[0] : (i + 1) * self.size[0],
                    j * self.size[1] : (j + 1) * self.size[1],
                    :,
                ] = x[:, i : i + 1, j : j + 1, :]
        return output

    def backward(self, dout, learning_rate: float = None):
        """Backward pass with GPU support"""
        xp_module = get_array_module(dout)
        batch_size, new_h, new_w, channels = dout.shape
        h = self.input.shape[1]
        w = self.input.shape[2]

        dx = xp_module.zeros_like(self.input)
        for i in range(h):
            for j in range(w):
                dx[:, i, j, :] = xp_module.sum(
                    dout[
                        :,
                        i * self.size[0] : (i + 1) * self.size[0],
                        j * self.size[1] : (j + 1) * self.size[1],
                        :,
                    ],
                    axis=(1, 2),
                )
        return dx
