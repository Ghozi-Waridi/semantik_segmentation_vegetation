from core.utils.gpu_utils import get_array_module, xp
import numpy as np


class ChannelAttention:
    def __init__(self, channels, reduction_ratio: int = 16) -> None:
        self.channels = channels
        self.reduction_ratio = reduction_ratio

        ### MLP for channel attention
        self.mlp_weights1 = (
            xp.random.randn(channels, channels // reduction_ratio) * 0.01
        )
        self.mlp_weights2 = (
            xp.random.randn(channels // reduction_ratio, channels) * 0.001
        )
        self.input = None
        self.channel_weights = None

    def __call__(self, x):
        """Forward pass with GPU support"""
        xp_module = get_array_module(x)
        self.input = x.copy()
        batch_size, h, w, c = x.shape

        gap = xp_module.mean(x, axis=(1, 2))

        ## MLP
        hidden = xp_module.maximum(0, xp_module.dot(gap, self.mlp_weights1))
        self.channel_weights = 1 / (1 + xp_module.exp(-xp_module.dot(hidden, self.mlp_weights2)))

        channel_weight = self.channel_weights.reshape((batch_size, 1, 1, c))
        return x * channel_weight

    def backward(self, dout, learning_rate: float):
        """Backward pass with GPU support"""
        xp_module = get_array_module(dout)
        batch_size, h, w, channels = self.input.shape

        d_input = dout * self.channel_weights.reshape(batch_size, 1, 1, channels)
        d_weight = xp_module.sum(dout * self.input, axis=(1, 2))

        ## Gradient through MLP (simplified)
        sigmoid_deriv = self.channel_weights * (1 - self.channel_weights)
        d_mlp2 = d_weight * sigmoid_deriv

        ## Update channel_weights
        gap = xp_module.mean(self.input, axis=(1, 2))
        hidden = xp_module.maximum(0, xp_module.dot(gap, self.mlp_weights1))

        d_mlp1 = xp_module.dot(d_mlp2, self.mlp_weights2.T)
        d_mlp1[hidden <= 0] = 0  # ReLU backprop

        ## Weight Updates
        self.mlp_weights1 -= learning_rate * xp_module.dot(gap.T, d_mlp1) / batch_size
        self.mlp_weights2 -= learning_rate * xp_module.dot(hidden.T, d_mlp2) / batch_size

        return d_input


class SpatialAttention:
    def __init__(self, kernel_size: int = 7) -> None:
        self.kernel_size = kernel_size
        self.conv_weights = xp.random.randn(kernel_size, kernel_size, 2, 1) * 0.01
        self.input = None
        self.spatial_weight = None

    def __call__(self, x):
        """Forward pass with GPU support"""
        xp_module = get_array_module(x)
        self.input = x.copy()
        batch_size, h, w, c = x.shape

        avg_pool = xp_module.mean(x, axis=3, keepdims=True)
        max_pool = xp_module.max(x, axis=3, keepdims=True)

        concat = xp_module.concatenate([avg_pool, max_pool], axis=3)

        self.spatial_weight = xp_module.zeros((batch_size, h, w, 1))
        pad = self.kernel_size // 2

        concat_padded = xp_module.pad(
            concat, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant"
        )

        for i in range(h):
            for j in range(w):
                h_start = i
                h_end = h_start + self.kernel_size
                w_start = j
                w_end = w_start + self.kernel_size

                x_slice = concat_padded[:, h_start:h_end, w_start:w_end, :]

                weight_flat = self.conv_weights.reshape(-1, 1)

                self.spatial_weight[:, i, j, 0] = xp_module.dot(
                    x_slice.reshape(batch_size, -1), weight_flat
                ).flatten()

        # Apply sigmoid to spatial weight
        self.spatial_weight = 1 / (1 + xp_module.exp(-self.spatial_weight))

        return x * self.spatial_weight

    def backward(self, dout, learning_rate: float):
        """Backward pass with GPU support"""
        xp_module = get_array_module(dout)
        batch_size, h, w, c = self.input.shape

        # Gradient w.r.t input of y = x * A (A = spatial_weight after sigmoid)
        d_input = dout * self.spatial_weight

        # Gradient w.r.t pre-activation (before sigmoid): dL/dP = dL/dA * sigmoid'(P)
        dA = xp_module.sum(dout * self.input, axis=3, keepdims=True)  # shape (b,h,w,1)
        sigmoid_deriv = self.spatial_weight * (1 - self.spatial_weight)
        dP = dA * sigmoid_deriv  # (b,h,w,1)

        pad = self.kernel_size // 2

        # Recompute the concat used in forward (avg+max across channels)
        avg_pool = xp_module.mean(self.input, axis=3, keepdims=True)
        max_pool = xp_module.max(self.input, axis=3, keepdims=True)
        concat = xp_module.concatenate([avg_pool, max_pool], axis=3)  # (b,h,w,2)

        concat_padded = xp_module.pad(
            concat, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant"
        )
        d_conv_weights = xp_module.zeros_like(self.conv_weights)  # (k,k,2,1)

        # Accumulate gradients for conv weights
        for i in range(h):
            for j in range(w):
                h_start = i
                h_end = h_start + self.kernel_size
                w_start = j
                w_end = w_start + self.kernel_size

                window = concat_padded[:, h_start:h_end, w_start:w_end, :]  # (b,k,k,2)
                # Sum over batch: (k,k,2) contribution, then expand to (k,k,2,1)
                contrib = xp_module.einsum("bhwc,b->hwc", window, dP[:, i, j, 0])[..., None]
                d_conv_weights += contrib

        # Update conv weights
        self.conv_weights -= learning_rate * (d_conv_weights / max(batch_size, 1))
        return d_input


class CBAM:
    def __init__(
        self, channel, reduction_ratio: int = 16, spatial_weight: int = 7
    ) -> None:
        self.channel_attention = ChannelAttention(channel, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_weight)
        self.input = None

    def __call__(self, x):
        """Forward pass with GPU support"""
        self.input = x.copy()
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

    def backward(self, dout, learning_rate: float):
        """Backward pass with GPU support"""
        dout = self.spatial_attention.backward(dout, learning_rate)
        dout = self.channel_attention.backward(dout, learning_rate)
        return dout
