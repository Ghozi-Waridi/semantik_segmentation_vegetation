import numpy as np


class ChannelAttention:
    def __init__(self, channels, reduction_ratio: int = 16) -> None:
        self.channels = channels
        self.reduction_ratio = reduction_ratio

        ### MLP for channel attention
        self.mlp_weights1 = (
            np.random.randn(channels, channels // reduction_ratio) * 0.01
        )
        self.mlp_weights2 = (
            np.random.randn(channels // reduction_ratio, channels) * 0.001
        )
        self.input = None
        self.channel_weights = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.input = x.copy()
        batch_size, h, w, c = x.shape

        gap = np.mean(x, axis=(1, 2))

        ## MLP
        hidden = np.maximum(0, np.dot(gap, self.mlp_weights1))
        self.channel_weights = 1 / (1 + np.exp(-np.dot(hidden, self.mlp_weights2)))

        channel_weight = self.channel_weights.reshape((batch_size, 1, 1, c))
        return x * channel_weight

    def backward(self, dout: np.ndarray, learning_rate: float) -> np.ndarray:
        batch_size, h, w, channels = self.input.shape

        d_input = dout * self.channel_weights.reshape(batch_size, 1, 1, channels)
        d_weight = np.sum(dout * self.input, axis=(1, 2))

        ## Gradient through MLP (simplified)
        sigmoid_deriv = self.channel_weights * (1 - self.channel_weights)
        d_mlp2 = d_weight * sigmoid_deriv

        ## Update channel_weights
        gap = np.mean(self.input, axis=(1, 2))
        hidden = np.maximum(0, np.dot(gap, self.mlp_weights1))

        d_mlp1 = np.dot(d_mlp2, self.mlp_weights2.T)
        d_mlp1[hidden <= 0] = 0  # ReLU backprop

        ## Weight Updates
        self.mlp_weights1 -= learning_rate * np.dot(gap.T, d_mlp1) / batch_size
        self.mlp_weights2 -= learning_rate * np.dot(hidden.T, d_mlp2) / batch_size

        return d_input


class SpatialAttention:
    def __init__(self, kernel_size: int = 7) -> None:
        self.kernel_size = kernel_size
        self.conv_weights = np.random.randn(kernel_size, kernel_size, 2, 1) * 0.01
        self.input = None
        self.spatial_weight = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.input = x.copy()
        batch_size, h, w, c = x.shape

        avg_pool = np.mean(x, axis=3, keepdims=True)
        max_pool = np.max(x, axis=3, keepdims=True)

        ### Concatenate along channel axis
        concat = np.concatenate([avg_pool, max_pool], axis=3)

        ## Apply Convolution
        self.spatial_weight = np.zeros((batch_size, h, w, 1))
        pad = self.kernel_size // 2

        concat_padded = np.pad(
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

                self.spatial_weight[:, i, j, 0] = np.dot(
                    x_slice.reshape(batch_size, -1), weight_flat
                ).flatten()

        # Apply sigmoid activation
        self.spatial_weight = 1 / (1 + np.exp(-self.spatial_weight))
        
        return x * self.spatial_weight

    def backward(self, dout: np.ndarray, learning_rate: float) -> np.ndarray:
        batch_size, h, w, c = self.input.shape

        d_input = dout * self.spatial_weight
        d_weight = np.sum(dout * self.input, axis=3, keepdims=True)

        sigmoid_deriv = self.spatial_weight * (1 - self.spatial_weight)
        d_weight = d_weight * sigmoid_deriv

        pad = self.kernel_size // 2

        avg_pool = np.mean(self.input, axis=3, keepdims=True)
        max_pool = np.max(self.input, axis=3, keepdims=True)
        concat = np.concatenate([avg_pool, max_pool], axis=3)

        concat_padded = np.pad(
            concat, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant"
        )
        d_conv_weights = np.zeros_like(self.conv_weights)

        for i in range(h):
            for j in range(w):
                h_start = i
                h_end = h_start + self.kernel_size
                w_start = j
                w_end = w_start + self.kernel_size

                slice = concat_padded[:, h_start:h_end, w_start:w_end, :]
                d_conv_weights += np.mean(slice, axis=0) * d_weight[:, i, j, 0].mean()

        self.conv_weights -= learning_rate * d_conv_weights / batch_size
        return d_input


class CBAM:
    def __init__(
        self, channel, reduction_ratio: int = 16, spatial_weight: int = 7
    ) -> None:
        self.channel_attention = ChannelAttention(channel, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_weight)
        self.input = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.input = x.copy()
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

    def backward(self, dout: np.ndarray, learning_rate: float) -> np.ndarray:
        dout = self.spatial_attention.backward(dout, learning_rate)
        dout = self.channel_attention.backward(dout, learning_rate)
        return dout

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward method untuk konsistensi dengan layer lain"""
        return self.__call__(x)
