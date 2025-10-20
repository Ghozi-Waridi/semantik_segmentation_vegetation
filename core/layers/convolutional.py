from core.utils.gpu_utils import get_array_module, to_gpu, to_cpu, xp

from core.utils.logging_config import get_logger
import logging

logger = get_logger(__name__)


class Conv2D:
    def __init__(
        self,
        out_channels: int,
        kernel_size,
        strides: tuple[int, int] = (1, 1),  
        padding: str = "same",
        activations: str = "relu",
    ) -> None:
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activations = activations

        
        self.weights = None
        self.bias = None

        

        logger.info(f"Conv2D Layer Initialized:")
        logger.info(f"  - Output Channels: {self.out_channels}")
        logger.info(f"  - Kernel Size: {self.kernel_size}")
        logger.info(f"  - Strides: {self.strides}")
        logger.info(f"  - Padding: {self.padding}")
        logger.info(f"  - Activation: {self.activations}")

    def initialize(self, input_shape) -> None:
        self.input_shape = input_shape
        input_channel = input_shape[-1]  

        std = float(xp.sqrt(2.0 / (input_channel * self.kernel_size[0] * self.kernel_size[1])))
        self.weights = (
            xp.random.randn(
                self.kernel_size[0],
                self.kernel_size[1],
                input_channel,
                self.out_channels,
            )
            * std
        )
        self.bias = xp.zeros((1, 1, 1, self.out_channels))

        
        logger.info(f"Parameters Initialized:")
        logger.info(f"  - Input Shape: {input_shape}")
        logger.info(f"  - Weights Shape: {self.weights.shape}")
        logger.info(f"  - Bias Shape: {self.bias.shape}")
        logger.info(
            f"  - Weights Stats - Mean: {float(xp.mean(self.weights)):.6f}, Std: {float(xp.std(self.weights)):.6f}"
        )
        logger.info(
            f"  - Bias Stats - Mean: {float(xp.mean(self.bias)):.6f}, Std: {float(xp.std(self.bias)):.6f}"
        )

    def pad_image(self, x, pad):
        """Pad image with GPU support"""
        xp_module = get_array_module(x)
        padded_x = xp_module.pad(
            x, ((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)), mode="constant"
        )

        logger.debug(f"Padding Applied:")
        logger.debug(f"  - Input Shape: {x.shape}")
        logger.debug(f"  - Padding: {pad}")
        logger.debug(f"  - Padded Shape: {padded_x.shape}")

        return padded_x

    def __call__(self, x):
        """Forward pass with GPU support"""
        xp_module = get_array_module(x)
        self.x = x
        batch_size, h, w, in_channels = x.shape

        
        
        
        
        
        

        if self.weights is None:
            self.initialize(x.shape)

        self.input = x.copy()

        if self.padding == "same":
            pad_h = (self.kernel_size[0] - 1) // 2
            pad_w = (self.kernel_size[1] - 1) // 2
        else:
            pad_h, pad_w = 0, 0

        x_padded = self.pad_image(x, (pad_h, pad_w))

        out_h = (h + 2 * pad_h - self.kernel_size[0]) // self.strides[0] + 1
        out_w = (w + 2 * pad_w - self.kernel_size[1]) // self.strides[1] + 1

        output = xp_module.zeros((batch_size, out_h, out_w, self.out_channels))

        
        logging.info(f"Convolution Details:")
        logging.info(f"  - Original Size: {h}x{w}")
        logging.info(f"  - Padded Size: {x_padded.shape[1]}x{x_padded.shape[2]}")
        logging.info(f"  - Output Size: {out_h}x{out_w}")
        logging.info(f"  - Strides: {self.strides}")

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.strides[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.strides[1]
                w_end = w_start + self.kernel_size[1]

                x_slice = x_padded[:, h_start:h_end, w_start:w_end, :]

                
                x_slice = x_slice.reshape(batch_size, -1)
                
                
                weight_flat = self.weights.reshape(-1, self.out_channels)

                output[:, i, j, :] = xp_module.dot(x_slice, weight_flat) + self.bias
        self.output = output.copy()

        
        
        
        
        
        
        return output

    def backward(self, dout, learning_rate: float = 0.001):
        """Backward pass with GPU support"""
        xp_module = get_array_module(dout)
        batch_size, h, w, channels = self.input.shape

        
        
        
        
        
        

        if self.padding == "same":
            pad_h = (self.kernel_size[0] - 1) // 2
            pad_w = (self.kernel_size[1] - 1) // 2
        else:
            pad_h, pad_w = 0, 0

        dx = xp_module.zeros_like(self.input)
        dw = xp_module.zeros_like(self.weights)
        db = xp_module.zeros_like(self.bias)

        x_padded = self.pad_image(self.input, (pad_h, pad_w))
        dx_padded = self.pad_image(dx, (pad_h, pad_w))

        out_h, out_w = dout.shape[1], dout.shape[2]

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.strides[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.strides[1]
                w_end = w_start + self.kernel_size[1]

                
                x_slice = x_padded[:, h_start:h_end, w_start:w_end, :]

                
                
                dw += xp_module.einsum("bhwc,bo->hwco", x_slice, dout[:, i, j, :])

                
                
                dx_padded[:, h_start:h_end, w_start:w_end, :] += xp_module.einsum(
                    "hwco,bo->bhwc", self.weights, dout[:, i, j, :]
                )
        db = xp_module.sum(dout, axis=(0, 1, 2), keepdims=True)
        if self.padding == "same":
            dx = dx_padded[:, pad_h : pad_h + h, pad_w : pad_w + w, :]
        else:
            dx = dx_padded

        
        logger.info(f"Gradient Statistics:")
        logger.info(
            f"  - dW Stats - Min: {float(xp_module.min(dw)):.6f}, Max: {float(xp_module.max(dw)):.6f}, Mean: {float(xp_module.mean(dw)):.6f}"
        )
        logger.info(
            f"  - dB Stats - Min: {float(xp_module.min(db)):.6f}, Max: {float(xp_module.max(db)):.6f}, Mean: {float(xp_module.mean(db)):.6f}"
        )
        logger.info(
            f"  - dX Stats - Min: {float(xp_module.min(dx)):.6f}, Max: {float(xp_module.max(dx)):.6f}, Mean: {float(xp_module.mean(dx)):.6f}"
        )

        self.weights -= learning_rate * dw / batch_size
        self.bias -= learning_rate * db / batch_size

        return dx

    def get_layer_info(self) -> dict:
        """Return layer information for logging purposes"""
        info = {
            "layer_type": "Conv2D",
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "activations": self.activations,
            "weights_shape": self.weights.shape if self.weights is not None else None,
            "bias_shape": self.bias.shape if self.bias is not None else None,
            "input_shape": getattr(self, "input_shape", None),
            "output_shape": (
                getattr(self, "output", None).shape
                if hasattr(self, "output") and self.output is not None
                else None
            ),
        }
        return info
