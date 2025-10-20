from core.utils.gpu_utils import get_array_module, xp
import numpy as np


class ReLU:
    def __init__(self):
        self.input = None
        self.output = None
    
    def __call__(self, X):
        """Make ReLU callable"""
        return self.forward(X)
    
    def forward(self, X):
        """
        Forward pass untuk ReLU: f(x) = max(0, x)

        Args:
            X: Input tensor

        Returns:
            Output setelah aplikasi ReLU
        """
        xp_module = get_array_module(X)
        self.input = X.copy()
        self.output = xp_module.maximum(0, X)
        return self.output

    def backward(self, dout, learning_rate=None):
        """
        Backward pass untuk ReLU

        Args:
            dout: Gradient dari layer berikutnya
            learning_rate: Not used, for API consistency

        Returns:
            Gradient untuk layer sebelumnya
        """
        if self.input is None:
            raise ValueError("Must call forward before backward")

        
        dX = dout * (self.input > 0)
        return dX


class Sigmoid:
    def __init__(self):
        self.input = None
        self.output = None

    def __call__(self, X):
        """Make Sigmoid callable"""
        return self.forward(X)

    def forward(self, X):
        """
        Forward pass untuk Sigmoid: f(x) = 1 / (1 + exp(-x))

        Args:
            X: Input tensor

        Returns:
            Output setelah aplikasi Sigmoid
        """
        xp_module = get_array_module(X)
        self.input = X.copy()

        
        X_clipped = xp_module.clip(X, -50, 50) 
        self.output = 1 / (1 + xp_module.exp(-X_clipped))
        return self.output

    def backward(self, dout, learning_rate=None):
        """
        Backward pass untuk Sigmoid

        Args:
            dout: Gradient dari layer berikutnya
            learning_rate: Not used, for API consistency

        Returns:
            Gradient untuk layer sebelumnya
        """
        if self.input is None:
            raise ValueError("Must call forward before backward")

        
        sigmoid_output = self.output
        dX = dout * sigmoid_output * (1 - sigmoid_output)
        return dX


class Tanh:
    def __init__(self):
        self.input = None
        self.output = None

    def __call__(self, X):
        """Make Tanh callable"""
        return self.forward(X)

    def forward(self, X):
        """
        Forward pass untuk Tanh: f(x) = tanh(x)

        Args:
            X: Input tensor

        Returns:
            Output setelah aplikasi Tanh
        """
        xp_module = get_array_module(X)
        self.input = X.copy()
        self.output = xp_module.tanh(X)
        return self.output

    def backward(self, dout, learning_rate=None):
        """
        Backward pass untuk Tanh

        Args:
            dout: Gradient dari layer berikutnya
            learning_rate: Not used, for API consistency

        Returns:
            Gradient untuk layer sebelumnya
        """
        if self.input is None:
            raise ValueError("Must call forward before backward")

        
        dX = dout * (1 - self.output**2)
        return dX


class LeakyReLU:
    def __init__(self, alpha=0.01):
        """
        Leaky ReLU: f(x) = x if x > 0 else alpha * x

        Args:
            alpha: Slope untuk nilai negatif (default: 0.01)
        """
        self.alpha = alpha
        self.input = None
        self.output = None

    def __call__(self, X):
        """Make LeakyReLU callable"""
        return self.forward(X)

    def forward(self, X):
        """
        Forward pass untuk Leaky ReLU
        """
        xp_module = get_array_module(X)
        self.input = X.copy()
        self.output = xp_module.where(X > 0, X, self.alpha * X)
        return self.output

    def backward(self, dout, learning_rate=None):
        """
        Backward pass untuk Leaky ReLU
        
        Args:
            dout: Gradient dari layer berikutnya
            learning_rate: Not used, for API consistency
        """
        xp_module = get_array_module(dout)
        if self.input is None:
            raise ValueError("Must call forward before backward")

        
        dX = dout * xp_module.where(self.input > 0, 1, self.alpha)
        return dX


class Softmax:
    def __init__(self):
        self.input = None
        self.output = None

    def __call__(self, X):
        """Make Softmax callable"""
        return self.forward(X)

    def forward(self, X):
        """
        Forward pass untuk Softmax dengan numerical stability

        Args:
            X: Input tensor

        Returns:
            Probabilities setelah aplikasi Softmax
        """
        xp_module = get_array_module(X)
        self.input = X.copy()

        
        exp_X = xp_module.exp(X - xp_module.max(X, axis=-1, keepdims=True))
        self.output = exp_X / xp_module.sum(exp_X, axis=-1, keepdims=True)
        return self.output

    def backward(self, dout, learning_rate=None):
        """
        Backward pass untuk Softmax
        NOTE: Biasanya digunakan dengan cross entropy loss dimana gradient simplified
        
        Args:
            dout: Gradient dari layer berikutnya
            learning_rate: Not used, for API consistency
        """
        
        
        return dout
