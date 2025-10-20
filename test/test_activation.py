"""
Unit tests for activation layers.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.activation.activation import ReLU, Sigmoid, Softmax, Tanh, LeakyReLU
from core.utils.logging_config import setup_logging

logger = setup_logging(module_name='test_activation', log_dir='logs/tests')


class TestReLU(unittest.TestCase):
    """Test cases for ReLU activation"""
    
    def setUp(self):
        self.X = np.array([[-2, -1, 0, 1, 2], [3, -4, 5, -6, 7]], dtype=np.float32)
        logger.info(f"ReLU test setup: Input shape {self.X.shape}")
    
    def test_forward(self):
        """Test ReLU forward pass"""
        logger.info("Testing ReLU forward pass")
        
        relu = ReLU()
        output = relu.forward(self.X)
        
        expected = np.array([[0, 0, 0, 1, 2], [3, 0, 5, 0, 7]], dtype=np.float32)
        np.testing.assert_array_equal(output, expected)
        
        logger.info("✓ ReLU forward test passed")
    
    def test_backward(self):
        """Test ReLU backward pass"""
        logger.info("Testing ReLU backward pass")
        
        relu = ReLU()
        output = relu.forward(self.X)
        
        dout = np.ones_like(output)
        dx = relu.backward(dout, learning_rate=0.01)
        
        # Gradient should be 1 where input > 0, 0 otherwise
        expected_mask = (self.X > 0).astype(np.float32)
        np.testing.assert_array_equal(dx, expected_mask)
        
        logger.info("✓ ReLU backward test passed")
    
    def test_callable(self):
        """Test ReLU callable interface"""
        logger.info("Testing ReLU callable interface")
        
        relu = ReLU()
        output1 = relu(self.X)
        output2 = relu.forward(self.X)
        
        np.testing.assert_array_equal(output1, output2)
        
        logger.info("✓ ReLU callable test passed")


class TestSigmoid(unittest.TestCase):
    """Test cases for Sigmoid activation"""
    
    def setUp(self):
        self.X = np.array([[0, 1, -1]], dtype=np.float32)
        logger.info(f"Sigmoid test setup: Input shape {self.X.shape}")
    
    def test_forward(self):
        """Test Sigmoid forward pass"""
        logger.info("Testing Sigmoid forward pass")
        
        sigmoid = Sigmoid()
        output = sigmoid.forward(self.X)
        
        # At x=0, sigmoid should be ~0.5
        self.assertAlmostEqual(output[0, 0], 0.5, places=5)
        
        # Output should be in [0, 1]
        self.assertTrue(np.all(output >= 0))
        self.assertTrue(np.all(output <= 1))
        
        logger.info(f"✓ Sigmoid forward test passed - Output: {output}")
    
    def test_backward(self):
        """Test Sigmoid backward pass"""
        logger.info("Testing Sigmoid backward pass")
        
        sigmoid = Sigmoid()
        output = sigmoid.forward(self.X)
        
        dout = np.ones_like(output)
        dx = sigmoid.backward(dout, learning_rate=0.01)
        
        # dx should have same shape as input
        self.assertEqual(dx.shape, self.X.shape)
        
        logger.info("✓ Sigmoid backward test passed")


class TestSoftmax(unittest.TestCase):
    """Test cases for Softmax activation"""
    
    def setUp(self):
        self.X = np.array([[[1, 2, 3]]], dtype=np.float32)  # (1,1,3)
        logger.info(f"Softmax test setup: Input shape {self.X.shape}")
    
    def test_forward(self):
        """Test Softmax forward pass"""
        logger.info("Testing Softmax forward pass")
        
        softmax = Softmax()
        output = softmax.forward(self.X)
        
        # Probabilities should sum to 1 along last axis
        sums = np.sum(output, axis=-1)
        np.testing.assert_array_almost_equal(sums, 1.0)
        
        # All values should be in [0, 1]
        self.assertTrue(np.all(output >= 0))
        self.assertTrue(np.all(output <= 1))
        
        logger.info(f"✓ Softmax forward test passed")
    
    def test_numerical_stability(self):
        """Test Softmax numerical stability with large values"""
        logger.info("Testing Softmax numerical stability")
        
        X_large = np.array([[[100, 101, 102]]], dtype=np.float32)
        
        softmax = Softmax()
        output = softmax.forward(X_large)
        
        # Should not produce NaN or Inf
        self.assertFalse(np.any(np.isnan(output)))
        self.assertFalse(np.any(np.isinf(output)))
        
        # Should still be valid probabilities
        sums = np.sum(output, axis=-1)
        np.testing.assert_array_almost_equal(sums, 1.0)
        
        logger.info("✓ Softmax stability test passed")


class TestLeakyReLU(unittest.TestCase):
    """Test cases for LeakyReLU activation"""
    
    def setUp(self):
        self.X = np.array([[-2, -1, 0, 1, 2]], dtype=np.float32)
        self.alpha = 0.1
        logger.info(f"LeakyReLU test setup: Input shape {self.X.shape}, alpha={self.alpha}")
    
    def test_forward(self):
        """Test LeakyReLU forward pass"""
        logger.info("Testing LeakyReLU forward pass")
        
        leaky = LeakyReLU(alpha=self.alpha)
        output = leaky.forward(self.X)
        
        # Negative values should be scaled by alpha
        expected = np.array([[-0.2, -0.1, 0, 1, 2]], dtype=np.float32)
        np.testing.assert_array_almost_equal(output, expected)
        
        logger.info("✓ LeakyReLU forward test passed")
    
    def test_backward(self):
        """Test LeakyReLU backward pass"""
        logger.info("Testing LeakyReLU backward pass")
        
        leaky = LeakyReLU(alpha=self.alpha)
        output = leaky.forward(self.X)
        
        dout = np.ones_like(output)
        dx = leaky.backward(dout, learning_rate=0.01)
        
        # Should have alpha for negative, 1 for positive
        expected_mask = np.where(self.X > 0, 1.0, self.alpha)
        np.testing.assert_array_almost_equal(dx, expected_mask)
        
        logger.info("✓ LeakyReLU backward test passed")


class TestTanh(unittest.TestCase):
    """Test cases for Tanh activation"""
    
    def setUp(self):
        self.X = np.array([[0, 1, -1]], dtype=np.float32)
        logger.info(f"Tanh test setup: Input shape {self.X.shape}")
    
    def test_forward(self):
        """Test Tanh forward pass"""
        logger.info("Testing Tanh forward pass")
        
        tanh = Tanh()
        output = tanh.forward(self.X)
        
        # At x=0, tanh should be 0
        self.assertAlmostEqual(output[0, 0], 0, places=5)
        
        # Output should be in [-1, 1]
        self.assertTrue(np.all(output >= -1))
        self.assertTrue(np.all(output <= 1))
        
        logger.info(f"✓ Tanh forward test passed")
    
    def test_backward(self):
        """Test Tanh backward pass"""
        logger.info("Testing Tanh backward pass")
        
        tanh = Tanh()
        output = tanh.forward(self.X)
        
        dout = np.ones_like(output)
        dx = tanh.backward(dout, learning_rate=0.01)
        
        self.assertEqual(dx.shape, self.X.shape)
        
        logger.info("✓ Tanh backward test passed")


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Starting Activation tests")
    logger.info("=" * 60)
    
    unittest.main(verbosity=2)
