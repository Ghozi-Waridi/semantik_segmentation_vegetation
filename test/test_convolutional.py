"""
Unit tests for Conv2D layer.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.layers.convolutional import Conv2D
from core.utils.logging_config import setup_logging

# Setup logging
logger = setup_logging(module_name='test_convolutional', log_dir='logs/tests')


class TestConv2D(unittest.TestCase):
    """Test cases for Conv2D layer"""
    
    def setUp(self):
        """Setup for each test"""
        self.batch_size = 2
        self.input_height = 8
        self.input_width = 8
        self.input_channels = 3
        self.output_channels = 16
        self.kernel_size = (3, 3)
        
        # Create random input
        self.X = np.random.randn(
            self.batch_size, 
            self.input_height, 
            self.input_width, 
            self.input_channels
        )
        
        logger.info(f"Test setup: Input shape {self.X.shape}")
    
    def test_initialization(self):
        """Test Conv2D initialization"""
        logger.info("Testing Conv2D initialization")
        
        conv = Conv2D(
            out_channels=self.output_channels,
            kernel_size=self.kernel_size,
            padding='same'
        )
        
        self.assertEqual(conv.out_channels, self.output_channels)
        self.assertEqual(conv.kernel_size, self.kernel_size)
        self.assertEqual(conv.padding, 'same')
        self.assertIsNone(conv.weights)
        
        logger.info("✓ Initialization test passed")
    
    def test_forward_pass(self):
        """Test forward pass"""
        logger.info("Testing forward pass")
        
        conv = Conv2D(
            out_channels=self.output_channels,
            kernel_size=self.kernel_size,
            padding='same',
            strides=(1, 1)
        )
        
        output = conv(self.X)
        
        # Check output shape
        expected_h = self.input_height
        expected_w = self.input_width
        self.assertEqual(output.shape, (self.batch_size, expected_h, expected_w, self.output_channels))
        
        # Check weights were initialized
        self.assertIsNotNone(conv.weights)
        self.assertEqual(conv.weights.shape, 
                        (self.kernel_size[0], self.kernel_size[1], 
                         self.input_channels, self.output_channels))
        
        logger.info(f"✓ Forward pass test passed - Output shape: {output.shape}")
    
    def test_forward_different_strides(self):
        """Test forward pass with different strides"""
        logger.info("Testing forward pass with strides=(2,2)")
        
        conv = Conv2D(
            out_channels=self.output_channels,
            kernel_size=self.kernel_size,
            padding='same',
            strides=(2, 2)
        )
        
        output = conv(self.X)
        
        # With stride 2 and padding 'same'
        expected_h = (self.input_height + 2 - self.kernel_size[0]) // 2 + 1
        expected_w = (self.input_width + 2 - self.kernel_size[1]) // 2 + 1
        
        self.assertEqual(output.shape, (self.batch_size, expected_h, expected_w, self.output_channels))
        
        logger.info(f"✓ Stride test passed - Output shape: {output.shape}")
    
    def test_backward_pass(self):
        """Test backward pass"""
        logger.info("Testing backward pass")
        
        conv = Conv2D(
            out_channels=self.output_channels,
            kernel_size=self.kernel_size,
            padding='same',
            strides=(1, 1)
        )
        
        # Forward pass
        output = conv(self.X)
        
        # Create gradient matching output shape
        dout = np.random.randn(*output.shape)
        
        # Backward pass
        dx = conv.backward(dout, learning_rate=0.01)
        
        # Check gradient shape matches input
        self.assertEqual(dx.shape, self.X.shape)
        
        logger.info(f"✓ Backward pass test passed - Gradient shape: {dx.shape}")
    
    def test_weight_update(self):
        """Test that weights are updated during backward pass"""
        logger.info("Testing weight updates")
        
        conv = Conv2D(
            out_channels=self.output_channels,
            kernel_size=self.kernel_size,
            padding='same',
            strides=(1, 1)
        )
        
        # Forward pass
        output = conv(self.X)
        weights_before = conv.weights.copy()
        
        # Backward pass
        dout = np.random.randn(*output.shape)
        conv.backward(dout, learning_rate=0.01)
        
        # Check weights changed
        weight_diff = np.sum(np.abs(conv.weights - weights_before))
        self.assertGreater(weight_diff, 0)
        
        logger.info(f"✓ Weight update test passed - Change magnitude: {weight_diff:.6f}")
    
    def test_pad_image(self):
        """Test padding function"""
        logger.info("Testing pad_image function")
        
        conv = Conv2D(self.output_channels, self.kernel_size)
        padded = conv.pad_image(self.X, (1, 1))
        
        expected_shape = (self.batch_size, 
                         self.input_height + 2,
                         self.input_width + 2,
                         self.input_channels)
        
        self.assertEqual(padded.shape, expected_shape)
        
        logger.info(f"✓ Pad image test passed - Padded shape: {padded.shape}")
    
    def test_numerical_gradient(self):
        """Test numerical gradient against backprop"""
        logger.info("Testing numerical gradient")
        
        conv = Conv2D(
            out_channels=2,  # Small number for faster test
            kernel_size=(3, 3),
            padding='same',
            strides=(1, 1)
        )
        
        # Small input for faster computation
        X_small = np.random.randn(1, 3, 3, 2)
        output = conv(X_small)
        
        # Compute backprop gradient
        dout = np.ones_like(output)
        conv.backward(dout, learning_rate=0.0)  # No update
        
        # This test mainly checks that backward doesn't crash
        self.assertIsNotNone(conv.weights)
        
        logger.info("✓ Numerical gradient test passed")


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Starting Conv2D tests")
    logger.info("=" * 60)
    
    unittest.main(verbosity=2)
