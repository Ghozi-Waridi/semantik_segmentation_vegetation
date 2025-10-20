"""
Unit tests for model save and load utilities.
"""

import unittest
import numpy as np
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.utils.model_io import save_model, load_model, list_saved_models
from core.vgg16_cbam_segmentation import VGG16_CBAM_Segmentation
from core.utils.logging_config import setup_logging

logger = setup_logging(module_name='test_model_io', log_dir='logs/tests')


class TestModelSaveLoad(unittest.TestCase):
    """Test cases for model save and load functions"""
    
    def setUp(self):
        """Setup for each test"""
        self.input_shape = (128, 128, 3)
        self.num_classes = 6
        self.test_dir = tempfile.mkdtemp(prefix='test_model_')
        
        logger.info(f"Test setup: Creating model with input_shape={self.input_shape}, "
                   f"num_classes={self.num_classes}")
        
        # Create a simple model for testing
        self.model = VGG16_CBAM_Segmentation(
            input_shape=self.input_shape,
            num_classes=self.num_classes
        )
        
        # Forward pass to initialize weights
        X_test = np.random.randn(1, *self.input_shape)
        _ = self.model.forward(X_test)
        
        logger.info(f"Test directory: {self.test_dir}")
    
    def tearDown(self):
        """Cleanup after each test"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            logger.info(f"Cleaned up test directory")
    
    def test_save_model(self):
        """Test model saving"""
        logger.info("Testing model save")
        
        save_path = save_model(self.model, save_dir=self.test_dir, model_name='test_model')
        
        # Check directory structure
        self.assertTrue(os.path.exists(save_path))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'metadata.json')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'architecture.json')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'weights')))
        
        logger.info(f"✓ Model save test passed - Path: {save_path}")
    
    def test_save_creates_weights(self):
        """Test that save creates weight files"""
        logger.info("Testing weight files creation")
        
        save_path = save_model(self.model, save_dir=self.test_dir, model_name='test_model')
        weights_dir = os.path.join(save_path, 'weights')
        
        # Check that some weight files were created
        weight_files = [f for f in os.listdir(weights_dir) if f.endswith('.npy')]
        self.assertGreater(len(weight_files), 0)
        
        logger.info(f"✓ Weight files created test passed - Files: {len(weight_files)}")
    
    def test_load_model(self):
        """Test model loading"""
        logger.info("Testing model load")
        
        # Save model
        save_path = save_model(self.model, save_dir=self.test_dir, model_name='test_model')
        
        # Create new model
        new_model = VGG16_CBAM_Segmentation(
            input_shape=self.input_shape,
            num_classes=self.num_classes
        )
        
        # Load weights
        loaded_model = load_model(new_model, save_path)
        
        # Check that weights were loaded
        self.assertIsNotNone(loaded_model)
        
        logger.info("✓ Model load test passed")
    
    def test_weights_match_after_load(self):
        """Test that loaded weights match saved weights"""
        logger.info("Testing weights match after load")
        
        # Get original Conv2D layer weights
        original_weights = None
        for layer in self.model.layers:
            if hasattr(layer, 'weights') and layer.weights is not None:
                original_weights = layer.weights.copy()
                break
        
        self.assertIsNotNone(original_weights)
        
        # Save and load
        save_path = save_model(self.model, save_dir=self.test_dir, model_name='test_model')
        
        new_model = VGG16_CBAM_Segmentation(
            input_shape=self.input_shape,
            num_classes=self.num_classes
        )
        loaded_model = load_model(new_model, save_path)
        
        # Get loaded weights
        loaded_weights = None
        for layer in loaded_model.layers:
            if hasattr(layer, 'weights') and layer.weights is not None:
                loaded_weights = layer.weights
                break
        
        self.assertIsNotNone(loaded_weights)
        
        # Compare
        np.testing.assert_array_almost_equal(original_weights, loaded_weights)
        
        logger.info("✓ Weights match test passed")
    
    def test_list_saved_models(self):
        """Test listing saved models"""
        logger.info("Testing list saved models")
        
        # Save a model
        save_model(self.model, save_dir=self.test_dir, model_name='test_model')
        
        # List models
        models = list_saved_models(self.test_dir)
        
        self.assertGreater(len(models), 0)
        
        logger.info(f"✓ List saved models test passed - Found {len(models)} model(s)")
    
    def test_save_with_different_names(self):
        """Test saving with different model names"""
        logger.info("Testing save with different names")
        
        path1 = save_model(self.model, save_dir=self.test_dir, model_name='model_v1')
        path2 = save_model(self.model, save_dir=self.test_dir, model_name='model_v2')
        
        self.assertNotEqual(path1, path2)
        self.assertTrue(os.path.exists(path1))
        self.assertTrue(os.path.exists(path2))
        
        logger.info("✓ Different names test passed")
    
    def test_metadata_content(self):
        """Test saved metadata content"""
        logger.info("Testing metadata content")
        
        import json
        
        save_path = save_model(self.model, save_dir=self.test_dir, model_name='test_model')
        metadata_path = os.path.join(save_path, 'metadata.json')
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.assertEqual(metadata['input_shape'], self.input_shape)
        self.assertEqual(metadata['num_classes'], self.num_classes)
        self.assertIn('timestamp', metadata)
        
        logger.info("✓ Metadata content test passed")


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Starting Model IO tests")
    logger.info("=" * 60)
    
    unittest.main(verbosity=2)
