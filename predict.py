"""
Script untuk load trained model dan perform inference/prediction.
Digunakan untuk testing dan deployment.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.vgg16_cbam_segmentation import VGG16_CBAM_Segmentation
from core.utils.model_io import load_model, list_saved_models
from core.utils.logging_config import setup_logging
from core.utils.gpu_utils import to_gpu, to_cpu, print_gpu_info

logger = setup_logging(module_name='predict', log_dir='logs')


def load_and_predict(model_dir: str, input_image: np.ndarray):
    """
    Load model and perform prediction on input image.
    
    Args:
        model_dir: Path to saved model directory
        input_image: Input image array with shape (H, W, 3) or (B, H, W, 3)
        
    Returns:
        prediction: Segmentation map with predicted classes
    """
    try:
        logger.info(f"Loading model from {model_dir}")
        
        # Create model instance
        model = VGG16_CBAM_Segmentation(
            input_shape=(128, 128, 3),
            num_classes=6
        )
        
        # Load weights
        model = load_model(model, model_dir)
        logger.info("Model loaded successfully")
        
        # Prepare input
        if len(input_image.shape) == 3:
            input_image = np.expand_dims(input_image, axis=0)
        
        logger.info(f"Input shape: {input_image.shape}")
        
        # Move to GPU (if available) and forward pass
        input_gpu = to_gpu(input_image)
        output = model.forward(input_gpu)
        logger.info(f"Output shape: {output.shape}")
        
        # Get class predictions
        prediction = np.argmax(to_cpu(output), axis=-1)
        logger.info(f"Prediction shape: {prediction.shape}")
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise


def main():
    """
    Main function demonstrating model usage.
    """
    logger.info("="*60)
    logger.info("MODEL PREDICTION SCRIPT")
    logger.info("="*60)
    print_gpu_info()
    
    # List available saved models
    logger.info("\nListing available models:")
    models = list_saved_models(save_dir='models')
    
    if not models:
        logger.warning("No saved models found in 'models' directory")
        logger.info("Please run main.py to train and save a model first")
        return
    
    print("\nAvailable models:")
    for i, model_name in enumerate(models, 1):
        print(f"  {i}. {model_name}")
    
    # Use latest model
    latest_model = models[0]
    model_path = os.path.join('models', latest_model)
    
    logger.info(f"\nUsing model: {latest_model}")
    
    # Create sample input image
    logger.info("Creating sample input image...")
    input_image = np.random.randn(128, 128, 3)
    
    # Normalize to [0, 1]
    input_image = (input_image - np.min(input_image)) / (np.max(input_image) - np.min(input_image))
    
    # Predict
    prediction = load_and_predict(model_path, input_image)
    
    print(f"\nPrediction shape: {prediction.shape}")
    print(f"Unique classes: {np.unique(prediction)}")
    
    logger.info("\nPrediction completed successfully!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
