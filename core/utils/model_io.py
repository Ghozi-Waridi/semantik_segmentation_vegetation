"""
Model save and load utilities for VGG16 CBAM segmentation model.
Handles serialization of weights and biases from all layers.
"""

import os
import json
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from core.utils.gpu_utils import to_cpu, to_gpu, is_gpu_available



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def save_model(model, save_dir: str = 'models', model_name: str = 'vgg16_cbam'):
    """
    Save model weights and architecture to disk.
    
    Args:
        model: VGG16_CBAM_Segmentation model instance
        save_dir: Directory to save model files
        model_name: Name prefix for saved files
        
    Returns:
        str: Path to saved model directory
    """
    try:
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(save_dir, f"{model_name}_{timestamp}")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting model save to {model_dir}")
        
        
        metadata = {
            'input_shape': model.input_shape,
            'num_classes': model.num_classes,
            'num_layers': len(model.layers),
            'timestamp': timestamp
        }
        
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
        
        
        weights_dir = os.path.join(model_dir, 'weights')
        Path(weights_dir).mkdir(parents=True, exist_ok=True)
        
        layer_info = []
        for idx, layer in enumerate(model.layers):
            layer_name = layer.__class__.__name__
            layer_data = {'index': idx, 'type': layer_name}
            
            
            if hasattr(layer, 'weights') and layer.weights is not None:
                weights_file = os.path.join(weights_dir, f'layer_{idx:03d}_weights.npy')
                
                np.save(weights_file, to_cpu(layer.weights))
                layer_data['weights'] = f'layer_{idx:03d}_weights.npy'
                logger.debug(f"Saved weights for layer {idx} ({layer_name}): {layer.weights.shape}")
            
            
            if hasattr(layer, 'bias') and layer.bias is not None:
                bias_file = os.path.join(weights_dir, f'layer_{idx:03d}_bias.npy')
                
                np.save(bias_file, to_cpu(layer.bias))
                layer_data['bias'] = f'layer_{idx:03d}_bias.npy'
                logger.debug(f"Saved bias for layer {idx} ({layer_name}): {layer.bias.shape}")
            
            
            if hasattr(layer, 'channel_attention'):
                ca_weights1 = os.path.join(weights_dir, f'layer_{idx:03d}_ca_weights1.npy')
                np.save(ca_weights1, to_cpu(layer.channel_attention.mlp_weights1))
                layer_data['ca_weights1'] = f'layer_{idx:03d}_ca_weights1.npy'
                
                ca_weights2 = os.path.join(weights_dir, f'layer_{idx:03d}_ca_weights2.npy')
                np.save(ca_weights2, to_cpu(layer.channel_attention.mlp_weights2))
                layer_data['ca_weights2'] = f'layer_{idx:03d}_ca_weights2.npy'
                logger.debug(f"Saved channel attention weights for layer {idx}")
            
            if hasattr(layer, 'spatial_attention'):
                sa_weights = os.path.join(weights_dir, f'layer_{idx:03d}_sa_weights.npy')
                np.save(sa_weights, to_cpu(layer.spatial_attention.conv_weights))
                layer_data['sa_weights'] = f'layer_{idx:03d}_sa_weights.npy'
                logger.debug(f"Saved spatial attention weights for layer {idx}")
            
            layer_info.append(layer_data)
        
        
        arch_path = os.path.join(model_dir, 'architecture.json')
        with open(arch_path, 'w') as f:
            json.dump(layer_info, f, indent=2)
        logger.info(f"Saved architecture info to {arch_path}")
        
        logger.info(f"Model successfully saved to {model_dir}")
        return model_dir
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}", exc_info=True)
        raise


def load_model(model, model_dir: str):
    """
    Load model weights from saved directory.
    
    Args:
        model: VGG16_CBAM_Segmentation model instance to load weights into
        model_dir: Directory containing saved model files
        
    Returns:
        model: Updated model with loaded weights
    """
    try:
        logger.info(f"Starting model load from {model_dir}")
        
        
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loading model from {metadata['timestamp']}")
        logger.info(f"Model architecture: input_shape={metadata['input_shape']}, "
                   f"num_classes={metadata['num_classes']}")
        
        if metadata['num_layers'] != len(model.layers):
            logger.warning(f"Layer count mismatch: saved={metadata['num_layers']}, "
                          f"current={len(model.layers)}")
        
        
        arch_path = os.path.join(model_dir, 'architecture.json')
        with open(arch_path, 'r') as f:
            layer_info = json.load(f)
        
        weights_dir = os.path.join(model_dir, 'weights')
        
        
        for layer_data in layer_info:
            idx = layer_data['index']
            layer_type = layer_data['type']
            
            if idx >= len(model.layers):
                logger.warning(f"Skipping layer {idx}: index out of bounds")
                continue
            
            layer = model.layers[idx]
            
            
            if 'weights' in layer_data and hasattr(layer, 'weights'):
                weights_file = os.path.join(weights_dir, layer_data['weights'])
                w = np.load(weights_file)
                
                layer.weights = to_gpu(w) if is_gpu_available() else w
                logger.debug(f"Loaded weights for layer {idx} ({layer_type}): {layer.weights.shape}")
            
            
            if 'bias' in layer_data and hasattr(layer, 'bias'):
                bias_file = os.path.join(weights_dir, layer_data['bias'])
                b = np.load(bias_file)
                layer.bias = to_gpu(b) if is_gpu_available() else b
                logger.debug(f"Loaded bias for layer {idx} ({layer_type}): {layer.bias.shape}")
            
            
            if 'ca_weights1' in layer_data and hasattr(layer, 'channel_attention'):
                w1_file = os.path.join(weights_dir, layer_data['ca_weights1'])
                w1 = np.load(w1_file)
                layer.channel_attention.mlp_weights1 = to_gpu(w1) if is_gpu_available() else w1
                
                w2_file = os.path.join(weights_dir, layer_data['ca_weights2'])
                w2 = np.load(w2_file)
                layer.channel_attention.mlp_weights2 = to_gpu(w2) if is_gpu_available() else w2
                logger.debug(f"Loaded channel attention weights for layer {idx}")
            
            if 'sa_weights' in layer_data and hasattr(layer, 'spatial_attention'):
                sa_file = os.path.join(weights_dir, layer_data['sa_weights'])
                sw = np.load(sa_file)
                layer.spatial_attention.conv_weights = to_gpu(sw) if is_gpu_available() else sw
                logger.debug(f"Loaded spatial attention weights for layer {idx}")
        
        logger.info(f"Model successfully loaded from {model_dir}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        raise


def list_saved_models(save_dir: str = 'models'):
    """
    List all saved models in directory.
    
    Args:
        save_dir: Directory containing saved models
        
    Returns:
        list: List of saved model directories
    """
    try:
        if not os.path.exists(save_dir):
            logger.warning(f"Save directory {save_dir} does not exist")
            return []
        
        models = [d for d in os.listdir(save_dir) 
                 if os.path.isdir(os.path.join(save_dir, d))]
        logger.info(f"Found {len(models)} saved models in {save_dir}")
        return sorted(models, reverse=True)
        
    except Exception as e:
        logger.error(f"Error listing saved models: {str(e)}", exc_info=True)
        return []
