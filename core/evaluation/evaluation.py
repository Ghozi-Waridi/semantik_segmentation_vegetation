from core.utils.gpu_utils import get_array_module, xp
import numpy as np


def categorical_crossentropy(y_true, y_pred):
    """Categorical crossentropy loss with GPU support"""
    xp_module = get_array_module(y_pred)
   
    y_pred = xp_module.clip(y_pred, 1e-7, 1 - 1e-7)
    return float(xp_module.mean(-xp_module.sum(y_true * xp_module.log(y_pred), axis=-1)))


def categorical_crossentropy_backward(y_true, y_pred):
    """Gradient of cross entropy with softmax - GPU compatible"""
   
    return y_pred - y_true
