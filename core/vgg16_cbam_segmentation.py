import numpy as np
from core.utils.gpu_utils import get_array_module, xp, to_gpu, to_cpu

from core.activation.activation import Softmax, ReLU, Sigmoid
from core.evaluation.evaluation import categorical_crossentropy_backward
from core.layers.convolutional import Conv2D
from core.layers.pooling import MaxPooling2D
from core.layers.attention import CBAM
from core.layers.upsampling import UpSampling2D




class VGG16_CBAM_Segmentation:
    def __init__(self, input_shape: tuple, num_classes: int) -> None:
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.layers = []
        self.skip_connections = {}

        self.build_encoder()
        self.build_decoder()

        self.softmax = Softmax()

    def build_encoder(self) -> None:
        
        self.layers.append(Conv2D(64, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(64, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(MaxPooling2D((2, 2), stride=(2,2)))
        
        
        self.layers.append(Conv2D(128, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(128, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(MaxPooling2D((2, 2), stride=(2,2)))
        
        
        self.layers.append(Conv2D(256, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(256, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(256, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(MaxPooling2D((2, 2), stride=(2,2)))
        
        
        self.layers.append(Conv2D(512, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(512, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(512, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(MaxPooling2D((2, 2), stride=(2,2)))

        
        self.layers.append(Conv2D(512, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(512, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(512, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(MaxPooling2D((2, 2), stride=(2,2)))
        
        
        self.layers.append(CBAM(channel=512))
    def build_decoder(self) -> None:
        
        self.layers.append(UpSampling2D((2,2)))
        self.layers.append(Conv2D(512, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(512, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(512, (3, 3), padding="same"))
        self.layers.append(ReLU())
        
        
        self.layers.append(UpSampling2D((2,2)))
        self.layers.append(Conv2D(512, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(256, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(256, (3, 3), padding="same"))
        self.layers.append(ReLU())
        
        
        self.layers.append(UpSampling2D((2,2)))
        self.layers.append(Conv2D(256, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(128, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(128, (3, 3), padding="same"))
        self.layers.append(ReLU())
        
        
        self.layers.append(UpSampling2D((2,2)))
        self.layers.append(Conv2D(64, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(64, (3, 3), padding="same"))
        self.layers.append(ReLU())
        
        
        self.layers.append(UpSampling2D((2,2)))
        self.layers.append(Conv2D(64, (3, 3), padding="same"))
        self.layers.append(ReLU())
       
        
        self.layers.append(Conv2D(self.num_classes, (1, 1), padding="same"))     
                   
    def forward(self, X):
        
        activations = [X]
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, MaxPooling2D):
                self.skip_connections[f'skip_{len(self.skip_connections)}'] = activations[-1]
            
            output = layer(activations[-1])
            activations.append(output)
            
        
        output = self.softmax.forward(activations[-1])
        activations.append(output)
            
        return activations[-1]
    
    def backward(self, y_true, learning_rate=0.001):
        
        dout = categorical_crossentropy_backward(y_true, self.softmax.output)
        
        
        dout = self.softmax.backward(dout)
        
        
        for layer in reversed(self.layers):
            dout = layer.backward(dout, learning_rate)
        
        return dout