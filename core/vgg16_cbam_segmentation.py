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
        ## Block 1
        self.layers.append(Conv2D(64, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(64, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(MaxPooling2D((2, 2), stride=(2,2)))
        
        ## Block 2
        self.layers.append(Conv2D(128, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(128, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(MaxPooling2D((2, 2), stride=(2,2)))
        
        ## Block 3
        self.layers.append(Conv2D(256, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(256, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(256, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(MaxPooling2D((2, 2), stride=(2,2)))
        
        ## Block 4
        self.layers.append(Conv2D(512, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(512, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(512, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(MaxPooling2D((2, 2), stride=(2,2)))

        ## Block 5
        self.layers.append(Conv2D(512, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(512, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(512, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(MaxPooling2D((2, 2), stride=(2,2)))
        
        ## CBAM
        self.layers.append(CBAM(channel=512))
    def build_decoder(self) -> None:
        ## Block 1
        self.layers.append(UpSampling2D((2,2)))
        self.layers.append(Conv2D(512, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(512, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(512, (3, 3), padding="same"))
        self.layers.append(ReLU())
        
        ## Block 2
        self.layers.append(UpSampling2D((2,2)))
        self.layers.append(Conv2D(512, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(256, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(256, (3, 3), padding="same"))
        self.layers.append(ReLU())
        
        ## Block 3
        self.layers.append(UpSampling2D((2,2)))
        self.layers.append(Conv2D(256, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(128, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(128, (3, 3), padding="same"))
        self.layers.append(ReLU())
        
        ## Block 4
        self.layers.append(UpSampling2D((2,2)))
        self.layers.append(Conv2D(64, (3, 3), padding="same"))
        self.layers.append(ReLU())
        self.layers.append(Conv2D(64, (3, 3), padding="same"))
        self.layers.append(ReLU())
        
        ## Block 5 - Final upsampling to match input size
        self.layers.append(UpSampling2D((2,2)))
        self.layers.append(Conv2D(64, (3, 3), padding="same"))
        self.layers.append(ReLU())
       
        ## Final Convolution
        self.layers.append(Conv2D(self.num_classes, (1, 1), padding="same"))     
                   
    def forward(self, X):
        # Ensure data is on the appropriate device
        activations = [X]
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, MaxPooling2D):
                self.skip_connections[f'skip_{len(self.skip_connections)}'] = activations[-1]
            
            output = layer(activations[-1])
            activations.append(output)
            
        # Apply softmax for segmentation
        output = self.softmax.forward(activations[-1])
        activations.append(output)
            
        return activations[-1]
    
    def backward(self, y_true, learning_rate=0.001):
        # Start with gradient from loss function
        dout = categorical_crossentropy_backward(y_true, self.softmax.output)
        
        # Backward through softmax
        dout = self.softmax.backward(dout)
        
        # Backward through all layers in reverse order
        for layer in reversed(self.layers):
            dout = layer.backward(dout, learning_rate)
        
        return dout