
import numpy as np
from Layers.Base import BaseLayer
from Optimization.Optimizers import Sgd

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True                                                       
        self.weights = np.random.uniform(0.0, 1.0, (input_size+1,output_size))      
        self._optimizer = None                                                      #Instance of optimizer(ex: SGD)

    #return input tensor for the next layer
    def forward(self, input_tensor):
        self.input_tensor = np.append(input_tensor, np.ones([len(input_tensor), 1]), 1)
        next_input_tensor = self.input_tensor @ self.weights
        return next_input_tensor

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    #return error tensor for the previous layer and perform weight update as per sgd
    def backward(self, error_tensor):
        error_tensor_prev = error_tensor @ np.transpose(self.weights[0:self.input_size,:])
        self.gradient_weights = np.transpose(self.input_tensor) @ error_tensor
        if(isinstance(self._optimizer, Sgd)):
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        return error_tensor_prev

    @property
    def gradients_weights(self):
        return self.gradients_weights



