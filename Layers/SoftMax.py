import numpy as np
from Layers.Base import *

class SoftMax(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        input_tensor = input_tensor - np.max(input_tensor)
        sum_vector = np.sum(np.exp(input_tensor), 1).reshape(len(input_tensor),1)
        self.output_tensor = np.exp(input_tensor) / sum_vector
        return self.output_tensor

    def backward(self, error_tensor):
        self.error_tensor = self.output_tensor * (error_tensor - np.sum(error_tensor * self.output_tensor, 1).reshape(len(error_tensor),1))
        return self.error_tensor