import numpy as np
from Layers.Base import *

class ReLU(BaseLayer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = self.input_tensor * (self.input_tensor > 0)
        return self.output_tensor

    #error_previous = 0 if x<=0 or = error
    def backward(self, error_tensor):
        self.error_tensor = error_tensor * (self.output_tensor > 0)
        return self.error_tensor
