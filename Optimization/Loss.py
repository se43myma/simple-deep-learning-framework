import numpy as np

class CrossEntropyLoss:
    def __init__(self) -> None:
        pass

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        desired_prediction = np.where(label_tensor == 1, self.prediction_tensor + np.finfo(float).eps, label_tensor)
        self.loss = np.sum(-np.log(desired_prediction[desired_prediction != 0.0]))
        return self.loss

    def backward(self, lable_tensor):
        pass