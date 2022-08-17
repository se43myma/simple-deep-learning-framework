class NeuralNetwork:
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer      #Optimizer selected for the problem e.g Stochastic descent, Stochastic + momentum, etc.
        self.loss = list()              #List containing loss in each layer
        self.layers = list()            #Architecture of neural network. Contains modules from folder Layers stored in a list. eg: [FullyConnected, ReLU, FullyConnected, Softmax, etc.]
        self.data_layer = list()        #contains list of training data, i.e input data and the corresponding lable data, stored in a list.
        self.loss_layer = None          #function to compute loss of the output layer. e.g: L2-loss, Cross-entropy, etc.

    #iterate along the network and returns loss, computed from last layer
    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        input_tensor = self.input_tensor
        for layer in self.layers:
            self.output_tensor = layer.forward(input_tensor)
            input_tensor = self.output_tensor
        self.output_loss = self.loss_layer.forward(self.output_tensor, self.label_tensor)
        return self.output_loss

    #
    def backward(self):
        #error_tensor  
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
