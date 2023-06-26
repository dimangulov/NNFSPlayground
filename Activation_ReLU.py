import numpy as np

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.output <= 0] = 0