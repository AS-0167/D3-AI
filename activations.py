import numpy as np

class ReLU:
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, d_out):
        return d_out * (self.X > 0)

class Softmax:
    def forward(self, X):
        exp = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output

    def backward(self, y_true):
        return self.output - y_true