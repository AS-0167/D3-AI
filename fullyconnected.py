import numpy as np
class Flatten:
    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, d_out):
        return d_out.reshape(self.input_shape)

class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros(output_size)

    def forward(self, X):
        self.X = X
        return np.dot(X, self.weights) + self.bias

    def backward(self, d_out, learning_rate):
        d_X = np.dot(d_out, self.weights.T)
        d_weights = np.dot(self.X.T, d_out)
        d_bias = np.sum(d_out, axis=0)

        self.weights -= learning_rate * d_weights / self.X.shape[0]
        self.bias -= learning_rate * d_bias / self.X.shape[0]

        return d_X