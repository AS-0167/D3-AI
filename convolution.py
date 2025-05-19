import numpy as np

class Conv2D:
    def __init__(self, filters, kernel_size, in_channels=3, stride=1, padding=0):
        self.filters = filters
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding

        # Initialize weights with proper dimensions
        self.weights = np.random.randn(kernel_size, kernel_size, in_channels, filters) * np.sqrt(2. / (kernel_size * kernel_size * in_channels))
        self.bias = np.zeros(filters)

    def forward(self, X):
        self.X = X
        batch_size, in_height, in_width, in_channels = X.shape

        # Calculate output dimensions
        out_height = (in_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (in_width - self.kernel_size + 2 * self.padding) // self.stride + 1

        # Add padding if needed
        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        else:
            X_padded = X

        # Initialize output
        output = np.zeros((batch_size, out_height, out_width, self.filters))

        # Perform convolution
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                X_slice = X_padded[:, h_start:h_end, w_start:w_end, :]

                # Vectorized implementation
                for f in range(self.filters):
                    output[:, i, j, f] = np.sum(
                        X_slice * self.weights[:, :, :, f],
                        axis=(1, 2, 3)
                    ) + self.bias[f]

        return output

    def backward(self, d_out, learning_rate):
        batch_size, out_height, out_width, _ = d_out.shape
        _, in_height, in_width, in_channels = self.X.shape

        # Initialize gradients
        d_weights = np.zeros_like(self.weights)
        d_bias = np.zeros_like(self.bias)
        d_X = np.zeros_like(self.X)

        # Add padding if needed
        if self.padding > 0:
            X_padded = np.pad(self.X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
            d_X_padded = np.pad(d_X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        else:
            X_padded = self.X
            d_X_padded = d_X

        # Calculate gradients
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                X_slice = X_padded[:, h_start:h_end, w_start:w_end, :]

                for f in range(self.filters):
                    # Gradient for weights
                    d_weights[:, :, :, f] += np.sum(
                        X_slice * d_out[:, i:i+1, j:j+1, f:f+1],
                        axis=0
                    )
                    # Gradient for input
                    d_X_padded[:, h_start:h_end, w_start:w_end, :] += \
                        self.weights[:, :, :, f] * d_out[:, i:i+1, j:j+1, f:f+1]

                # Gradient for bias (moved outside f loop for efficiency)
                d_bias += np.sum(d_out[:, i, j, :], axis=0)

        # Remove padding from gradient if needed
        if self.padding > 0:
            d_X = d_X_padded[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            d_X = d_X_padded

        # Update weights
        self.weights -= learning_rate * d_weights / batch_size
        self.bias -= learning_rate * d_bias / batch_size

        return d_X