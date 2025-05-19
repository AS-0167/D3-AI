import numpy as np

class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, X):
        self.X = X
        batch_size, in_height, in_width, channels = X.shape

        out_height = (in_height - self.pool_size) // self.stride + 1
        out_width = (in_width - self.pool_size) // self.stride + 1

        output = np.zeros((batch_size, out_height, out_width, channels))
        self.max_indices = np.zeros((batch_size, out_height, out_width, channels, 2), dtype=int)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                X_slice = X[:, h_start:h_end, w_start:w_end, :]
                output[:, i, j, :] = np.max(X_slice, axis=(1, 2))

                for b in range(batch_size):
                    for c in range(channels):
                        max_idx = np.unravel_index(np.argmax(X_slice[b, :, :, c]), (self.pool_size, self.pool_size))
                        self.max_indices[b, i, j, c] = [h_start + max_idx[0], w_start + max_idx[1]]

        return output

    def backward(self, d_out):
        batch_size, out_height, out_width, channels = d_out.shape
        d_X = np.zeros_like(self.X)

        for i in range(out_height):
            for j in range(out_width):
                for b in range(batch_size):
                    for c in range(channels):
                        h, w = self.max_indices[b, i, j, c]
                        d_X[b, h, w, c] += d_out[b, i, j, c]

        return d_X