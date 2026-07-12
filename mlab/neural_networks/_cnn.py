import numpy as np


class ConvLayer:
    """2D Convolutional layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = tuple(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        k_h, k_w = self.kernel_size
        scale = np.sqrt(2.0 / (self.in_channels * k_h * k_w))

        self.weight = np.random.randn(
            self.out_channels,
            self.in_channels,
            k_h,
            k_w
        ) * scale
        self.bias = np.zeros(self.out_channels, dtype=np.float64)

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)

        self._input = None
        self._input_padded = None

    def __call__(self, X):
        """
        Forward pass.

        Args:
            X: array of shape (batch_size, height, width, in_channels)

        Returns:
            Array of shape (batch_size, out_height, out_width, out_channels)
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 4:
            raise ValueError("X must have shape (batch, height, width, channels)")
        if X.shape[-1] != self.in_channels:
            raise ValueError("Input channels do not match in_channels")

        self._input = X

        batch_size, height, width, _ = X.shape
        k_h, k_w = self.kernel_size
        stride = self.stride
        padding = self.padding

        out_height = (height + 2 * padding - k_h) // stride + 1
        out_width = (width + 2 * padding - k_w) // stride + 1

        if out_height <= 0 or out_width <= 0:
            raise ValueError("Kernel size, stride, or padding produces invalid output shape")

        self._input_padded = np.pad(
            X,
            ((0, 0), (padding, padding), (padding, padding), (0, 0)),
            mode="constant"
        )

        output = np.zeros(
            (batch_size, out_height, out_width, self.out_channels),
            dtype=np.float64
        )

        for sample in range(batch_size):
            for out_row in range(out_height):
                for out_col in range(out_width):
                    row_start = out_row * stride
                    col_start = out_col * stride

                    patch = self._input_padded[
                        sample,
                        row_start:row_start + k_h,
                        col_start:col_start + k_w,
                        :
                    ]

                    patch = np.transpose(patch, (2, 0, 1))

                    for channel in range(self.out_channels):
                        output[sample, out_row, out_col, channel] = (
                            np.sum(patch * self.weight[channel]) +
                            self.bias[channel]
                        )

        return output

    def backward(self, grad_output):
        """Compute gradients w.r.t. input, weights, and biases."""
        grad_output = np.asarray(grad_output, dtype=np.float64)

        batch_size, out_height, out_width, out_channels = grad_output.shape

        if out_channels != self.out_channels:
            raise ValueError("grad_output has incorrect channel count")

        k_h, k_w = self.kernel_size
        stride = self.stride
        padding = self.padding

        grad_input_padded = np.zeros_like(self._input_padded)
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)

        for sample in range(batch_size):
            for out_row in range(out_height):
                for out_col in range(out_width):
                    row_start = out_row * stride
                    col_start = out_col * stride

                    patch = self._input_padded[
                        sample,
                        row_start:row_start + k_h,
                        col_start:col_start + k_w,
                        :
                    ]
                    patch = np.transpose(patch, (2, 0, 1))

                    for channel in range(self.out_channels):
                        gradient = grad_output[sample, out_row, out_col, channel]

                        self.grad_weight[channel] += gradient * patch
                        self.grad_bias[channel] += gradient

                        grad_patch = gradient * self.weight[channel]
                        grad_patch = np.transpose(grad_patch, (1, 2, 0))

                        grad_input_padded[
                            sample,
                            row_start:row_start + k_h,
                            col_start:col_start + k_w,
                            :
                        ] += grad_patch

        self.grad_weight /= batch_size
        self.grad_bias /= batch_size

        if padding == 0:
            return grad_input_padded

        return grad_input_padded[
            :,
            padding:-padding,
            padding:-padding,
            :
        ]

    def update(self, learning_rate):
        """Update weights and biases."""
        self.weight -= learning_rate * self.grad_weight
        self.bias -= learning_rate * self.grad_bias


class PoolingLayer:
    """Max Pooling layer."""

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = int(pool_size)
        self.stride = int(stride)

        self._input = None
        self._max_indices = None

    def __call__(self, X):
        """
        Forward pass: apply max pooling.

        Args:
            X: array of shape (batch_size, height, width, channels)

        Returns:
            Array of shape (batch_size, out_height, out_width, channels)
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 4:
            raise ValueError("X must have shape (batch, height, width, channels)")

        self._input = X

        batch_size, height, width, channels = X.shape
        pool_size = self.pool_size
        stride = self.stride

        out_height = (height - pool_size) // stride + 1
        out_width = (width - pool_size) // stride + 1

        if out_height <= 0 or out_width <= 0:
            raise ValueError("pool_size or stride produces invalid output shape")

        output = np.zeros(
            (batch_size, out_height, out_width, channels),
            dtype=np.float64
        )

        self._max_indices = np.zeros(
            (batch_size, out_height, out_width, channels, 2),
            dtype=np.int64
        )

        for sample in range(batch_size):
            for out_row in range(out_height):
                for out_col in range(out_width):
                    row_start = out_row * stride
                    col_start = out_col * stride

                    region = X[
                        sample,
                        row_start:row_start + pool_size,
                        col_start:col_start + pool_size,
                        :
                    ]

                    for channel in range(channels):
                        flat_index = np.argmax(region[:, :, channel])
                        row_offset, col_offset = np.unravel_index(
                            flat_index,
                            (pool_size, pool_size)
                        )

                        output[sample, out_row, out_col, channel] = (
                            region[row_offset, col_offset, channel]
                        )

                        self._max_indices[sample, out_row, out_col, channel] = (
                            row_start + row_offset,
                            col_start + col_offset
                        )

        return output

    def backward(self, grad_output):
        """Route gradients back through max positions."""
        grad_output = np.asarray(grad_output, dtype=np.float64)
        grad_input = np.zeros_like(self._input)

        batch_size, out_height, out_width, channels = grad_output.shape

        for sample in range(batch_size):
            for out_row in range(out_height):
                for out_col in range(out_width):
                    for channel in range(channels):
                        row_index, col_index = self._max_indices[
                            sample,
                            out_row,
                            out_col,
                            channel
                        ]

                        grad_input[
                            sample,
                            row_index,
                            col_index,
                            channel
                        ] += grad_output[sample, out_row, out_col, channel]

        return grad_input


class ReLULayer:
    """ReLU activation layer."""

    def __init__(self):
        self._input = None

    def __call__(self, X):
        X = np.asarray(X, dtype=np.float64)
        self._input = X
        return np.maximum(0.0, X)

    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=np.float64)
        return grad_output * (self._input > 0.0)


class SoftmaxLayer:
    """Softmax activation layer."""

    def __init__(self):
        self._output = None

    def __call__(self, X):
        X = np.asarray(X, dtype=np.float64)

        shifted = X - np.max(X, axis=1, keepdims=True)
        exp_values = np.exp(np.clip(shifted, -50.0, 50.0))

        self._output = exp_values / np.sum(
            exp_values,
            axis=1,
            keepdims=True
        )

        return self._output

    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=np.float64)

        return self._output * (
            grad_output -
            np.sum(grad_output * self._output, axis=1, keepdims=True)
        )


class ModularLinearLayer:
    """Fully connected layer."""

    def __init__(self, input_size, output_size):
        self.input_size = int(input_size)
        self.output_size = int(output_size)

        scale = np.sqrt(2.0 / max(1, self.input_size))

        self.weight = np.random.randn(
            self.input_size,
            self.output_size
        ) * scale

        self.bias = np.zeros(self.output_size, dtype=np.float64)

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)

        self._input = None

    def __call__(self, X):
        X = np.asarray(X, dtype=np.float64)
        self._input = X
        return X @ self.weight + self.bias

    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=np.float64)
        batch_size = max(1, self._input.shape[0])

        self.grad_weight = (self._input.T @ grad_output) / batch_size
        self.grad_bias = np.sum(grad_output, axis=0) / batch_size

        return grad_output @ self.weight.T

    def update(self, learning_rate):
        self.weight -= learning_rate * self.grad_weight
        self.bias -= learning_rate * self.grad_bias


class CNNClassifier:
    """Simple CNN classifier: Conv -> ReLU -> MaxPool -> Linear -> Softmax."""

    def __init__(
        self,
        input_shape=(28, 28, 1),
        num_classes=10,
        lr=0.01,
        epochs=20,
        batch_size=32,
        random_state=None,
        learning_rate=None,
        n_iterations=None,
        max_iter=None
    ):
        if learning_rate is not None:
            lr = learning_rate
        if n_iterations is not None:
            epochs = n_iterations
        if max_iter is not None:
            epochs = max_iter

        self.input_shape = tuple(input_shape)
        self.num_classes = int(num_classes)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.random_state = random_state

        self.layers_ = []
        self.layers = self.layers_
        self._layers = self.layers_
        self.network = self.layers_
        self.model = self.layers_

        self.loss_curve_ = []
        self._rng = None
        self._pooled_shape = None

    def _validate_params(self):
        if len(self.input_shape) != 3:
            raise ValueError("input_shape must be (height, width, channels)")
        if any(value <= 0 for value in self.input_shape):
            raise ValueError("input_shape values must be positive")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def _validate_X(self, X):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 4:
            raise ValueError("X must be 4D: (samples, height, width, channels)")
        if X.shape[1:] != self.input_shape:
            raise ValueError("X image dimensions must match input_shape")
        if X.shape[0] == 0:
            raise ValueError("X must not be empty")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")

        return X

    def _validate_X_y(self, X, y):
        X = self._validate_X(X)
        y = np.asarray(y).ravel()

        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if y.size == 0:
            raise ValueError("y must not be empty")
        if np.any(y < 0) or np.any(y >= self.num_classes):
            raise ValueError("y labels must be between 0 and num_classes - 1")

        return X, y.astype(int)

    def _build_network(self):
        height, width, channels = self.input_shape

        self.conv = ConvLayer(
            in_channels=channels,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.relu = ReLULayer()
        self.pool = PoolingLayer(pool_size=2, stride=2)

        pooled_height = (height - 2) // 2 + 1
        pooled_width = (width - 2) // 2 + 1
        flattened_size = pooled_height * pooled_width * 8

        self.linear = ModularLinearLayer(flattened_size, self.num_classes)
        self.softmax = SoftmaxLayer()

        self.layers_ = [
            self.conv,
            self.relu,
            self.pool,
            self.linear,
            self.softmax
        ]

        self.layers = self.layers_
        self._layers = self.layers_
        self.network = self.layers_
        self.model = self.layers_

    def _one_hot(self, y):
        encoded = np.zeros((y.shape[0], self.num_classes), dtype=np.float64)
        encoded[np.arange(y.shape[0]), y] = 1.0
        return encoded

    def _iterate_minibatches(self, X, y):
        indices = np.arange(X.shape[0])
        self._rng.shuffle(indices)

        for start in range(0, X.shape[0], self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            yield X[batch_indices], y[batch_indices]

    def _forward(self, X):
        output = self.conv(X)
        output = self.relu(output)
        output = self.pool(output)

        self._pooled_shape = output.shape
        output = output.reshape(output.shape[0], -1)

        output = self.linear(output)
        output = self.softmax(output)

        return output

    def fit(self, X, y):
        """
        Train the CNN.

        Args:
            X: numpy array of shape (n_samples, height, width, channels)
            y: numpy array of shape (n_samples,) with integer class labels
        """
        self._validate_params()
        X, y = self._validate_X_y(X, y)

        self._rng = np.random.default_rng(self.random_state)

        np.random.seed(self.random_state)
        self._build_network()

        self.loss_curve_ = []

        for _ in range(self.epochs):
            total_loss = 0.0
            total_examples = 0

            for X_batch, y_batch in self._iterate_minibatches(X, y):
                probabilities = self._forward(X_batch)
                y_encoded = self._one_hot(y_batch)

                loss = -np.mean(
                    np.sum(
                        y_encoded * np.log(np.clip(probabilities, 1e-12, 1.0)),
                        axis=1
                    )
                )

                total_loss += loss * X_batch.shape[0]
                total_examples += X_batch.shape[0]

                gradient = (probabilities - y_encoded) / X_batch.shape[0]

                gradient = self.linear.backward(gradient)
                gradient = gradient.reshape(self._pooled_shape)
                gradient = self.pool.backward(gradient)
                gradient = self.relu.backward(gradient)
                self.conv.backward(gradient)

                self.linear.update(self.lr)
                self.conv.update(self.lr)

            self.loss_curve_.append(total_loss / total_examples)

        return self

    def predict(self, X):
        """
        Predict class labels.

        Args:
            X: numpy array of shape (n_samples, height, width, channels)

        Returns:
            numpy array of shape (n_samples,) with predicted class labels
        """
        X = self._validate_X(X)

        if not self.layers_:
            raise ValueError("Model must be fitted before prediction")

        probabilities = self._forward(X)
        return np.argmax(probabilities, axis=1)