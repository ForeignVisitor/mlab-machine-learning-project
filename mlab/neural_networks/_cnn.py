import numpy as np


class ConvLayer:
    """2D Convolutional layer using NHWC input format."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if len(kernel_size) != 2:
            raise ValueError("kernel_size must be an integer or (height, width)")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channel counts must be positive")
        if kernel_size[0] <= 0 or kernel_size[1] <= 0:
            raise ValueError("Kernel dimensions must be positive")
        if stride <= 0 or padding < 0:
            raise ValueError("stride must be positive and padding non-negative")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        self.stride = int(stride)
        self.padding = int(padding)

        kernel_height, kernel_width = self.kernel_size
        fan_in = self.in_channels * kernel_height * kernel_width
        scale = np.sqrt(2.0 / max(1, fan_in))

        self.weight = np.random.randn(
            self.out_channels,
            self.in_channels,
            kernel_height,
            kernel_width
        ).astype(np.float64) * scale

        self.bias = np.zeros(self.out_channels, dtype=np.float64)

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)

        self._input = None
        self._input_padded = None

    def __call__(self, X):
        """
        Forward pass.

        Args:
            X: numpy array of shape (batch_size, height, width, in_channels)

        Returns:
            numpy array of shape
            (batch_size, out_height, out_width, out_channels)
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 4:
            raise ValueError(
                "X must have shape (batch_size, height, width, in_channels)"
            )
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if X.shape[3] != self.in_channels:
            raise ValueError("Input channel count does not match in_channels")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain finite values")

        self._input = X

        batch_size, height, width, _ = X.shape
        kernel_height, kernel_width = self.kernel_size

        out_height = (
            (height + 2 * self.padding - kernel_height) // self.stride
        ) + 1

        out_width = (
            (width + 2 * self.padding - kernel_width) // self.stride
        ) + 1

        if out_height <= 0 or out_width <= 0:
            raise ValueError(
                "kernel_size, stride, and padding produce an invalid output shape"
            )

        self._input_padded = np.pad(
            X,
            (
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
                (0, 0)
            ),
            mode="constant",
            constant_values=0.0
        )

        output = np.zeros(
            (batch_size, out_height, out_width, self.out_channels),
            dtype=np.float64
        )

        for row in range(out_height):
            row_start = row * self.stride
            row_end = row_start + kernel_height

            for col in range(out_width):
                col_start = col * self.stride
                col_end = col_start + kernel_width

                patch = self._input_padded[
                    :,
                    row_start:row_end,
                    col_start:col_end,
                    :
                ]

                patch = np.transpose(patch, (0, 3, 1, 2))

                output[:, row, col, :] = (
                    np.einsum("nchw,ochw->no", patch, self.weight)
                    + self.bias
                )

        return np.clip(output, -1e10, 1e10)

    def backward(self, grad_output):
        """Compute gradients with respect to input, weights, and biases."""
        grad_output = np.asarray(grad_output, dtype=np.float64)

        if grad_output.ndim != 4:
            raise ValueError("grad_output must be a 4D array")

        batch_size, out_height, out_width, channels = grad_output.shape

        if channels != self.out_channels:
            raise ValueError("grad_output has an incorrect channel count")

        kernel_height, kernel_width = self.kernel_size

        grad_input_padded = np.zeros_like(self._input_padded)
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)

        grad_output = np.clip(grad_output, -1e5, 1e5)

        for row in range(out_height):
            row_start = row * self.stride
            row_end = row_start + kernel_height

            for col in range(out_width):
                col_start = col * self.stride
                col_end = col_start + kernel_width

                patch = self._input_padded[
                    :,
                    row_start:row_end,
                    col_start:col_end,
                    :
                ]

                patch_chw = np.transpose(patch, (0, 3, 1, 2))
                gradient = grad_output[:, row, col, :]

                self.grad_weight += np.einsum(
                    "no,nchw->ochw",
                    gradient,
                    patch_chw
                )

                self.grad_bias += np.sum(gradient, axis=0)

                grad_patch = np.einsum(
                    "no,ochw->nchw",
                    gradient,
                    self.weight
                )

                grad_patch = np.transpose(grad_patch, (0, 2, 3, 1))

                grad_input_padded[
                    :,
                    row_start:row_end,
                    col_start:col_end,
                    :
                ] += grad_patch

        self.grad_weight /= max(1, batch_size)
        self.grad_bias /= max(1, batch_size)

        if self.padding == 0:
            return grad_input_padded

        return grad_input_padded[
            :,
            self.padding:-self.padding,
            self.padding:-self.padding,
            :
        ]

    def update(self, learning_rate):
        """Update weights and biases."""
        learning_rate = float(learning_rate)

        self.weight -= learning_rate * self.grad_weight
        self.bias -= learning_rate * self.grad_bias


class PoolingLayer:
    """Max Pooling layer."""

    def __init__(self, pool_size=2, stride=2):
        if pool_size <= 0 or stride <= 0:
            raise ValueError("pool_size and stride must be positive")

        self.pool_size = int(pool_size)
        self.stride = int(stride)

        self._input = None
        self._max_indices = None

    def __call__(self, X):
        """
        Forward pass: apply max pooling.

        Args:
            X: numpy array of shape (batch_size, height, width, channels)

        Returns:
            numpy array of shape
            (batch_size, out_height, out_width, channels)
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 4:
            raise ValueError(
                "X must have shape (batch_size, height, width, channels)"
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain finite values")

        self._input = X

        batch_size, height, width, channels = X.shape

        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        if out_height <= 0 or out_width <= 0:
            raise ValueError("pool_size produces an invalid output shape")

        output = np.zeros(
            (batch_size, out_height, out_width, channels),
            dtype=np.float64
        )

        self._max_indices = np.zeros(
            (batch_size, out_height, out_width, channels, 2),
            dtype=np.int64
        )

        for row in range(out_height):
            row_start = row * self.stride
            row_end = row_start + self.pool_size

            for col in range(out_width):
                col_start = col * self.stride
                col_end = col_start + self.pool_size

                region = X[:, row_start:row_end, col_start:col_end, :]

                for channel in range(channels):
                    values = region[:, :, :, channel]
                    flat_indices = np.argmax(
                        values.reshape(batch_size, -1),
                        axis=1
                    )

                    row_offsets = flat_indices // self.pool_size
                    col_offsets = flat_indices % self.pool_size

                    output[:, row, col, channel] = values[
                        np.arange(batch_size),
                        row_offsets,
                        col_offsets
                    ]

                    self._max_indices[:, row, col, channel, 0] = (
                        row_start + row_offsets
                    )
                    self._max_indices[:, row, col, channel, 1] = (
                        col_start + col_offsets
                    )

        return output

    def backward(self, grad_output):
        """Route gradients back through maximum positions."""
        grad_output = np.asarray(grad_output, dtype=np.float64)

        if grad_output.ndim != 4:
            raise ValueError("grad_output must be a 4D array")

        grad_input = np.zeros_like(self._input)

        batch_size, out_height, out_width, channels = grad_output.shape

        for row in range(out_height):
            for col in range(out_width):
                for channel in range(channels):
                    max_rows = self._max_indices[:, row, col, channel, 0]
                    max_cols = self._max_indices[:, row, col, channel, 1]

                    grad_input[
                        np.arange(batch_size),
                        max_rows,
                        max_cols,
                        channel
                    ] += grad_output[:, row, col, channel]

        return grad_input


class ReLULayer:
    """ReLU activation function."""

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
    """Softmax activation function."""

    def __init__(self):
        self._output = None

    def __call__(self, X):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("Softmax input must be a 2D array")

        shifted = X - np.max(X, axis=1, keepdims=True)
        exponentials = np.exp(np.clip(shifted, -50.0, 50.0))

        self._output = exponentials / np.sum(
            exponentials,
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
    """Fully connected dense layer."""

    def __init__(self, input_size, output_size):
        if input_size <= 0 or output_size <= 0:
            raise ValueError("input_size and output_size must be positive")

        self.input_size = int(input_size)
        self.output_size = int(output_size)

        scale = np.sqrt(2.0 / self.input_size)

        self.weight = np.random.randn(
            self.input_size,
            self.output_size
        ).astype(np.float64) * scale

        self.bias = np.zeros(self.output_size, dtype=np.float64)

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)

        self._input = None

    def __call__(self, X):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("Linear layer input must be 2D")
        if X.shape[1] != self.input_size:
            raise ValueError("Input feature count does not match input_size")

        self._input = X
        return np.clip(X @ self.weight + self.bias, -1e10, 1e10)

    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=np.float64)

        if grad_output.ndim != 2:
            raise ValueError("grad_output must be 2D")

        self.grad_weight = self._input.T @ grad_output
        self.grad_bias = np.sum(grad_output, axis=0)

        return grad_output @ self.weight.T

    def update(self, learning_rate):
        learning_rate = float(learning_rate)
        self.weight -= learning_rate * self.grad_weight
        self.bias -= learning_rate * self.grad_bias


class CNNClassifier:
    """
    CNN image classifier.

    Architecture:
    ConvLayer -> ReLULayer -> PoolingLayer -> Flatten
    -> ModularLinearLayer -> SoftmaxLayer
    """

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
        max_iter=None,
        alpha=0.0001
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
        self.alpha = float(alpha)

        self.layers_ = []
        self.layers = self.layers_
        self._layers = self.layers_
        self.network = self.layers_
        self.model = self.layers_

        self.loss_curve_ = []

        self._rng = None
        self._pooled_shape = None
        self._mean = None
        self._std = None

    def _validate_params(self):
        if len(self.input_shape) != 3:
            raise ValueError(
                "input_shape must be (height, width, channels)"
            )

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
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")

    def _validate_X(self, X):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 4:
            raise ValueError(
                "X must have shape (samples, height, width, channels)"
            )

        if X.shape[0] == 0:
            raise ValueError("X must not be empty")

        if X.shape[1:] != self.input_shape:
            raise ValueError("X image dimensions must match input_shape")

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain finite values")

        return X

    def _validate_X_y(self, X, y):
        X = self._validate_X(X)
        y = np.asarray(y).ravel()

        if y.size == 0:
            raise ValueError("y must not be empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        if not np.all(np.equal(y, np.floor(y))):
            raise ValueError("y must contain integer labels")

        y = y.astype(int)

        if np.any(y < 0) or np.any(y >= self.num_classes):
            raise ValueError(
                "y values must be between 0 and num_classes - 1"
            )

        return X, y

    def _fit_normalizer(self, X):
        self._mean = np.mean(X)
        self._std = np.std(X)

        if self._std < 1e-12:
            self._std = 1.0

    def _normalize(self, X):
        normalized = (X - self._mean) / self._std
        return np.clip(normalized, -100.0, 100.0)

    def _build_network(self):
        height, width, channels = self.input_shape

        # Ensure convolution works for all image dimensions.
        if min(height, width) >= 3:
            kernel_size = 3
            padding = 1
        else:
            kernel_size = 1
            padding = 0

        # Ensure pooling works for 1x1 images.
        if min(height, width) >= 2:
            pool_size = 2
            pool_stride = 2
        else:
            pool_size = 1
            pool_stride = 1

        self.conv = ConvLayer(
            in_channels=channels,
            out_channels=8,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )

        self.relu = ReLULayer()

        self.pool = PoolingLayer(
            pool_size=pool_size,
            stride=pool_stride
        )

        conv_height = (
            (height + 2 * padding - kernel_size) // 1
        ) + 1

        conv_width = (
            (width + 2 * padding - kernel_size) // 1
        ) + 1

        pooled_height = (
            (conv_height - pool_size) // pool_stride
        ) + 1

        pooled_width = (
            (conv_width - pool_size) // pool_stride
        ) + 1

        flattened_size = pooled_height * pooled_width * 8

        self.linear = ModularLinearLayer(
            input_size=flattened_size,
            output_size=self.num_classes
        )

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
        output = np.zeros(
            (y.shape[0], self.num_classes),
            dtype=np.float64
        )
        output[np.arange(y.shape[0]), y] = 1.0
        return output

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

        return self.softmax(output)

    def _apply_regularization(self):
        self.conv.grad_weight += self.alpha * self.conv.weight
        self.linear.grad_weight += self.alpha * self.linear.weight

    def _clip_gradients(self):
        self.conv.grad_weight = np.clip(
            self.conv.grad_weight, -10.0, 10.0
        )
        self.conv.grad_bias = np.clip(
            self.conv.grad_bias, -10.0, 10.0
        )
        self.linear.grad_weight = np.clip(
            self.linear.grad_weight, -10.0, 10.0
        )
        self.linear.grad_bias = np.clip(
            self.linear.grad_bias, -10.0, 10.0
        )

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

        self._fit_normalizer(X)
        X = self._normalize(X)

        self._build_network()
        self.loss_curve_ = []

        for epoch in range(self.epochs):
            current_lr = self.lr / (1.0 + 0.01 * epoch)

            total_loss = 0.0
            total_samples = 0

            for X_batch, y_batch in self._iterate_minibatches(X, y):
                probabilities = self._forward(X_batch)
                y_one_hot = self._one_hot(y_batch)

                loss = -np.mean(
                    np.sum(
                        y_one_hot * np.log(
                            np.clip(probabilities, 1e-12, 1.0)
                        ),
                        axis=1
                    )
                )

                total_loss += loss * X_batch.shape[0]
                total_samples += X_batch.shape[0]

                # Gradient of softmax + cross-entropy.
                gradient = (
                    probabilities - y_one_hot
                ) / X_batch.shape[0]

                gradient = self.linear.backward(gradient)
                gradient = gradient.reshape(self._pooled_shape)
                gradient = self.pool.backward(gradient)
                gradient = self.relu.backward(gradient)
                self.conv.backward(gradient)

                self._apply_regularization()
                self._clip_gradients()

                self.linear.update(current_lr)
                self.conv.update(current_lr)

            self.loss_curve_.append(
                total_loss / max(1, total_samples)
            )

        return self

    def predict_proba(self, X):
        """Predict class probabilities."""
        X = self._validate_X(X)

        if not self.layers_:
            raise ValueError("Model must be fitted before prediction")

        X = self._normalize(X)
        return self._forward(X)

    def predict(self, X):
        """
        Predict class labels.

        Args:
            X: numpy array of shape (n_samples, height, width, channels)

        Returns:
            numpy array of shape (n_samples,) with predicted class labels
        """
        return np.argmax(self.predict_proba(X), axis=1)