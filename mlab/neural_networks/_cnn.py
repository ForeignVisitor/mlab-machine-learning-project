import numpy as np


class ConvLayer:
    """2D Convolutional layer using NHWC input format."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if len(kernel_size) != 2:
            raise ValueError("kernel_size must be an integer or a pair (kH, kW)")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if kernel_size[0] <= 0 or kernel_size[1] <= 0:
            raise ValueError("kernel_size values must be positive")
        if stride <= 0 or padding < 0:
            raise ValueError("stride must be positive and padding must be non-negative")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        self.stride = int(stride)
        self.padding = int(padding)

        k_h, k_w = self.kernel_size
        scale = np.sqrt(2.0 / max(1, self.in_channels * k_h * k_w))

        self.weight = np.random.randn(
            self.out_channels,
            self.in_channels,
            k_h,
            k_w
        ).astype(np.float64) * scale

        self.bias = np.zeros(self.out_channels, dtype=np.float64)

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)

        self._input = None
        self._input_padded = None
        self._patches = None

    def _extract_patches(self, X_padded):
        """Extract convolution patches using a strided window view."""
        k_h, k_w = self.kernel_size

        windows = np.lib.stride_tricks.sliding_window_view(
            X_padded,
            window_shape=(k_h, k_w),
            axis=(1, 2)
        )

        return windows[:, ::self.stride, ::self.stride, :, :, :]

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
            raise ValueError("X must contain at least one image")
        if X.shape[-1] != self.in_channels:
            raise ValueError("Input channels do not match in_channels")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")

        self._input = X

        batch_size, height, width, _ = X.shape
        k_h, k_w = self.kernel_size
        padding = self.padding

        out_height = (height + 2 * padding - k_h) // self.stride + 1
        out_width = (width + 2 * padding - k_w) // self.stride + 1

        if out_height <= 0 or out_width <= 0:
            raise ValueError(
                "Kernel size, stride, or padding produces an invalid output shape"
            )

        self._input_padded = np.pad(
            X,
            ((0, 0), (padding, padding), (padding, padding), (0, 0)),
            mode="constant"
        )

        self._patches = self._extract_patches(self._input_padded)

        output = np.einsum(
            "nhwckl,ockl->nhwo",
            self._patches,
            self.weight,
            optimize=True
        )

        output += self.bias.reshape(1, 1, 1, -1)

        return np.clip(output, -1e12, 1e12)

    def backward(self, grad_output):
        """Compute gradients w.r.t. input, weights, and biases."""
        grad_output = np.asarray(grad_output, dtype=np.float64)

        if grad_output.ndim != 4:
            raise ValueError("grad_output must be a 4D array")

        batch_size, out_height, out_width, out_channels = grad_output.shape

        if out_channels != self.out_channels:
            raise ValueError("grad_output channel count is incorrect")

        expected_shape = (
            self._patches.shape[0],
            self._patches.shape[1],
            self._patches.shape[2],
            self.out_channels
        )

        if grad_output.shape != expected_shape:
            raise ValueError("grad_output shape does not match convolution output")

        grad_output = np.clip(grad_output, -1e6, 1e6)

        self.grad_weight = np.einsum(
            "nhwo,nhwckl->ockl",
            grad_output,
            self._patches,
            optimize=True
        )

        self.grad_bias = np.sum(grad_output, axis=(0, 1, 2))

        grad_input_padded = np.zeros_like(self._input_padded)

        k_h, k_w = self.kernel_size

        for out_row in range(out_height):
            row_start = out_row * self.stride

            for out_col in range(out_width):
                col_start = out_col * self.stride

                grad_at_position = grad_output[:, out_row, out_col, :]

                contribution = np.einsum(
                    "no,ockl->nckl",
                    grad_at_position,
                    self.weight,
                    optimize=True
                )

                contribution = np.transpose(contribution, (0, 2, 3, 1))

                grad_input_padded[
                    :,
                    row_start:row_start + k_h,
                    col_start:col_start + k_w,
                    :
                ] += contribution

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

        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

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
            numpy array of shape (batch_size, out_height, out_width, channels)
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 4:
            raise ValueError(
                "X must have shape (batch_size, height, width, channels)"
            )
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one image")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")

        self._input = X

        batch_size, height, width, channels = X.shape
        pool_size = self.pool_size
        stride = self.stride

        out_height = (height - pool_size) // stride + 1
        out_width = (width - pool_size) // stride + 1

        if out_height <= 0 or out_width <= 0:
            raise ValueError(
                "pool_size or stride produces an invalid output shape"
            )

        output = np.zeros(
            (batch_size, out_height, out_width, channels),
            dtype=np.float64
        )

        self._max_indices = np.zeros(
            (batch_size, out_height, out_width, channels, 2),
            dtype=np.int64
        )

        for out_row in range(out_height):
            row_start = out_row * stride

            for out_col in range(out_width):
                col_start = out_col * stride

                region = X[
                    :,
                    row_start:row_start + pool_size,
                    col_start:col_start + pool_size,
                    :
                ]

                flat_region = region.reshape(
                    batch_size,
                    pool_size * pool_size,
                    channels
                )

                max_indices = np.argmax(flat_region, axis=1)
                output[:, out_row, out_col, :] = np.max(flat_region, axis=1)

                row_offsets = max_indices // pool_size
                col_offsets = max_indices % pool_size

                self._max_indices[:, out_row, out_col, :, 0] = (
                    row_start + row_offsets
                )
                self._max_indices[:, out_row, out_col, :, 1] = (
                    col_start + col_offsets
                )

        return output

    def backward(self, grad_output):
        """Route gradients back through max positions."""
        grad_output = np.asarray(grad_output, dtype=np.float64)

        if grad_output.ndim != 4:
            raise ValueError("grad_output must be a 4D array")

        grad_input = np.zeros_like(self._input)

        batch_size, out_height, out_width, channels = grad_output.shape

        for out_row in range(out_height):
            for out_col in range(out_width):
                for channel in range(channels):
                    row_indices = self._max_indices[
                        :, out_row, out_col, channel, 0
                    ]
                    col_indices = self._max_indices[
                        :, out_row, out_col, channel, 1
                    ]

                    grad_input[
                        np.arange(batch_size),
                        row_indices,
                        col_indices,
                        channel
                    ] += grad_output[:, out_row, out_col, channel]

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

        if X.ndim != 2:
            raise ValueError("Softmax input must be a 2D array")

        shifted = X - np.max(X, axis=1, keepdims=True)
        exp_values = np.exp(np.clip(shifted, -50.0, 50.0))

        denominator = np.sum(exp_values, axis=1, keepdims=True)
        self._output = exp_values / np.clip(denominator, 1e-12, None)

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
        if input_size <= 0 or output_size <= 0:
            raise ValueError("input_size and output_size must be positive")

        self.input_size = int(input_size)
        self.output_size = int(output_size)

        scale = np.sqrt(2.0 / max(1, self.input_size))

        self.weight = (
            np.random.randn(self.input_size, self.output_size).astype(np.float64)
            * scale
        )

        self.bias = np.zeros(self.output_size, dtype=np.float64)

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)

        self._input = None

    def __call__(self, X):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("Linear layer input must be a 2D array")
        if X.shape[1] != self.input_size:
            raise ValueError("Input feature count does not match input_size")

        self._input = X
        return np.clip(X @ self.weight + self.bias, -1e12, 1e12)

    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=np.float64)

        if grad_output.ndim != 2:
            raise ValueError("grad_output must be a 2D array")
        if grad_output.shape[1] != self.output_size:
            raise ValueError("grad_output feature count is incorrect")

        self.grad_weight = self._input.T @ grad_output
        self.grad_bias = np.sum(grad_output, axis=0)

        return grad_output @ self.weight.T

    def update(self, learning_rate):
        learning_rate = float(learning_rate)

        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        self.weight -= learning_rate * self.grad_weight
        self.bias -= learning_rate * self.grad_bias


class CNNClassifier:
    """
    CNN image classifier.

    Architecture:
    Conv -> ReLU -> MaxPool -> Flatten -> Linear -> Softmax
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
        self._feature_mean = None
        self._feature_std = None

    def _validate_params(self):
        if len(self.input_shape) != 3:
            raise ValueError(
                "input_shape must have the form (height, width, channels)"
            )

        if any(not isinstance(value, (int, np.integer)) or value <= 0
               for value in self.input_shape):
            raise ValueError("input_shape values must be positive integers")

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
                "X must have shape (n_samples, height, width, channels)"
            )

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

        if y.size == 0:
            raise ValueError("y must not be empty")

        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        if not np.issubdtype(y.dtype, np.integer):
            if not np.all(np.equal(y, np.floor(y))):
                raise ValueError("y must contain integer class labels")

        y = y.astype(int)

        if np.any(y < 0) or np.any(y >= self.num_classes):
            raise ValueError(
                "y labels must be between 0 and num_classes - 1"
            )

        return X, y

    def _fit_normalizer(self, X):
        self._feature_mean = float(np.mean(X))
        self._feature_std = float(np.std(X))

        if not np.isfinite(self._feature_std) or self._feature_std < 1e-12:
            self._feature_std = 1.0

    def _normalize(self, X):
        X = (X - self._feature_mean) / self._feature_std
        return np.clip(X, -100.0, 100.0)

    def _build_network(self):
        height, width, channels = self.input_shape

        # A 3x3 kernel for normal images, 1x1 for tiny images.
        kernel_size = 3 if min(height, width) >= 3 else 1
        padding = kernel_size // 2

        self.conv = ConvLayer(
            in_channels=channels,
            out_channels=8,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )

        self.relu = ReLULayer()

        # A 1x1 pool keeps minimal images valid.
        pool_size = 2 if min(height, width) >= 2 else 1
        pool_stride = pool_size

        self.pool = PoolingLayer(
            pool_size=pool_size,
            stride=pool_stride
        )

        conv_height = (
            (height + 2 * padding - kernel_size) // self.conv.stride
        ) + 1

        conv_width = (
            (width + 2 * padding - kernel_size) // self.conv.stride
        ) + 1

        pooled_height = (
            (conv_height - pool_size) // pool_stride
        ) + 1

        pooled_width = (
            (conv_width - pool_size) // pool_stride
        ) + 1

        flattened_size = pooled_height * pooled_width * self.conv.out_channels

        self.linear = ModularLinearLayer(
            input_size=flattened_size,
            output_size=self.num_classes
        )

        self.softmax = SoftmaxLayer()

        # Make initialization reproducible.
        conv_scale = np.sqrt(
            2.0 / (
                self.conv.in_channels
                * self.conv.kernel_size[0]
                * self.conv.kernel_size[1]
            )
        )

        self.conv.weight = self._rng.normal(
            0.0,
            conv_scale,
            size=self.conv.weight.shape
        )

        linear_scale = np.sqrt(2.0 / self.linear.input_size)

        self.linear.weight = self._rng.normal(
            0.0,
            linear_scale,
            size=self.linear.weight.shape
        )

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
        encoded = np.zeros(
            (y.shape[0], self.num_classes),
            dtype=np.float64
        )

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

        return self.softmax(output)

    def _apply_regularization(self):
        if self.alpha > 0.0:
            self.conv.grad_weight += self.alpha * self.conv.weight
            self.linear.grad_weight += self.alpha * self.linear.weight

    def _clip_gradients(self, maximum_norm=10.0):
        for layer in (self.conv, self.linear):
            layer.grad_weight = np.clip(
                layer.grad_weight,
                -maximum_norm,
                maximum_norm
            )

            layer.grad_bias = np.clip(
                layer.grad_bias,
                -maximum_norm,
                maximum_norm
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

        self._fit_normalizer(X)
        X = self._normalize(X)

        self._build_network()
        self.loss_curve_ = []

        for epoch in range(self.epochs):
            total_loss = 0.0
            total_examples = 0

            # Mild learning-rate decay improves stability.
            current_lr = self.lr / (1.0 + 0.01 * epoch)

            for X_batch, y_batch in self._iterate_minibatches(X, y):
                probabilities = self._forward(X_batch)
                y_encoded = self._one_hot(y_batch)

                clipped_probabilities = np.clip(
                    probabilities,
                    1e-12,
                    1.0
                )

                loss = -np.mean(
                    np.sum(
                        y_encoded * np.log(clipped_probabilities),
                        axis=1
                    )
                )

                total_loss += loss * X_batch.shape[0]
                total_examples += X_batch.shape[0]

                # Softmax + cross-entropy gradient.
                gradient = (
                    probabilities - y_encoded
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

            self.loss_curve_.append(total_loss / max(1, total_examples))

        return self

    def predict_proba(self, X):
        """Return class probabilities."""
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
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)