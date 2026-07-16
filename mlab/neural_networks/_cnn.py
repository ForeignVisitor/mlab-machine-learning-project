# GenAI usage note:
# I used GenAI for general guidance and explanations.
# I reviewed, edited, and tested this implementation myself.

import numpy as np


class ConvLayer:
    """2D convolution for images shaped (batch, height, width, channels)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        rng=None,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = tuple(kernel_size)
        self.stride = stride
        self.padding = padding

        kernel_height, kernel_width = self.kernel_size
        scale = np.sqrt(2.0 / (in_channels * kernel_height * kernel_width))
        rng = np.random.default_rng() if rng is None else rng

        self.weight = rng.normal(
            0.0,
            scale,
            size=(out_channels, in_channels, kernel_height, kernel_width),
        )
        self.bias = np.zeros(out_channels)

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)

        self._input_padded = None
        self._patches = None
        self._padding = None

    def _extract_patches(self, X):
        kernel_height, kernel_width = self.kernel_size
        windows = np.lib.stride_tricks.sliding_window_view(
            X,
            window_shape=(kernel_height, kernel_width),
            axis=(1, 2),
        )
        return windows[:, ::self.stride, ::self.stride]

    def __call__(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 4:
            raise ValueError("X must have shape (batch, height, width, channels)")
        if X.shape[-1] != self.in_channels:
            raise ValueError("Input channels do not match ConvLayer configuration")

        height, width = X.shape[1:3]
        kernel_height, kernel_width = self.kernel_size

        # Pad very small images enough for one valid kernel window.
        extra_height = max(0, kernel_height - height)
        extra_width = max(0, kernel_width - width)

        padding_height = max(self.padding, int(np.ceil(extra_height / 2)))
        padding_width = max(self.padding, int(np.ceil(extra_width / 2)))
        self._padding = (padding_height, padding_width)

        self._input_padded = np.pad(
            X,
            (
                (0, 0),
                (padding_height, padding_height),
                (padding_width, padding_width),
                (0, 0),
            ),
        )
        self._patches = self._extract_patches(self._input_padded)

        output = np.einsum(
            "nhwckl,ockl->nhwo",
            self._patches,
            self.weight,
            optimize=True,
        )
        return output + self.bias

    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=float)
        batch_size = len(grad_output)

        self.grad_weight = np.einsum(
            "nhwo,nhwckl->ockl",
            grad_output,
            self._patches,
            optimize=True,
        ) / batch_size
        self.grad_bias = grad_output.sum(axis=(0, 1, 2)) / batch_size

        gradient_padded = np.zeros_like(self._input_padded)
        kernel_height, kernel_width = self.kernel_size
        output_height, output_width = grad_output.shape[1:3]

        for row in range(output_height):
            row_start = row * self.stride

            for column in range(output_width):
                column_start = column * self.stride

                patch_gradient = np.einsum(
                    "no,ockl->nckl",
                    grad_output[:, row, column],
                    self.weight,
                    optimize=True,
                )
                patch_gradient = np.transpose(patch_gradient, (0, 2, 3, 1))

                gradient_padded[
                    :,
                    row_start:row_start + kernel_height,
                    column_start:column_start + kernel_width,
                    :,
                ] += patch_gradient

        padding_height, padding_width = self._padding

        height_end = (
            -padding_height if padding_height > 0 else gradient_padded.shape[1]
        )
        width_end = (
            -padding_width if padding_width > 0 else gradient_padded.shape[2]
        )

        return gradient_padded[
            :,
            padding_height:height_end,
            padding_width:width_end,
            :,
        ]

    def update(self, learning_rate):
        self.weight -= learning_rate * self.grad_weight
        self.bias -= learning_rate * self.grad_bias


class PoolingLayer:
    """Max pooling layer."""

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

        self._input_shape = None
        self._row_indices = None
        self._column_indices = None
        self._identity = False

    def __call__(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 4:
            raise ValueError("X must have shape (batch, height, width, channels)")

        self._input_shape = X.shape
        batch_size, height, width, channels = X.shape

        pool_size = min(self.pool_size, height, width)
        stride = min(self.stride, height, width)

        if pool_size == 1:
            self._identity = True
            return X

        self._identity = False
        output_height = (height - pool_size) // stride + 1
        output_width = (width - pool_size) // stride + 1

        windows = np.lib.stride_tricks.sliding_window_view(
            X,
            window_shape=(pool_size, pool_size),
            axis=(1, 2),
        )
        patches = windows[:, ::stride, ::stride]
        output = patches.max(axis=(4, 5))

        flattened_patches = patches.reshape(
            batch_size,
            output_height,
            output_width,
            channels,
            -1,
        )
        maximum_indices = np.argmax(flattened_patches, axis=-1)

        row_offsets = maximum_indices // pool_size
        column_offsets = maximum_indices % pool_size

        self._row_indices = (
            np.arange(output_height)[None, :, None, None] * stride + row_offsets
        )
        self._column_indices = (
            np.arange(output_width)[None, None, :, None] * stride + column_offsets
        )

        return output

    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=float)

        if self._identity:
            return grad_output

        gradient = np.zeros(self._input_shape)
        batch_size, _, _, channels = grad_output.shape

        batch_indices = np.arange(batch_size)[:, None, None, None]
        channel_indices = np.arange(channels)[None, None, None, :]

        np.add.at(
            gradient,
            (
                batch_indices,
                self._row_indices,
                self._column_indices,
                channel_indices,
            ),
            grad_output,
        )
        return gradient


class ReLULayer:
    """ReLU activation."""

    def __init__(self):
        self._input = None

    def __call__(self, X):
        self._input = np.asarray(X, dtype=float)
        return np.maximum(0.0, self._input)

    def backward(self, grad_output):
        return np.asarray(grad_output, dtype=float) * (self._input > 0.0)


class SoftmaxLayer:
    """Softmax over class scores."""

    def __init__(self):
        self._output = None

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        shifted = X - X.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)

        self._output = exponentials / exponentials.sum(axis=1, keepdims=True)
        return self._output

    def backward(self, grad_output):
        weighted_sum = np.sum(
            grad_output * self._output,
            axis=1,
            keepdims=True,
        )
        return self._output * (grad_output - weighted_sum)


class ModularLinearLayer:
    """Fully connected layer."""

    def __init__(self, input_size, output_size, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        scale = np.sqrt(2.0 / input_size)

        self.weight = rng.normal(0.0, scale, size=(input_size, output_size))
        self.bias = np.zeros(output_size)

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)
        self._input = None

    def __call__(self, X):
        self._input = np.asarray(X, dtype=float)
        return self._input @ self.weight + self.bias

    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=float)
        batch_size = len(self._input)

        self.grad_weight = self._input.T @ grad_output / batch_size
        self.grad_bias = grad_output.sum(axis=0) / batch_size

        return grad_output @ self.weight.T

    def update(self, learning_rate):
        self.weight -= learning_rate * self.grad_weight
        self.bias -= learning_rate * self.grad_bias


class GlobalAvgPool:
    """Average each feature map over its height and width."""

    def __init__(self):
        self._input_shape = None

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        self._input_shape = X.shape
        return X.mean(axis=(1, 2))

    def backward(self, grad_output):
        _, height, width, _ = self._input_shape
        return np.broadcast_to(
            grad_output[:, None, None, :] / (height * width),
            self._input_shape,
        ).copy()


class CNNClassifier:
    """Conv-ReLU-MaxPool-GlobalAvgPool-Linear-Softmax classifier."""

    def __init__(
        self,
        input_shape=(28, 28, 1),
        num_classes=10,
        lr=0.01,
        epochs=20,
        batch_size=32,
        random_state=None,
    ):
        self.input_shape = tuple(input_shape)
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self.layers_ = []
        self.loss_curve_ = []

        self._rng = None
        self._built = False
        self._normalization_median = 0.0
        self._normalization_scale = 1.0

    def _validate_params(self):
        if len(self.input_shape) != 3 or any(size <= 0 for size in self.input_shape):
            raise ValueError("input_shape must contain positive height, width, and channels")

        if not isinstance(self.num_classes, (int, np.integer)) or self.num_classes < 2:
            raise ValueError("num_classes must be an integer of at least 2")

        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(self.epochs, (int, np.integer)) or self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if not isinstance(self.batch_size, (int, np.integer)) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

    def _validate_X(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 4:
            raise ValueError("X must have shape (batch, height, width, channels)")
        if X.shape[0] == 0:
            raise ValueError("X must not be empty")
        if X.shape[-1] != self.input_shape[-1]:
            raise ValueError("Channel count must match input_shape")
        if X.shape[1] == 0 or X.shape[2] == 0:
            raise ValueError("Image height and width must be positive")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")

        return X

    def _validate_training_data(self, X, y):
        X = self._validate_X(X)
        y = np.asarray(y).ravel()

        if len(y) == 0:
            raise ValueError("y must not be empty")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite values")
        if not np.all(y == y.astype(int)):
            raise ValueError("y must contain integer class labels")

        y = y.astype(int)

        if np.any(y < 0) or np.any(y >= self.num_classes):
            raise ValueError("y labels must be between 0 and num_classes - 1")

        return X, y

    def _build_network(self, channels):
        out_channels = 8

        self.conv = ConvLayer(
            channels,
            out_channels,
            kernel_size=3,
            padding=1,
            rng=self._rng,
        )
        self.relu = ReLULayer()
        self.pool = PoolingLayer(pool_size=2, stride=2)
        self.global_average_pool = GlobalAvgPool()
        self.linear = ModularLinearLayer(
            out_channels,
            self.num_classes,
            rng=self._rng,
        )
        self.softmax = SoftmaxLayer()

        self.layers_ = [
            self.conv,
            self.relu,
            self.pool,
            self.global_average_pool,
            self.linear,
            self.softmax,
        ]
        self._built = True

    @staticmethod
    def _cross_entropy(probabilities, y):
        correct_probabilities = probabilities[np.arange(len(y)), y]
        return float(-np.mean(np.log(np.clip(correct_probabilities, 1e-12, 1.0))))

    def _one_hot(self, y):
        encoded = np.zeros((len(y), self.num_classes))
        encoded[np.arange(len(y)), y] = 1.0
        return encoded

    def _prepare_input(self, X, fitting=False):
        if fitting:
            self._normalization_median = np.median(X)
            lower_quartile, upper_quartile = np.percentile(X, [25, 75])
            self._normalization_scale = upper_quartile - lower_quartile

            if self._normalization_scale < 1e-8:
                self._normalization_scale = np.std(X)

            if self._normalization_scale < 1e-8:
                self._normalization_scale = 1.0

        return (X - self._normalization_median) / self._normalization_scale

    def _mini_batches(self, X, y):
        indices = self._rng.permutation(len(X))

        for start in range(0, len(X), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            yield X[batch_indices], y[batch_indices]

    def _forward(self, X):
        output = self.conv(X)
        output = self.relu(output)
        output = self.pool(output)
        output = self.global_average_pool(output)
        output = self.linear(output)
        return self.softmax(output)

    def _backward(self, probabilities, y):
        # Softmax followed by cross-entropy has this simple gradient.
        gradient = (probabilities - self._one_hot(y)) / len(y)

        gradient = self.linear.backward(gradient)
        gradient = self.global_average_pool.backward(gradient)
        gradient = self.pool.backward(gradient)
        gradient = self.relu.backward(gradient)
        self.conv.backward(gradient)

    def fit(self, X, y):
        self._validate_params()
        X, y = self._validate_training_data(X, y)

        self._rng = np.random.default_rng(self.random_state)
        self._build_network(X.shape[-1])
        X = self._prepare_input(X, fitting=True)
        self.loss_curve_ = []

        for _ in range(self.epochs):
            total_loss = 0.0

            for X_batch, y_batch in self._mini_batches(X, y):
                probabilities = self._forward(X_batch)
                total_loss += self._cross_entropy(probabilities, y_batch) * len(X_batch)

                self._backward(probabilities, y_batch)

                # Clipping keeps training stable on small coursework datasets.
                self.conv.grad_weight = np.clip(self.conv.grad_weight, -5.0, 5.0)
                self.linear.grad_weight = np.clip(self.linear.grad_weight, -5.0, 5.0)

                self.conv.update(self.lr)
                self.linear.update(self.lr)

            self.loss_curve_.append(total_loss / len(X))

        return self

    def predict_proba(self, X):
        if not self._built:
            raise ValueError("Model must be fitted before prediction")

        X = self._validate_X(X)
        return self._forward(self._prepare_input(X))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)