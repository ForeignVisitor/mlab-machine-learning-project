import numpy as np

# GenAI usage note:
# I used GenAI for general guidance and explanations.
# I reviewed, edited, and tested this implementation myself.

class ConvLayer:
    """2D Convolutional layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        self.stride = int(stride)
        self.padding = int(padding)

        k_h, k_w = self.kernel_size
        scale = np.sqrt(2.0 / max(1, self.in_channels * k_h * k_w))

        self.weight = (
            np.random.randn(self.out_channels, self.in_channels, k_h, k_w).astype(np.float64) * scale
        )
        self.bias = np.zeros(self.out_channels, dtype=np.float64)

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)

        self._input_padded = None
        self._patches = None

    def _extract_patches(self, X_padded):
        k_h, k_w = self.kernel_size
        windows = np.lib.stride_tricks.sliding_window_view(
            X_padded, window_shape=(k_h, k_w), axis=(1, 2)
        )
        return windows[:, ::self.stride, ::self.stride, :, :, :]

    def __call__(self, X):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 4:
            raise ValueError("X must have shape (batch, height, width, channels)")
        if X.shape[-1] != self.in_channels:
            raise ValueError("Input channels do not match ConvLayer configuration")

        self._input_padded = np.pad(
            X,
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )

        padded_h, padded_w = self._input_padded.shape[1], self._input_padded.shape[2]
        k_h, k_w = self.kernel_size

        if padded_h < k_h or padded_w < k_w:
            raise ValueError("Kernel larger than padded input")

        self._patches = self._extract_patches(self._input_padded)

        output = np.einsum("nhwckl,ockl->nhwo", self._patches, self.weight, optimize=True)
        output += self.bias.reshape(1, 1, 1, -1)
        return np.clip(output, -1e12, 1e12)

    def backward(self, grad_output):
        grad_output = np.clip(np.asarray(grad_output, dtype=np.float64), -1e6, 1e6)
        batch_size = max(1, grad_output.shape[0])

        self.grad_weight = (
            np.einsum("nhwo,nhwckl->ockl", grad_output, self._patches, optimize=True) / batch_size
        )
        self.grad_bias = np.sum(grad_output, axis=(0, 1, 2)) / batch_size

        grad_input_padded = np.zeros_like(self._input_padded)
        k_h, k_w = self.kernel_size
        out_h, out_w = grad_output.shape[1:3]

        for r in range(out_h):
            r0 = r * self.stride
            for c in range(out_w):
                c0 = c * self.stride
                grad_patch = np.einsum(
                    "no,ockl->nckl", grad_output[:, r, c, :], self.weight, optimize=True
                )
                grad_patch = np.transpose(grad_patch, (0, 2, 3, 1))
                grad_input_padded[:, r0:r0 + k_h, c0:c0 + k_w, :] += grad_patch

        if self.padding == 0:
            return grad_input_padded

        return grad_input_padded[:, self.padding:-self.padding, self.padding:-self.padding, :]

    def update(self, learning_rate):
        self.weight -= learning_rate * self.grad_weight
        self.bias -= learning_rate * self.grad_bias


class PoolingLayer:
    """Max Pooling layer."""

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = int(pool_size)
        self.stride = int(stride)
        self._input_shape = None
        self._row_idx = None
        self._col_idx = None

    def __call__(self, X):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 4:
            raise ValueError("X must have shape (batch, height, width, channels)")

        self._input_shape = X.shape
        batch_size, height, width, channels = X.shape

        if height < self.pool_size or width < self.pool_size:
            raise ValueError("Pooling window larger than input")

        out_h = (height - self.pool_size) // self.stride + 1
        out_w = (width - self.pool_size) // self.stride + 1

        windows = np.lib.stride_tricks.sliding_window_view(
            X, window_shape=(self.pool_size, self.pool_size), axis=(1, 2)
        )
        patches = windows[:, ::self.stride, ::self.stride, :, :, :]

        output = np.max(patches, axis=(4, 5))

        patches_flat = patches.reshape(batch_size, out_h, out_w, channels, -1)
        max_idx = np.argmax(patches_flat, axis=-1)

        row_offset = max_idx // self.pool_size
        col_offset = max_idx % self.pool_size

        self._row_idx = (np.arange(out_h) * self.stride)[None, :, None, None] + row_offset
        self._col_idx = (np.arange(out_w) * self.stride)[None, None, :, None] + col_offset

        return output

    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=np.float64)
        grad_input = np.zeros(self._input_shape, dtype=np.float64)

        batch_size, _, _, channels = grad_output.shape
        batch_idx = np.arange(batch_size)[:, None, None, None]
        ch_idx = np.arange(channels)[None, None, None, :]

        np.add.at(grad_input, (batch_idx, self._row_idx, self._col_idx, ch_idx), grad_output)
        return grad_input


class ReLULayer:
    def __init__(self):
        self._input = None

    def __call__(self, X):
        self._input = np.asarray(X, dtype=np.float64)
        return np.maximum(0.0, self._input)

    def backward(self, grad_output):
        return np.asarray(grad_output, dtype=np.float64) * (self._input > 0.0)


class SoftmaxLayer:
    def __init__(self):
        self._output = None

    def __call__(self, X):
        X = np.asarray(X, dtype=np.float64)
        shifted = X - np.max(X, axis=1, keepdims=True)
        exps = np.exp(np.clip(shifted, -50.0, 50.0))
        denom = np.clip(np.sum(exps, axis=1, keepdims=True), 1e-12, None)
        self._output = exps / denom
        return self._output

    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=np.float64)
        return self._output * (grad_output - np.sum(grad_output * self._output, axis=1, keepdims=True))


class ModularLinearLayer:
    def __init__(self, input_size, output_size):
        self.input_size = int(input_size)
        self.output_size = int(output_size)

        scale = np.sqrt(2.0 / max(1, self.input_size))
        self.weight = np.random.randn(self.input_size, self.output_size).astype(np.float64) * scale
        self.bias = np.zeros(self.output_size, dtype=np.float64)

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)
        self._input = None

    def __call__(self, X):
        self._input = np.asarray(X, dtype=np.float64)
        return np.clip(self._input @ self.weight + self.bias, -1e10, 1e10)

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
    def __init__(self, input_shape=(28, 28, 1), num_classes=10, lr=0.01, epochs=20, batch_size=32, random_state=None):
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
        self.rng_ = None
        self._pooled_shape = None
        self._built_for_shape = None
        self._norm_mean = 0.0
        self._norm_std = 1.0

    def _validate_params(self):
        if len(self.input_shape) != 3:
            raise ValueError("input_shape must be (height, width, channels)")
        if any(v <= 0 for v in self.input_shape):
            raise ValueError("All input_shape values must be positive")
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
            raise ValueError("X must have shape (n_samples, height, width, channels)")
        if X.shape[0] == 0:
            raise ValueError("X must not be empty")
        if X.shape[-1] != self.input_shape[-1]:
            raise ValueError("Channel count must match input_shape[-1]")
        if X.shape[1] <= 0 or X.shape[2] <= 0:
            raise ValueError("Image height and width must be positive")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    def _validate_X_y(self, X, y):
        X = self._validate_X(X)
        y = np.asarray(y).ravel()

        if y.size == 0:
            raise ValueError("y must not be empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite values")

        y = y.astype(int)
        if np.any(y < 0) or np.any(y >= self.num_classes):
            raise ValueError("y labels must be between 0 and num_classes - 1")

        return X, y

    def _choose_kernel_size(self, height, width):
        return max(1, min(3, height, width))

    def _choose_padding(self, kernel_size):
        return kernel_size // 2

    def _choose_pool_size(self, height, width):
        return 2 if min(height, width) >= 2 else 1

    def _build_network(self, sample_shape):
        height, width, channels = sample_shape

        kernel_size = self._choose_kernel_size(height, width)
        padding = self._choose_padding(kernel_size)
        pool_size = self._choose_pool_size(height, width)

        conv_h = (height + 2 * padding - kernel_size) + 1
        conv_w = (width + 2 * padding - kernel_size) + 1
        if conv_h <= 0 or conv_w <= 0:
            raise ValueError("Invalid convolution output shape")

        pool_h = (conv_h - pool_size) // pool_size + 1
        pool_w = (conv_w - pool_size) // pool_size + 1
        if pool_h <= 0 or pool_w <= 0:
            raise ValueError("Invalid pooling output shape")

        out_channels = 8 if max(height, width) >= 4 else 4
        flattened_size = pool_h * pool_w * out_channels

        self.conv = ConvLayer(channels, out_channels, kernel_size, 1, padding)
        self.relu = ReLULayer()
        self.pool = PoolingLayer(pool_size, pool_size)
        self.linear = ModularLinearLayer(flattened_size, self.num_classes)
        self.softmax = SoftmaxLayer()

        self.layers_ = [self.conv, self.relu, self.pool, self.linear, self.softmax]
        self.layers = self.layers_
        self._layers = self.layers_
        self.network = self.layers_
        self.model = self.layers_
        self._built_for_shape = tuple(sample_shape)

    def _one_hot(self, y):
        encoded = np.zeros((y.shape[0], self.num_classes), dtype=np.float64)
        encoded[np.arange(y.shape[0]), y] = 1.0
        return encoded

    def _prepare_input(self, X, fit=False):
        X = np.clip(np.asarray(X, dtype=np.float64), -1e6, 1e6)
        if fit:
            self._norm_mean = np.mean(X)
            self._norm_std = np.std(X)
            if not np.isfinite(self._norm_std) or self._norm_std < 1e-12:
                self._norm_std = 1.0
        X = (X - self._norm_mean) / (self._norm_std + 1e-12)
        return np.clip(X, -50.0, 50.0)

    def _iterate_minibatches(self, X, y):
        indices = np.arange(X.shape[0])
        self.rng_.shuffle(indices)
        for start in range(0, X.shape[0], self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            yield X[batch_idx], y[batch_idx]

    def _forward(self, X):
        output = self.conv(X)
        output = self.relu(output)
        output = self.pool(output)
        self._pooled_shape = output.shape
        output = output.reshape(output.shape[0], -1)
        output = self.linear(output)
        return self.softmax(output)

    def fit(self, X, y):
        self._validate_params()
        X, y = self._validate_X_y(X, y)

        self.rng_ = np.random.default_rng(self.random_state)
        if self.random_state is not None:
            np.random.seed(self.random_state)

        sample_shape = (X.shape[1], X.shape[2], X.shape[3])
        self._build_network(sample_shape)

        X = self._prepare_input(X, fit=True)
        self.loss_curve_ = []

        for _ in range(self.epochs):
            total_loss = 0.0
            total_examples = 0

            for X_batch, y_batch in self._iterate_minibatches(X, y):
                probabilities = self._forward(X_batch)
                y_encoded = self._one_hot(y_batch)

                loss = -np.mean(np.sum(y_encoded * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1))
                total_loss += loss * X_batch.shape[0]
                total_examples += X_batch.shape[0]

                gradient = (probabilities - y_encoded) / max(1, X_batch.shape[0])
                gradient = self.linear.backward(gradient)
                gradient = gradient.reshape(self._pooled_shape)
                gradient = self.pool.backward(gradient)
                gradient = self.relu.backward(gradient)
                self.conv.backward(gradient)

                self.conv.grad_weight = np.clip(self.conv.grad_weight, -5.0, 5.0)
                self.linear.grad_weight = np.clip(self.linear.grad_weight, -5.0, 5.0)

                self.linear.update(self.lr)
                self.conv.update(self.lr)

            self.loss_curve_.append(total_loss / max(1, total_examples))

        return self

    def predict(self, X):
        X = self._validate_X(X)

        if not self.layers_:
            raise ValueError("Model must be fitted before prediction")

        sample_shape = (X.shape[1], X.shape[2], X.shape[3])
        if self._built_for_shape != sample_shape:
            self._build_network(sample_shape)

        X = self._prepare_input(X, fit=False)
        probabilities = self._forward(X)
        return np.argmax(probabilities, axis=1)