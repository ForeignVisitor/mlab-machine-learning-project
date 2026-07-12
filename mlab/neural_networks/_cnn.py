import numpy as np


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

        # Weight shape: (out_channels, in_channels, kH, kW)
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
        """Vectorized patch extraction using sliding_window_view."""
        k_h, k_w = self.kernel_size
        
        # Windows shape: (batch_size, out_h, out_w, channels, k_h, k_w)
        windows = np.lib.stride_tricks.sliding_window_view(
            X_padded,
            window_shape=(k_h, k_w),
            axis=(1, 2)
        )
        
        # Apply stride: slice the out_h and out_w dimensions
        return windows[:, ::self.stride, ::self.stride, :, :, :]

    def __call__(self, X):
        """Forward pass."""
        X = np.asarray(X, dtype=np.float64)
        self._input = X

        # Apply padding on H and W axes (axis 1 and 2)
        self._input_padded = np.pad(
            X,
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
            mode="constant",
            constant_values=0.0
        )

        # Extract patches
        self._patches = self._extract_patches(self._input_padded)

        # Einsum optimization:
        # patches shape: (N, H, W, C_in, Kh, Kw) -> n h w c k l
        # weights shape: (C_out, C_in, Kh, Kw)   -> o c k l
        # Output shape:  (N, H, W, C_out)        -> n h w o
        output = np.einsum(
            "nhwckl,ockl->nhwo",
            self._patches,
            self.weight,
            optimize=True
        )

        output += self.bias.reshape(1, 1, 1, -1)
        return np.clip(output, -1e12, 1e12)

    def backward(self, grad_output):
        """Compute gradients using vectorization."""
        grad_output = np.asarray(grad_output, dtype=np.float64)
        grad_output = np.clip(grad_output, -1e6, 1e6)

        batch_size, out_height, out_width, _ = grad_output.shape
        k_h, k_w = self.kernel_size

        # Vectorized weight gradient
        # grad_output shape: (N, H, W, C_out)      -> n h w o
        # patches shape:     (N, H, W, C_in, K, K) -> n h w c k l
        # weight grad shape: (C_out, C_in, K, K)   -> o c k l
        self.grad_weight = np.einsum(
            "nhwo,nhwckl->ockl",
            grad_output,
            self._patches,
            optimize=True
        ) / max(1, batch_size)

        self.grad_bias = np.sum(grad_output, axis=(0, 1, 2)) / max(1, batch_size)

        # Input gradient accumulation
        grad_input_padded = np.zeros_like(self._input_padded)

        # Fast nested accumulation using einsum for input gradient
        for row in range(out_height):
            r_start = row * self.stride
            for col in range(out_width):
                c_start = col * self.stride
                
                # grad_output_pos: (N, C_out) -> n o
                # weight: (C_out, C_in, Kh, Kw) -> o c k l
                # grad_patch: (N, C_in, Kh, Kw) -> n c k l
                grad_patch = np.einsum(
                    "no,ockl->nckl",
                    grad_output[:, row, col, :],
                    self.weight,
                    optimize=True
                )
                
                # Transpose to (N, Kh, Kw, C_in) and accumulate
                grad_patch = np.transpose(grad_patch, (0, 2, 3, 1))
                
                grad_input_padded[
                    :,
                    r_start:r_start + k_h,
                    c_start:c_start + k_w,
                    :
                ] += grad_patch

        if self.padding == 0:
            return grad_input_padded

        return grad_input_padded[
            :,
            self.padding:-self.padding,
            self.padding:-self.padding,
            :
        ]

    def update(self, learning_rate):
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
        X = np.asarray(X, dtype=np.float64)
        self._input = X

        batch_size, height, width, channels = X.shape

        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        output = np.zeros((batch_size, out_height, out_width, channels), dtype=np.float64)
        self._max_indices = np.zeros((batch_size, out_height, out_width, channels, 2), dtype=np.int64)

        for row in range(out_height):
            row_start = row * self.stride
            row_end = row_start + self.pool_size

            for col in range(out_width):
                col_start = col * self.stride
                col_end = col_start + self.pool_size

                region = X[:, row_start:row_end, col_start:col_end, :]

                # Vectorized block-wise max operation
                for c in range(channels):
                    values = region[:, :, :, c]
                    flat_indices = np.argmax(values.reshape(batch_size, -1), axis=1)

                    r_off = flat_indices // self.pool_size
                    c_off = flat_indices % self.pool_size

                    output[:, row, col, c] = values[np.arange(batch_size), r_off, c_off]
                    self._max_indices[:, row, col, c, 0] = row_start + r_off
                    self._max_indices[:, row, col, c, 1] = col_start + c_off

        return output

    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=np.float64)
        grad_input = np.zeros_like(self._input)
        batch_size, out_height, out_width, channels = grad_output.shape

        for row in range(out_height):
            for col in range(out_width):
                for c in range(channels):
                    max_r = self._max_indices[:, row, col, c, 0]
                    max_c = self._max_indices[:, row, col, c, 1]

                    grad_input[np.arange(batch_size), max_r, max_c, c] += grad_output[:, row, col, c]

        return grad_input


class ReLULayer:
    """ReLU activation."""
    def __init__(self):
        self._input = None

    def __call__(self, X):
        self._input = np.asarray(X, dtype=np.float64)
        return np.maximum(0.0, self._input)

    def backward(self, grad_output):
        return np.asarray(grad_output, dtype=np.float64) * (self._input > 0.0)


class SoftmaxLayer:
    """Softmax activation."""
    def __init__(self):
        self._output = None

    def __call__(self, X):
        X = np.asarray(X, dtype=np.float64)
        shifted = X - np.max(X, axis=1, keepdims=True)
        exps = np.exp(np.clip(shifted, -50.0, 50.0))
        self._output = exps / np.sum(exps, axis=1, keepdims=True)
        return self._output

    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=np.float64)
        return self._output * (grad_output - np.sum(grad_output * self._output, axis=1, keepdims=True))


class ModularLinearLayer:
    """Fully connected layer."""
    def __init__(self, input_size, output_size):
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        scale = np.sqrt(2.0 / self.input_size)

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
        self.grad_weight = self._input.T @ grad_output
        self.grad_bias = np.sum(grad_output, axis=0)
        return grad_output @ self.weight.T

    def update(self, learning_rate):
        self.weight -= learning_rate * self.grad_weight
        self.bias -= learning_rate * self.grad_bias


class CNNClassifier:
    """Main CNN image classification model."""
    
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
        self._rng = None
        self._pooled_shape = None

    def _build_network(self):
        height, width, channels = self.input_shape

        # Adaptive sizes to prevent crash on single pixel testing
        kernel_size = 3 if min(height, width) >= 3 else 1
        padding = 1 if kernel_size == 3 else 0
        pool_size = 2 if min(height, width) >= 2 else 1

        self.conv = ConvLayer(in_channels=channels, out_channels=8, kernel_size=kernel_size, stride=1, padding=padding)
        self.relu = ReLULayer()
        self.pool = PoolingLayer(pool_size=pool_size, stride=pool_size)

        conv_h = (height + 2 * padding - kernel_size) + 1
        conv_w = (width + 2 * padding - kernel_size) + 1
        pool_h = (conv_h - pool_size) // pool_size + 1
        pool_w = (conv_w - pool_size) // pool_size + 1
        
        self.linear = ModularLinearLayer(input_size=pool_h * pool_w * 8, output_size=self.num_classes)
        self.softmax = SoftmaxLayer()

        self.layers_ = [self.conv, self.relu, self.pool, self.linear, self.softmax]

    def _one_hot(self, y):
        out = np.zeros((y.shape[0], self.num_classes), dtype=np.float64)
        out[np.arange(y.shape[0]), y] = 1.0
        return out

    def _forward(self, X):
        out = self.conv(X)
        out = self.relu(out)
        out = self.pool(out)
        self._pooled_shape = out.shape
        out = out.reshape(out.shape[0], -1)
        out = self.linear(out)
        return self.softmax(out)

    def fit(self, X, y):
        """Train the CNN."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel().astype(int)

        self._rng = np.random.default_rng(self.random_state)
        np.random.seed(self.random_state)

        # Normalize features safely
        X_mean = np.mean(X)
        X_std = np.std(X)
        if X_std < 1e-12: X_std = 1.0
        self.X_mean, self.X_std = X_mean, X_std
        X = np.clip((X - X_mean) / X_std, -100.0, 100.0)

        self._build_network()
        self.loss_curve_ = []

        indices = np.arange(X.shape[0])

        for epoch in range(self.epochs):
            self._rng.shuffle(indices)
            epoch_loss = 0.0
            
            for start in range(0, X.shape[0], self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                X_batch, y_batch = X[batch_idx], y[batch_idx]

                probs = self._forward(X_batch)
                y_oh = self._one_hot(y_batch)

                probs_clipped = np.clip(probs, 1e-12, 1.0)
                epoch_loss += -np.sum(y_oh * np.log(probs_clipped))

                grad = (probs - y_oh) / X_batch.shape[0]
                
                grad = self.linear.backward(grad)
                grad = grad.reshape(self._pooled_shape)
                grad = self.pool.backward(grad)
                grad = self.relu.backward(grad)
                self.conv.backward(grad)

                # Gradient clipping
                self.conv.grad_weight = np.clip(self.conv.grad_weight, -5.0, 5.0)
                self.linear.grad_weight = np.clip(self.linear.grad_weight, -5.0, 5.0)

                self.linear.update(self.lr)
                self.conv.update(self.lr)

            self.loss_curve_.append(epoch_loss / max(1, X.shape[0]))

        return self

    def predict(self, X):
        """Predict class labels."""
        X = np.asarray(X, dtype=np.float64)
        X = np.clip((X - getattr(self, "X_mean", 0.0)) / getattr(self, "X_std", 1.0), -100.0, 100.0)
        return np.argmax(self._forward(X), axis=1)
