import numpy as np

# GenAI usage note:
# I used GenAI for general guidance and explanations.
# I reviewed, edited, and tested this implementation myself.


class ModularLinearLayer:
    """Fully connected (dense) layer."""

    def __init__(self, input_size, output_size):
        self.weight = np.zeros((input_size, output_size), dtype=float)
        self.bias = np.zeros(output_size, dtype=float)

        self._input = None
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)

    def __call__(self, X):
        """Forward pass: X @ weight + bias."""
        X = np.asarray(X, dtype=float)
        self._input = X
        return X @ self.weight + self.bias

    def backward(self, grad_output):
        """Backward pass: compute gradients for input, weight, and bias."""
        grad_output = np.asarray(grad_output, dtype=float)

        self.grad_weight = self._input.T @ grad_output
        self.grad_bias = np.sum(grad_output, axis=0)
        grad_input = grad_output @ self.weight.T

        return grad_input

    def update(self, learning_rate):
        """Update weights and biases using stored gradients."""
        self.weight -= learning_rate * self.grad_weight
        self.bias -= learning_rate * self.grad_bias


class SigmoidLayer:
    """Sigmoid activation function."""

    def __init__(self):
        self._output = None

    def __call__(self, X):
        """Forward: 1 / (1 + exp(-X))."""
        X = np.asarray(X, dtype=float)
        X = np.clip(X, -500.0, 500.0)
        self._output = 1.0 / (1.0 + np.exp(-X))
        return self._output

    def backward(self, grad_output):
        """Backward: grad * sigmoid(X) * (1 - sigmoid(X))."""
        grad_output = np.asarray(grad_output, dtype=float)
        return grad_output * self._output * (1.0 - self._output)


class TanhLayer:
    """Tanh activation function."""

    def __init__(self):
        self._output = None

    def __call__(self, X):
        """Forward pass for tanh."""
        X = np.asarray(X, dtype=float)
        self._output = np.tanh(X)
        return self._output

    def backward(self, grad_output):
        """Backward pass for tanh."""
        grad_output = np.asarray(grad_output, dtype=float)
        return grad_output * (1.0 - self._output ** 2)


class ReLULayer:
    """ReLU activation function."""

    def __init__(self):
        self._input = None

    def __call__(self, X):
        """Forward: max(0, X)."""
        X = np.asarray(X, dtype=float)
        self._input = X
        return np.maximum(0.0, X)

    def backward(self, grad_output):
        """Backward pass for ReLU."""
        grad_output = np.asarray(grad_output, dtype=float)
        grad_input = grad_output.copy()
        grad_input[self._input <= 0.0] = 0.0
        return grad_input


class SoftmaxLayer:
    """Softmax activation (for multi-class output)."""

    def __init__(self):
        self._output = None

    def __call__(self, X):
        """Forward pass for softmax."""
        X = np.asarray(X, dtype=float)
        shifted = X - np.max(X, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        self._output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self._output

    def backward(self, grad_output):
        """Backward pass for softmax."""
        grad_output = np.asarray(grad_output, dtype=float)
        return self._output * (
            grad_output - np.sum(grad_output * self._output, axis=1, keepdims=True)
        )


class MLPRegressor:
    """Multi-layer perceptron for regression using backpropagation."""

    def __init__(
        self,
        hidden_layer_sizes=(50, 30),
        lr=0.01,
        epochs=100,
        random_state=None,
        alpha=0.0001,
        activation="relu",
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

        self.hidden_layer_sizes = hidden_layer_sizes
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.alpha = alpha
        self.activation = activation

        self.layers_ = []
        self.layers = self.layers_
        self.network = self.layers_

        self._rng = None
        self._input_size = None
        self._output_size = None
        self._feature_means = None

    def _validate_params(self):
        """Validate constructor parameters."""
        if isinstance(self.hidden_layer_sizes, int):
            self.hidden_layer_sizes = (self.hidden_layer_sizes,)

        if not isinstance(self.hidden_layer_sizes, tuple):
            raise ValueError("hidden_layer_sizes must be a tuple or int")
        if any(
            (not isinstance(size, (int, np.integer)) or size <= 0)
            for size in self.hidden_layer_sizes
        ):
            raise ValueError("All hidden layer sizes must be positive integers")

        if not isinstance(self.lr, (int, float, np.integer, np.floating)):
            raise ValueError("lr must be numeric")
        if self.lr <= 0:
            raise ValueError("lr must be positive")

        if not isinstance(self.epochs, (int, np.integer)):
            raise ValueError("epochs must be an integer")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

        if not isinstance(self.alpha, (int, float, np.integer, np.floating)):
            raise ValueError("alpha must be numeric")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")

        if self.activation not in ("relu", "sigmoid", "tanh"):
            raise ValueError("activation must be 'relu', 'sigmoid', or 'tanh'")

        if self.random_state is not None and not isinstance(
            self.random_state, (int, np.integer)
        ):
            raise ValueError("random_state must be an integer or None")

    def _validate_X(self, X):
        """Validate input features."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.size == 0:
            raise ValueError("X must not be empty")
        if np.any(np.isinf(X)):
            raise ValueError("X must not contain infinite values")

        return X

    def _validate_X_y(self, X, y):
        """Validate features and targets."""
        X = self._validate_X(X)
        y = np.asarray(y, dtype=float)

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim != 2:
            raise ValueError("y must be a 1D or 2D array")

        if y.size == 0:
            raise ValueError("y must not be empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if np.any(np.isinf(y)):
            raise ValueError("y must not contain infinite values")

        return X, y

    def _fit_imputer(self, X):
        """Store feature means for missing-value imputation."""
        feature_means = np.nanmean(X, axis=0)
        feature_means = np.where(np.isnan(feature_means), 0.0, feature_means)
        self._feature_means = feature_means

    def _impute_missing(self, X):
        """Replace NaN values with stored feature means."""
        X = np.asarray(X, dtype=float).copy()
        nan_mask = np.isnan(X)
        if np.any(nan_mask):
            X[nan_mask] = np.take(self._feature_means, np.where(nan_mask)[1])
        return X

    def _get_activation_layer(self):
        """Create the configured hidden activation layer."""
        if self.activation == "relu":
            return ReLULayer()
        if self.activation == "sigmoid":
            return SigmoidLayer()
        return TanhLayer()

    def _initialize_linear_layer(self, input_size, output_size):
        """Create and initialize one dense layer."""
        layer = ModularLinearLayer(input_size, output_size)

        if self.activation == "relu":
            scale = np.sqrt(2.0 / input_size)
        else:
            scale = np.sqrt(1.0 / input_size)

        layer.weight = self._rng.normal(
            loc=0.0,
            scale=scale,
            size=(input_size, output_size)
        )
        layer.bias = np.zeros(output_size, dtype=float)

        return layer

    def _build_network(self, input_size, output_size):
        """Build the sequence of layers."""
        self.layers_ = []
        self.layers = self.layers_
        self.network = self.layers_

        previous_size = input_size

        for hidden_size in self.hidden_layer_sizes:
            linear = self._initialize_linear_layer(previous_size, hidden_size)
            self.layers_.append(linear)
            self.layers_.append(self._get_activation_layer())
            previous_size = hidden_size

        output_layer = ModularLinearLayer(previous_size, output_size)
        output_scale = np.sqrt(1.0 / max(1, previous_size))
        output_layer.weight = self._rng.normal(
            loc=0.0,
            scale=output_scale,
            size=(previous_size, output_size)
        )
        output_layer.bias = np.zeros(output_size, dtype=float)
        self.layers_.append(output_layer)

    def _forward(self, X):
        """Run a forward pass through the network."""
        output = X
        for layer in self.layers_:
            output = layer(output)
        return output

    def _backward(self, grad_output):
        """Run backpropagation through the network."""
        grad = grad_output
        for layer in reversed(self.layers_):
            grad = layer.backward(grad)

    def _apply_regularization(self, n_samples):
        """Add L2 regularization gradient to linear layers."""
        if self.alpha == 0:
            return

        for layer in self.layers_:
            if isinstance(layer, ModularLinearLayer):
                layer.grad_weight += (self.alpha / n_samples) * layer.weight

    def _update(self):
        """Update all linear layers."""
        for layer in self.layers_:
            if hasattr(layer, "update"):
                layer.update(self.lr)

    def fit(self, X, y):
        """
        Train the network using backpropagation.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,) or (n_samples, n_outputs)
        """
        self._validate_params()
        X, y = self._validate_X_y(X, y)

        self._fit_imputer(X)
        X = self._impute_missing(X)

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values after imputation")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite values")

        self._rng = np.random.default_rng(self.random_state)
        self._input_size = X.shape[1]
        self._output_size = y.shape[1]

        self._build_network(self._input_size, self._output_size)

        n_samples = X.shape[0]
        normalizer = y.size

        for _ in range(self.epochs):
            predictions = self._forward(X)

            grad_output = (2.0 / normalizer) * (predictions - y)

            self._backward(grad_output)
            self._apply_regularization(n_samples)
            self._update()

        return self

    def predict(self, X):
        """
        Predict target values.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,)
        """
        X = self._validate_X(X)

        if not self.layers_:
            raise ValueError("Model must be fitted before prediction")
        if X.shape[1] != self._input_size:
            raise ValueError("X must have the same number of features as during fit")

        X = self._impute_missing(X)
        predictions = self._forward(X)

        if predictions.shape[1] == 1:
            return predictions.ravel()

        return predictions