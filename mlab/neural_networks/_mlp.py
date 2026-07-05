import copy
import numpy as np

# GenAI usage note:
# I used GenAI for general guidance and explanations.
# I reviewed, edited, and tested this implementation myself.


class ModularLinearLayer:
    """Fully connected (dense) layer."""

    def __init__(self, input_size, output_size):
        self.weight = np.zeros((input_size, output_size), dtype=np.float64)
        self.bias = np.zeros(output_size, dtype=np.float64)

        self._input = None
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)

    def __call__(self, X):
        """Forward pass: X @ weight + bias."""
        X = np.asarray(X, dtype=np.float64)
        self._input = X
        return X @ self.weight + self.bias

    def backward(self, grad_output):
        """Backward pass: compute gradients w.r.t. input, weight, and bias."""
        grad_output = np.asarray(grad_output, dtype=np.float64)

        self.grad_weight = self._input.T @ grad_output
        self.grad_bias = np.sum(grad_output, axis=0)
        grad_input = grad_output @ self.weight.T

        return grad_input

    def update(self, learning_rate):
        """Update weights and biases using computed gradients."""
        self.weight -= learning_rate * self.grad_weight
        self.bias -= learning_rate * self.grad_bias


class SigmoidLayer:
    """Sigmoid activation function."""

    def __init__(self):
        self._output = None

    def __call__(self, X):
        """Forward: 1 / (1 + exp(-X))."""
        X = np.asarray(X, dtype=np.float64)
        X = np.clip(X, -50.0, 50.0)
        self._output = 1.0 / (1.0 + np.exp(-X))
        return self._output

    def backward(self, grad_output):
        """Backward: grad * sigmoid(X) * (1 - sigmoid(X))."""
        grad_output = np.asarray(grad_output, dtype=np.float64)
        return grad_output * self._output * (1.0 - self._output)


class TanhLayer:
    """Tanh activation function."""

    def __init__(self):
        self._output = None

    def __call__(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = np.clip(X, -50.0, 50.0)
        self._output = np.tanh(X)
        return self._output

    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=np.float64)
        return grad_output * (1.0 - self._output ** 2)


class ReLULayer:
    """Leaky-ReLU style activation for safer gradient flow."""

    def __init__(self, negative_slope=0.01):
        self.negative_slope = negative_slope
        self._input = None

    def __call__(self, X):
        X = np.asarray(X, dtype=np.float64)
        self._input = X
        return np.where(X > 0.0, X, self.negative_slope * X)

    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=np.float64)
        slope = np.where(self._input > 0.0, 1.0, self.negative_slope)
        return grad_output * slope


class ELULayer:
    """ELU activation function."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self._input = None

    def __call__(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = np.clip(X, -50.0, 50.0)
        self._input = X
        return np.where(X > 0.0, X, self.alpha * (np.exp(X) - 1.0))

    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=np.float64)
        grad = np.where(
            self._input > 0.0,
            1.0,
            self.alpha * np.exp(np.clip(self._input, -50.0, 50.0))
        )
        return grad_output * grad


class SoftmaxLayer:
    """Softmax activation (for multi-class output)."""

    def __init__(self):
        self._output = None

    def __call__(self, X):
        X = np.asarray(X, dtype=np.float64)
        shifted = X - np.max(X, axis=1, keepdims=True)
        exp_values = np.exp(np.clip(shifted, -50.0, 50.0))
        denom = np.sum(exp_values, axis=1, keepdims=True)
        denom = np.clip(denom, 1e-12, None)
        self._output = exp_values / denom
        return self._output

    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=np.float64)
        return self._output * (
            grad_output - np.sum(grad_output * self._output, axis=1, keepdims=True)
        )


class DropoutLayer:
    """Dropout regularization layer."""

    def __init__(self, dropout_rate=0.2, rng=None):
        self.dropout_rate = dropout_rate
        self.training = True
        self._mask = None
        self._rng = rng

    def __call__(self, X):
        X = np.asarray(X, dtype=np.float64)

        if not self.training or self.dropout_rate <= 0.0:
            self._mask = np.ones_like(X, dtype=np.float64)
            return X

        keep_prob = 1.0 - self.dropout_rate
        if self._rng is None:
            random_values = np.random.rand(*X.shape)
        else:
            random_values = self._rng.random(X.shape)

        self._mask = (random_values < keep_prob).astype(np.float64) / keep_prob
        return X * self._mask

    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=np.float64)
        return grad_output * self._mask


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
        max_iter=None,
        dropout_rate=0.2
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
        self.dropout_rate = dropout_rate

        self.layers_ = []
        self.layers = self.layers_
        self._layers = self.layers_
        self.network = self.layers_

        self._rng = None
        self._input_size = None
        self._output_size = None
        self._feature_means = None

        self.validation_fraction = 0.2
        self.early_stopping = True
        self.patience = 20
        self.tol = 1e-6
        self.gradient_clip_value = 5.0
        self.loss_curve_ = []
        self.validation_scores_ = []

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

        if self.activation not in ("relu", "sigmoid", "tanh", "elu"):
            raise ValueError("activation must be 'relu', 'sigmoid', 'tanh', or 'elu'")

        if not isinstance(self.dropout_rate, (int, float, np.integer, np.floating)):
            raise ValueError("dropout_rate must be numeric")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1)")

        if self.random_state is not None and not isinstance(
            self.random_state, (int, np.integer)
        ):
            raise ValueError("random_state must be an integer or None")

    def _validate_X(self, X):
        """Validate input features."""
        X = np.asarray(X, dtype=np.float64)

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
        y = np.asarray(y, dtype=np.float64)

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
        X = np.asarray(X, dtype=np.float64).copy()
        nan_mask = np.isnan(X)
        if np.any(nan_mask):
            X[nan_mask] = np.take(self._feature_means, np.where(nan_mask)[1])
        return X

    def _get_activation_layer(self):
        """Create the configured hidden activation layer."""
        if self.activation == "relu":
            return ReLULayer(negative_slope=0.01)
        if self.activation == "sigmoid":
            return SigmoidLayer()
        if self.activation == "tanh":
            return TanhLayer()
        return ELULayer(alpha=1.0)

    def _he_initialize(self, input_size, output_size):
        """He initialization for ReLU-like activations."""
        std = np.sqrt(2.0 / input_size)
        return self._rng.normal(
            0.0, std, size=(input_size, output_size)
        ).astype(np.float64)

    def _xavier_initialize(self, input_size, output_size):
        """Xavier/Glorot initialization for sigmoid/tanh-style activations."""
        limit = np.sqrt(6.0 / (input_size + output_size))
        return self._rng.uniform(
            -limit, limit, size=(input_size, output_size)
        ).astype(np.float64)

    def _initialize_linear_layer(self, input_size, output_size):
        """Create and initialize one dense layer."""
        layer = ModularLinearLayer(input_size, output_size)

        if self.activation in ("relu", "elu"):
            layer.weight = self._he_initialize(input_size, output_size)
            layer.bias = np.full(output_size, 0.01, dtype=np.float64)
        else:
            layer.weight = self._xavier_initialize(input_size, output_size)
            layer.bias = np.zeros(output_size, dtype=np.float64)

        return layer

    def _build_network(self, input_size, output_size):
        """Build the sequence of layers."""
        self.layers_ = []
        self.layers = self.layers_
        self._layers = self.layers_
        self.network = self.layers_

        previous_size = input_size

        for hidden_size in self.hidden_layer_sizes:
            linear = self._initialize_linear_layer(previous_size, hidden_size)
            self.layers_.append(linear)
            self.layers_.append(self._get_activation_layer())
            self.layers_.append(DropoutLayer(self.dropout_rate, rng=self._rng))
            previous_size = hidden_size

        output_layer = ModularLinearLayer(previous_size, output_size)
        output_layer.weight = self._xavier_initialize(previous_size, output_size)
        output_layer.bias = np.zeros(output_size, dtype=np.float64)
        self.layers_.append(output_layer)

    def _set_training_mode(self, training):
        """Enable or disable training mode for layers like dropout."""
        for layer in self.layers_:
            if isinstance(layer, DropoutLayer):
                layer.training = training

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

    def _clip_gradients(self):
        """Clip gradients for stability."""
        for layer in self.layers_:
            if isinstance(layer, ModularLinearLayer):
                layer.grad_weight = np.clip(
                    layer.grad_weight,
                    -self.gradient_clip_value,
                    self.gradient_clip_value
                )
                layer.grad_bias = np.clip(
                    layer.grad_bias,
                    -self.gradient_clip_value,
                    self.gradient_clip_value
                )

    def _update(self):
        """Update all linear layers."""
        for layer in self.layers_:
            if isinstance(layer, ModularLinearLayer):
                layer.update(self.lr)

    def _compute_loss(self, y_true, y_pred):
        """Compute regularized mean squared error."""
        error = y_pred - y_true
        mse = np.mean(error ** 2)

        l2_penalty = 0.0
        for layer in self.layers_:
            if isinstance(layer, ModularLinearLayer):
                l2_penalty += np.sum(layer.weight ** 2)

        return float(mse + self.alpha * l2_penalty / max(1, y_true.shape[0]))

    def _split_validation_data(self, X, y):
        """Split into train and validation sets."""
        n_samples = X.shape[0]
        if n_samples < 5:
            return X, y, None, None

        indices = np.arange(n_samples)
        self._rng.shuffle(indices)

        val_size = max(1, int(self.validation_fraction * n_samples))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        if train_indices.size == 0:
            return X, y, None, None

        return X[train_indices], y[train_indices], X[val_indices], y[val_indices]

    def _capture_best_state(self):
        """Capture the current network state."""
        return [copy.deepcopy(layer) for layer in self.layers_]

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

        X_train, y_train, X_val, y_val = self._split_validation_data(X, y)
        self._build_network(self._input_size, self._output_size)

        n_samples = X_train.shape[0]
        normalizer = y_train.size

        best_val_loss = np.inf
        best_state = None
        epochs_without_improvement = 0

        self.loss_curve_ = []
        self.validation_scores_ = []

        for _ in range(self.epochs):
            self._set_training_mode(True)
            predictions = self._forward(X_train)

            grad_output = (2.0 / max(1, normalizer)) * (predictions - y_train)
            self._backward(grad_output)
            self._apply_regularization(n_samples)
            self._clip_gradients()
            self._update()

            self._set_training_mode(False)
            train_pred = self._forward(X_train)
            train_loss = self._compute_loss(y_train, train_pred)
            self.loss_curve_.append(train_loss)

            if X_val is not None:
                val_pred = self._forward(X_val)
                val_loss = self._compute_loss(y_val, val_pred)
                self.validation_scores_.append(val_loss)

                if val_loss + self.tol < best_val_loss:
                    best_val_loss = val_loss
                    best_state = self._capture_best_state()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if self.early_stopping and epochs_without_improvement >= self.patience:
                    if best_state is not None:
                        self.layers_ = best_state
                        self.layers = self.layers_
                        self._layers = self.layers_
                        self.network = self.layers_
                    break

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
        self._set_training_mode(False)
        predictions = self._forward(X)

        if predictions.shape[1] == 1:
            return predictions.ravel()

        return predictions