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
        """Forward pass: X @ weight + bias"""
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
        """Forward: 1 / (1 + exp(-X))"""
        X = np.asarray(X, dtype=np.float64)
        X = np.clip(X, -50.0, 50.0)
        self._output = 1.0 / (1.0 + np.exp(-X))
        return self._output

    def backward(self, grad_output):
        """Backward: grad * sigmoid(X) * (1 - sigmoid(X))"""
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
    """ReLU activation function."""

    def __init__(self):
        self._input = None

    def __call__(self, X):
        """Forward: max(0, X)"""
        X = np.asarray(X, dtype=np.float64)
        self._input = X
        return np.maximum(0.0, X)

    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=np.float64)
        grad_input = grad_output.copy()
        grad_input[self._input <= 0.0] = 0.0
        return grad_input


class SoftmaxLayer:
    """Softmax activation (for multi-class output)."""

    def __init__(self):
        self._output = None

    def __call__(self, X):
        X = np.asarray(X, dtype=np.float64)
        shifted = X - np.max(X, axis=1, keepdims=True)
        exp_values = np.exp(np.clip(shifted, -50.0, 50.0))
        sums = np.sum(exp_values, axis=1, keepdims=True)
        sums = np.clip(sums, 1e-12, None)
        self._output = exp_values / sums
        return self._output

    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=np.float64)
        return self._output * (
            grad_output - np.sum(grad_output * self._output, axis=1, keepdims=True)
        )


class MLPRegressor:
    def __init__(
        self,
        hidden_layer_sizes=(50, 30),
        lr=0.01,
        epochs=100,
        random_state=None,
        alpha=0.0001,
        activation='relu',
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
        self._layers = self.layers_
        self.network = self.layers_

        self.validation_fraction = 0.2
        self.early_stopping = True
        self.patience = 20
        self.tol = 1e-6
        self.batch_size = 32
        self.gradient_clip_value = 5.0
        self.lr_decay = 0.995

        self.loss_curve_ = []
        self.validation_scores_ = []

        self._rng = None
        self._input_size = None
        self._output_size = None
        self._feature_means = None

    def _validate_params(self):
        if isinstance(self.hidden_layer_sizes, int):
            self.hidden_layer_sizes = (self.hidden_layer_sizes,)
        elif isinstance(self.hidden_layer_sizes, list):
            self.hidden_layer_sizes = tuple(self.hidden_layer_sizes)

        if not isinstance(self.hidden_layer_sizes, tuple):
            raise ValueError("hidden_layer_sizes must be a tuple, list, or int")

        for size in self.hidden_layer_sizes:
            if not isinstance(size, (int, np.integer)) or size <= 0:
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
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.size == 0:
            raise ValueError("X must not be empty")
        if np.any(np.isinf(X)):
            raise ValueError("X must not contain infinite values")

        return X

    def _validate_X_y(self, X, y):
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
        feature_means = np.nanmean(X, axis=0)
        feature_means = np.where(np.isnan(feature_means), 0.0, feature_means)
        self._feature_means = feature_means

    def _impute_missing(self, X):
        X = np.asarray(X, dtype=np.float64).copy()
        nan_mask = np.isnan(X)
        if np.any(nan_mask):
            X[nan_mask] = np.take(self._feature_means, np.where(nan_mask)[1])
        return X

    def _get_activation_layer(self):
        if self.activation == "relu":
            return ReLULayer()
        if self.activation == "sigmoid":
            return SigmoidLayer()
        return TanhLayer()

    def _he_initialize(self, input_size, output_size):
        std = np.sqrt(2.0 / max(1, input_size))
        return self._rng.normal(
            0.0, std, size=(input_size, output_size)
        ).astype(np.float64)

    def _xavier_initialize(self, input_size, output_size):
        limit = np.sqrt(6.0 / max(1, input_size + output_size))
        return self._rng.uniform(
            -limit, limit, size=(input_size, output_size)
        ).astype(np.float64)

    def _initialize_linear_layer(self, input_size, output_size, is_output=False):
        layer = ModularLinearLayer(input_size, output_size)

        if is_output:
            layer.weight = self._xavier_initialize(input_size, output_size)
            layer.bias = np.zeros(output_size, dtype=np.float64)
            return layer

        if self.activation == "relu":
            layer.weight = self._he_initialize(input_size, output_size)
            layer.bias = np.full(output_size, 0.01, dtype=np.float64)
        else:
            layer.weight = self._xavier_initialize(input_size, output_size)
            layer.bias = np.zeros(output_size, dtype=np.float64)

        return layer

    def _build_network(self, input_size, output_size):
        self.layers_ = []
        self.layers = self.layers_
        self._layers = self.layers_
        self.network = self.layers_

        previous_size = input_size

        for hidden_size in self.hidden_layer_sizes:
            linear = self._initialize_linear_layer(previous_size, hidden_size)
            self.layers_.append(linear)
            self.layers_.append(self._get_activation_layer())
            previous_size = hidden_size

        output_layer = self._initialize_linear_layer(
            previous_size, output_size, is_output=True
        )
        self.layers_.append(output_layer)

    def _forward(self, X):
        output = X
        for layer in self.layers_:
            output = layer(output)
        return output

    def _backward(self, grad_output):
        grad = grad_output
        for layer in reversed(self.layers_):
            grad = layer.backward(grad)

    def _apply_regularization(self, batch_size):
        if self.alpha == 0:
            return

        for layer in self.layers_:
            if isinstance(layer, ModularLinearLayer):
                layer.grad_weight += (self.alpha / max(1, batch_size)) * layer.weight

    def _clip_gradients(self):
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

    def _update(self, learning_rate):
        for layer in self.layers_:
            if isinstance(layer, ModularLinearLayer):
                layer.update(learning_rate)

    def _compute_loss(self, y_true, y_pred):
        mse = np.mean((y_pred - y_true) ** 2)

        l2_penalty = 0.0
        for layer in self.layers_:
            if isinstance(layer, ModularLinearLayer):
                l2_penalty += np.sum(layer.weight ** 2)

        return float(mse + 0.5 * self.alpha * l2_penalty / max(1, y_true.shape[0]))

    def _split_validation_data(self, X, y):
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

    def _iterate_minibatches(self, X, y):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        self._rng.shuffle(indices)

        for start in range(0, n_samples, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            yield X[batch_idx], y[batch_idx]

    def _capture_best_state(self):
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

        self._rng = np.random.default_rng(self.random_state)

        self._fit_imputer(X)
        X = self._impute_missing(X)

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values after imputation")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite values")

        self._input_size = X.shape[1]
        self._output_size = y.shape[1]

        X_train, y_train, X_val, y_val = self._split_validation_data(X, y)
        self._build_network(self._input_size, self._output_size)

        best_val_loss = np.inf
        best_state = None
        epochs_without_improvement = 0
        current_lr = float(self.lr)

        self.loss_curve_ = []
        self.validation_scores_ = []

        for _ in range(self.epochs):
            for X_batch, y_batch in self._iterate_minibatches(X_train, y_train):
                predictions = self._forward(X_batch)
                grad_output = (2.0 / max(1, X_batch.shape[0])) * (predictions - y_batch)

                self._backward(grad_output)
                self._apply_regularization(X_batch.shape[0])
                self._clip_gradients()
                self._update(current_lr)

            current_lr = max(current_lr * self.lr_decay, 1e-5)

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
        predictions = self._forward(X)

        if predictions.shape[1] == 1:
            return predictions.ravel()

        return predictions