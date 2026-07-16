# GenAI usage note:
# I used GenAI for general guidance and explanations.
# I reviewed, edited, and tested this implementation myself.

import copy

import numpy as np


class ModularLinearLayer:
    """Fully connected layer: X @ weight + bias."""

    def __init__(self, input_size, output_size):
        self.weight = np.zeros((input_size, output_size), dtype=float)
        self.bias = np.zeros(output_size, dtype=float)

        self._input = None
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)

    def __call__(self, X):
        self._input = np.asarray(X, dtype=float)
        return self._input @ self.weight + self.bias

    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=float)

        self.grad_weight = self._input.T @ grad_output
        self.grad_bias = grad_output.sum(axis=0)

        return grad_output @ self.weight.T

    def update(self, learning_rate):
        self.weight -= learning_rate * self.grad_weight
        self.bias -= learning_rate * self.grad_bias


class SigmoidLayer:
    """Sigmoid activation."""

    def __init__(self):
        self._output = None

    def __call__(self, X):
        X = np.clip(np.asarray(X, dtype=float), -50.0, 50.0)
        self._output = 1.0 / (1.0 + np.exp(-X))
        return self._output

    def backward(self, grad_output):
        return grad_output * self._output * (1.0 - self._output)


class TanhLayer:
    """Hyperbolic tangent activation."""

    def __init__(self):
        self._output = None

    def __call__(self, X):
        self._output = np.tanh(np.asarray(X, dtype=float))
        return self._output

    def backward(self, grad_output):
        return grad_output * (1.0 - self._output ** 2)


class ReLULayer:
    """ReLU activation."""

    def __init__(self):
        self._input = None

    def __call__(self, X):
        self._input = np.asarray(X, dtype=float)
        return np.maximum(0.0, self._input)

    def backward(self, grad_output):
        return grad_output * (self._input > 0.0)


class SoftmaxLayer:
    """Softmax activation for multi-class classification."""

    def __init__(self):
        self._output = None

    def __call__(self, X):
        X = np.asarray(X, dtype=float)

        # Subtracting the row maximum keeps exp() numerically stable.
        shifted = X - X.max(axis=1, keepdims=True)
        exp_values = np.exp(shifted)

        self._output = exp_values / exp_values.sum(axis=1, keepdims=True)
        return self._output

    def backward(self, grad_output):
        weighted_sum = np.sum(
            grad_output * self._output,
            axis=1,
            keepdims=True,
        )
        return self._output * (grad_output - weighted_sum)


class MLPRegressor:
    """Multi-layer perceptron regressor trained with backpropagation."""

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
    ):
        # These aliases keep the model compatible with the assignment API.
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

        self.validation_fraction = 0.2
        self.early_stopping = True
        self.patience = 20
        self.tol = 1e-6
        self.batch_size = 32
        self.gradient_clip_value = 5.0
        self.lr_decay = 0.995

        self.layers_ = []
        self.loss_curve_ = []
        self.validation_scores_ = []

        self._rng = None
        self._input_size = None
        self._output_size = None
        self._feature_means = None

    def _validate_params(self):
        if isinstance(self.hidden_layer_sizes, (int, np.integer)):
            self.hidden_layer_sizes = (self.hidden_layer_sizes,)
        elif isinstance(self.hidden_layer_sizes, list):
            self.hidden_layer_sizes = tuple(self.hidden_layer_sizes)

        if not isinstance(self.hidden_layer_sizes, tuple):
            raise ValueError("hidden_layer_sizes must be a tuple, list, or integer")

        for size in self.hidden_layer_sizes:
            if not isinstance(size, (int, np.integer)) or size <= 0:
                raise ValueError("Hidden layer sizes must be positive integers")

        if not isinstance(self.lr, (int, float, np.integer, np.floating)):
            raise ValueError("lr must be numeric")
        if self.lr <= 0:
            raise ValueError("lr must be positive")

        if not isinstance(self.epochs, (int, np.integer)) or self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")

        if not isinstance(self.alpha, (int, float, np.integer, np.floating)):
            raise ValueError("alpha must be numeric")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")

        if self.activation not in ("relu", "sigmoid", "tanh"):
            raise ValueError("activation must be 'relu', 'sigmoid', or 'tanh'")

        if self.random_state is not None and not isinstance(
            self.random_state,
            (int, np.integer),
        ):
            raise ValueError("random_state must be an integer or None")

    @staticmethod
    def _validate_X(X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] == 0:
            raise ValueError("X must not be empty")
        if np.any(np.isinf(X)):
            raise ValueError("X must not contain infinite values")

        return X

    def _validate_training_data(self, X, y):
        X = self._validate_X(X)
        y = np.asarray(y, dtype=float)

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim != 2:
            raise ValueError("y must be a 1D or 2D array")

        if y.shape[0] == 0:
            raise ValueError("y must not be empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite values")

        return X, y

    def _fit_imputer(self, X):
        valid_counts = np.sum(~np.isnan(X), axis=0)
        feature_sums = np.nansum(X, axis=0)

        # An entirely missing feature is replaced by zero.
        self._feature_means = np.divide(
            feature_sums,
            valid_counts,
            out=np.zeros(X.shape[1]),
            where=valid_counts > 0,
        )

    def _impute_missing(self, X):
        X = X.copy()
        rows, columns = np.where(np.isnan(X))
        X[rows, columns] = self._feature_means[columns]
        return X

    def _activation_layer(self):
        if self.activation == "relu":
            return ReLULayer()
        if self.activation == "sigmoid":
            return SigmoidLayer()
        return TanhLayer()

    def _initialize_weights(self, input_size, output_size, output_layer=False):
        layer = ModularLinearLayer(input_size, output_size)

        if output_layer or self.activation != "relu":
            limit = np.sqrt(6.0 / (input_size + output_size))
            layer.weight = self._rng.uniform(
                -limit,
                limit,
                size=(input_size, output_size),
            )
        else:
            standard_deviation = np.sqrt(2.0 / input_size)
            layer.weight = self._rng.normal(
                0.0,
                standard_deviation,
                size=(input_size, output_size),
            )
            layer.bias.fill(0.01)

        return layer

    def _build_network(self, input_size, output_size):
        self.layers_ = []
        previous_size = input_size

        for hidden_size in self.hidden_layer_sizes:
            self.layers_.append(
                self._initialize_weights(previous_size, hidden_size)
            )
            self.layers_.append(self._activation_layer())
            previous_size = hidden_size

        self.layers_.append(
            self._initialize_weights(
                previous_size,
                output_size,
                output_layer=True,
            )
        )

    def _forward(self, X):
        output = X

        for layer in self.layers_:
            output = layer(output)

        return output

    def _backward(self, gradient):
        for layer in reversed(self.layers_):
            gradient = layer.backward(gradient)

    def _apply_regularization(self):
        if self.alpha == 0:
            return

        for layer in self.layers_:
            if isinstance(layer, ModularLinearLayer):
                layer.grad_weight += self.alpha * layer.weight

    def _clip_gradients(self):
        for layer in self.layers_:
            if isinstance(layer, ModularLinearLayer):
                layer.grad_weight = np.clip(
                    layer.grad_weight,
                    -self.gradient_clip_value,
                    self.gradient_clip_value,
                )
                layer.grad_bias = np.clip(
                    layer.grad_bias,
                    -self.gradient_clip_value,
                    self.gradient_clip_value,
                )

    def _update_layers(self, learning_rate):
        for layer in self.layers_:
            if isinstance(layer, ModularLinearLayer):
                layer.update(learning_rate)

    def _loss(self, y_true, y_pred):
        mean_squared_error = np.mean((y_pred - y_true) ** 2)

        l2_penalty = sum(
            np.sum(layer.weight ** 2)
            for layer in self.layers_
            if isinstance(layer, ModularLinearLayer)
        )

        return float(mean_squared_error + 0.5 * self.alpha * l2_penalty)

    def _split_validation_data(self, X, y):
        if len(X) < 5:
            return X, y, None, None

        indices = self._rng.permutation(len(X))
        validation_size = max(1, int(self.validation_fraction * len(X)))

        validation_indices = indices[:validation_size]
        training_indices = indices[validation_size:]

        return (
            X[training_indices],
            y[training_indices],
            X[validation_indices],
            y[validation_indices],
        )

    def _mini_batches(self, X, y):
        indices = self._rng.permutation(len(X))

        for start in range(0, len(X), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            yield X[batch_indices], y[batch_indices]

    def _best_state(self):
        return copy.deepcopy(self.layers_)

    def fit(self, X, y):
        self._validate_params()
        X, y = self._validate_training_data(X, y)

        self._rng = np.random.default_rng(self.random_state)

        self._fit_imputer(X)
        X = self._impute_missing(X)

        self._input_size = X.shape[1]
        self._output_size = y.shape[1]

        X_train, y_train, X_validation, y_validation = (
            self._split_validation_data(X, y)
        )
        self._build_network(self._input_size, self._output_size)

        self.loss_curve_ = []
        self.validation_scores_ = []

        best_validation_loss = np.inf
        best_layers = None
        epochs_without_improvement = 0
        learning_rate = float(self.lr)

        for _ in range(self.epochs):
            for X_batch, y_batch in self._mini_batches(X_train, y_train):
                predictions = self._forward(X_batch)

                # Derivative of mean squared error.
                gradient = 2.0 * (predictions - y_batch) / len(X_batch)

                self._backward(gradient)
                self._apply_regularization()
                self._clip_gradients()
                self._update_layers(learning_rate)

            train_predictions = self._forward(X_train)
            self.loss_curve_.append(self._loss(y_train, train_predictions))

            if X_validation is not None:
                validation_predictions = self._forward(X_validation)
                validation_loss = self._loss(
                    y_validation,
                    validation_predictions,
                )
                self.validation_scores_.append(validation_loss)

                if validation_loss + self.tol < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_layers = self._best_state()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if self.early_stopping and epochs_without_improvement >= self.patience:
                    self.layers_ = best_layers
                    break

            learning_rate = max(learning_rate * self.lr_decay, 1e-5)

        return self

    def predict(self, X):
        if not self.layers_:
            raise ValueError("Model must be fitted before prediction")

        X = self._validate_X(X)

        if X.shape[1] != self._input_size:
            raise ValueError("X must have the same number of features as during fit")

        predictions = self._forward(self._impute_missing(X))

        if self._output_size == 1:
            return predictions.ravel()

        return predictions