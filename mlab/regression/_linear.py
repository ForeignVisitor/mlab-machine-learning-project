import numpy as np

def _validate_training_data(X, y):
    """Check that X and y can be used for supervised learning."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    elif y.ndim != 1:
        raise ValueError("y must be a 1D array or a column vector")

    if X.shape[0] == 0:
        raise ValueError("X and y must not be empty")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    return X, y


def _validate_prediction_data(X, n_features):
    """Check new samples before making predictions."""
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    if X.shape[0] == 0:
        raise ValueError("X must not be empty")

    if X.shape[1] != n_features:
        raise ValueError("X has a different number of features than the training data")

    return X


class LinearRegressor:
    """Linear regression fitted with the closed-form least-squares solution."""

    def __init__(self):
        self.weights_ = None
        self.bias_ = None

    def fit(self, X, y):
        X, y = _validate_training_data(X, y)

        # Add a column of ones so the first coefficient becomes the bias.
        X_augmented = np.c_[np.ones(X.shape[0]), X]
        coefficients = np.linalg.lstsq(X_augmented, y, rcond=None)[0]

        self.bias_ = coefficients[0]
        self.weights_ = coefficients[1:]
        return self

    def predict(self, X):
        if self.weights_ is None:
            raise ValueError("Model must be fitted before prediction")

        X = _validate_prediction_data(X, len(self.weights_))
        return X @ self.weights_ + self.bias_


class SGDRegression:
    """Linear regression trained with mini-batch gradient descent."""

    def __init__(
        self,
        learning_rate=0.01,
        n_iterations=1000,
        batch_size=32,
        random_state=None,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if n_iterations <= 0:
            raise ValueError("n_iterations must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.random_state = random_state
        self.weights_ = None
        self.bias_ = None

    def fit(self, X, y):
        X, y = _validate_training_data(X, y)

        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_iterations):
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[start:start + self.batch_size]
                y_batch = y_shuffled[start:start + self.batch_size]

                errors = X_batch @ self.weights_ + self.bias_ - y_batch

                # Gradient of mean squared error for this mini-batch.
                gradient_weights = 2 * X_batch.T @ errors / len(X_batch)
                gradient_bias = 2 * errors.mean()

                self.weights_ -= self.learning_rate * gradient_weights
                self.bias_ -= self.learning_rate * gradient_bias

        return self

    def predict(self, X):
        if self.weights_ is None:
            raise ValueError("Model must be fitted before prediction")

        X = _validate_prediction_data(X, len(self.weights_))
        return X @ self.weights_ + self.bias_