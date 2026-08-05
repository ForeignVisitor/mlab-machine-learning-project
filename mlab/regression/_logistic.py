import numpy as np


def _validate_training_data(X, y):
    """Check training data for binary classification."""
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

    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must contain only binary labels 0 or 1")

    return X, y


def _validate_prediction_data(X, n_features):
    """Check new samples before prediction."""
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    if X.shape[0] == 0:
        raise ValueError("X must not be empty")

    if X.shape[1] != n_features:
        raise ValueError("X has a different number of features than the training data")

    return X


def _sigmoid(values):
    """Convert scores into probabilities between 0 and 1."""
    values = np.clip(values, -500, 500)
    return 1 / (1 + np.exp(-values))


class LogisticRegression:
    """Binary logistic regression trained with batch gradient descent."""

    def __init__(self, learning_rate=0.01, n_iterations=1000, reg_strength=0.01):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if n_iterations <= 0:
            raise ValueError("n_iterations must be positive")
        if reg_strength < 0:
            raise ValueError("reg_strength must not be negative")

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.reg_strength = reg_strength
        self.weights_ = None
        self.bias_ = None
        self.single_class_ = None

    def fit(self, X, y):
        X, y = _validate_training_data(X, y)
        n_samples, n_features = X.shape

        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        classes = np.unique(y)

        # A classifier can only return the observed label in this case.
        if len(classes) == 1:
            self.single_class_ = int(classes[0])
            return self

        self.single_class_ = None
        sample_weights = self._class_weights(y)

        for _ in range(self.n_iterations):
            probabilities = _sigmoid(X @ self.weights_ + self.bias_)
            errors = (probabilities - y) * sample_weights

            gradient_weights = (
                X.T @ errors / n_samples
                + self.reg_strength * self.weights_ / n_samples
            )
            gradient_bias = errors.mean()

            self.weights_ -= self.learning_rate * gradient_weights
            self.bias_ -= self.learning_rate * gradient_bias

        return self

    @staticmethod
    def _class_weights(y):
        """Give both classes the same total influence during training."""
        n_samples = len(y)
        count_0 = np.sum(y == 0)
        count_1 = np.sum(y == 1)

        weight_0 = n_samples / (2 * count_0)
        weight_1 = n_samples / (2 * count_1)
        return np.where(y == 0, weight_0, weight_1)

    def predict_proba(self, X):
        if self.weights_ is None:
            raise ValueError("Model must be fitted before prediction")

        X = _validate_prediction_data(X, len(self.weights_))

        if self.single_class_ is not None:
            probabilities = np.zeros((X.shape[0], 2))
            probabilities[:, self.single_class_] = 1.0
            return probabilities

        probability_class_1 = _sigmoid(X @ self.weights_ + self.bias_)
        return np.column_stack((1 - probability_class_1, probability_class_1))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SGDClassifier:
    """Binary logistic regression trained with mini-batch gradient descent."""

    def __init__(
        self,
        learning_rate=0.01,
        n_iterations=1000,
        batch_size=32,
        reg_strength=0.01,
        random_state=None,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if n_iterations <= 0:
            raise ValueError("n_iterations must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if reg_strength < 0:
            raise ValueError("reg_strength must not be negative")

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.reg_strength = reg_strength
        self.random_state = random_state
        self.weights_ = None
        self.bias_ = None
        self.single_class_ = None

    def fit(self, X, y):
        X, y = _validate_training_data(X, y)
        n_samples, n_features = X.shape

        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        classes = np.unique(y)

        if len(classes) == 1:
            self.single_class_ = int(classes[0])
            return self

        self.single_class_ = None
        sample_weights = LogisticRegression._class_weights(y)
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_iterations):
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            weights_shuffled = sample_weights[indices]

            for start in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[start:start + self.batch_size]
                y_batch = y_shuffled[start:start + self.batch_size]
                weight_batch = weights_shuffled[start:start + self.batch_size]

                probabilities = _sigmoid(X_batch @ self.weights_ + self.bias_)
                errors = (probabilities - y_batch) * weight_batch

                gradient_weights = (
                    X_batch.T @ errors / len(X_batch)
                    + self.reg_strength * self.weights_ / n_samples
                )
                gradient_bias = errors.mean()

                self.weights_ -= self.learning_rate * gradient_weights
                self.bias_ -= self.learning_rate * gradient_bias

        return self

    def predict_proba(self, X):
        if self.weights_ is None:
            raise ValueError("Model must be fitted before prediction")

        X = _validate_prediction_data(X, len(self.weights_))

        if self.single_class_ is not None:
            probabilities = np.zeros((X.shape[0], 2))
            probabilities[:, self.single_class_] = 1.0
            return probabilities

        probability_class_1 = _sigmoid(X @ self.weights_ + self.bias_)
        return np.column_stack((1 - probability_class_1, probability_class_1))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)