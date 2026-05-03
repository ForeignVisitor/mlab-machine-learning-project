import numpy as np

# GenAI usage note:
# I used GenAI for general guidance and explanations.
# I reviewed, edited, and tested this implementation myself.


class LogisticRegression:
    """
    Logistic regression classifier for binary classification.
    Uses batch gradient descent to learn weights and bias.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights_ = None
        self.bias_ = None

    def _sigmoid(self, z):
        """Apply the sigmoid function element-wise."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Train the logistic regression model.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,) with binary labels 0 or 1
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            y = y.ravel()

        if X.size == 0 or y.size == 0:
            raise ValueError("X and y must not be empty")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must contain only binary labels 0 or 1")

        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features, dtype=float)
        self.bias_ = 0.0

        for _ in range(self.n_iterations):
            linear_output = X @ self.weights_ + self.bias_
            predictions = self._sigmoid(linear_output)

            errors = predictions - y
            grad_w = (1.0 / n_samples) * (X.T @ errors)
            grad_b = (1.0 / n_samples) * np.sum(errors)

            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for the input samples.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples, 2)
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.size == 0:
            raise ValueError("X must not be empty")
        if self.weights_ is None or self.bias_ is None:
            raise ValueError("Model must be fitted before prediction")

        probabilities_class_1 = self._sigmoid(X @ self.weights_ + self.bias_)
        probabilities_class_0 = 1 - probabilities_class_1

        return np.column_stack((probabilities_class_0, probabilities_class_1))

    def predict(self, X):
        """
        Predict binary class labels for the input samples.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with labels 0 or 1
        """
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= 0.5).astype(int)


class SGDClassifier:
    """
    Logistic regression classifier trained with stochastic gradient descent.
    Uses mini-batches during training.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.weights_ = None
        self.bias_ = None

    def _sigmoid(self, z):
        """Apply the sigmoid function element-wise."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Train the classifier using stochastic gradient descent.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,) with binary labels 0 or 1
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            y = y.ravel()

        if X.size == 0 or y.size == 0:
            raise ValueError("X and y must not be empty")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must contain only binary labels 0 or 1")

        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features, dtype=float)
        self.bias_ = 0.0

        for _ in range(self.n_iterations):
            order = np.random.permutation(n_samples)
            X_epoch = X[order]
            y_epoch = y[order]

            for start in range(0, n_samples, self.batch_size):
                stop = start + self.batch_size
                X_batch = X_epoch[start:stop]
                y_batch = y_epoch[start:stop]

                linear_output = X_batch @ self.weights_ + self.bias_
                predictions = self._sigmoid(linear_output)

                errors = predictions - y_batch
                grad_w = (1.0 / len(X_batch)) * (X_batch.T @ errors)
                grad_b = (1.0 / len(X_batch)) * np.sum(errors)

                self.weights_ -= self.learning_rate * grad_w
                self.bias_ -= self.learning_rate * grad_b

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for the input samples.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples, 2)
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.size == 0:
            raise ValueError("X must not be empty")
        if self.weights_ is None or self.bias_ is None:
            raise ValueError("Model must be fitted before prediction")

        probabilities_class_1 = self._sigmoid(X @ self.weights_ + self.bias_)
        probabilities_class_0 = 1 - probabilities_class_1

        return np.column_stack((probabilities_class_0, probabilities_class_1))

    def predict(self, X):
        """
        Predict binary class labels for the input samples.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with labels 0 or 1
        """
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= 0.5).astype(int)