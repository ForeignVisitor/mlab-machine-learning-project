import numpy as np

# GenAI usage note:
# I used GenAI for general guidance and explanations.
# I reviewed, edited, and tested this implementation myself.


class _KernelMixin:
    """Shared kernel helpers for SVM models."""

    def _validate_kernel_params(self):
        """Validate common kernel-related parameters."""
        if self.kernel not in ("linear", "rbf", "poly"):
            raise ValueError("kernel must be 'linear', 'rbf', or 'poly'")

        if not isinstance(self.C, (int, float, np.integer, np.floating)):
            raise ValueError("C must be numeric")
        if self.C <= 0:
            raise ValueError("C must be positive")

        if not isinstance(self.degree, (int, np.integer)):
            raise ValueError("degree must be an integer")
        if self.degree < 1:
            raise ValueError("degree must be at least 1")

        if self.gamma is not None:
            if not isinstance(self.gamma, (int, float, np.integer, np.floating)):
                raise ValueError("gamma must be numeric or None")
            if self.gamma <= 0:
                raise ValueError("gamma must be positive when provided")

        if not isinstance(
            self.learning_rate, (int, float, np.integer, np.floating)
        ):
            raise ValueError("learning_rate must be numeric")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if not isinstance(self.n_iterations, (int, np.integer)):
            raise ValueError("n_iterations must be an integer")
        if self.n_iterations <= 0:
            raise ValueError("n_iterations must be positive")

        if self.random_state is not None and not isinstance(
            self.random_state, (int, np.integer)
        ):
            raise ValueError("random_state must be an integer or None")

    def _validate_X(self, X):
        """Validate feature input."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.size == 0:
            raise ValueError("X must not be empty")
        if np.any(np.isinf(X)):
            raise ValueError("X must not contain infinite values")

        return X

    def _validate_X_y(self, X, y):
        """Validate feature and target input."""
        X = self._validate_X(X)
        y = np.asarray(y)

        if y.ndim != 1:
            y = y.ravel()
        if y.size == 0:
            raise ValueError("y must not be empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

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

    def _resolve_gamma(self, n_features):
        """Resolve kernel gamma."""
        if self.gamma is None:
            return 1.0 / max(1, n_features)
        return float(self.gamma)

    def _kernel_matrix(self, X, Z):
        """Compute the kernel matrix between two datasets."""
        if self.kernel == "linear":
            return X @ Z.T

        if self.kernel == "poly":
            return (X @ Z.T + 1.0) ** self.degree

        diff = X[:, np.newaxis, :] - Z[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=2)
        return np.exp(-self.gamma_ * sq_dist)


class SVC(_KernelMixin):
    """
    Support Vector Machine classifier with kernel support.
    Uses a simple one-vs-rest training strategy.
    """

    def __init__(
        self,
        kernel="rbf",
        C=1.0,
        degree=3,
        gamma=None,
        learning_rate=0.1,
        n_iterations=200,
        random_state=None
    ):
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state

        self.classes_ = None
        self.dual_coef_ = None
        self.bias_ = None
        self.support_vectors_ = None

        self._X_train = None
        self._feature_means = None
        self._single_class = None

    def _fit_binary_classifier(self, K, y_binary, rng):
        """Fit one binary classifier in a one-vs-rest setup."""
        n_samples = y_binary.size
        alpha = np.zeros(n_samples, dtype=float)
        bias = 0.0

        positive_count = max(1, np.sum(y_binary == 1.0))
        negative_count = max(1, np.sum(y_binary == -1.0))
        pos_weight = n_samples / (2.0 * positive_count)
        neg_weight = n_samples / (2.0 * negative_count)

        current_lr = self.learning_rate

        for _ in range(self.n_iterations):
            for idx in rng.permutation(n_samples):
                decision = np.dot(alpha * y_binary, K[:, idx]) + bias
                margin = y_binary[idx] * decision

                if margin < 1.0:
                    sample_weight = pos_weight if y_binary[idx] > 0 else neg_weight
                    alpha[idx] = min(self.C, alpha[idx] + current_lr * sample_weight)
                    bias += current_lr * sample_weight * y_binary[idx]

            alpha *= (1.0 - min(current_lr * 0.01, 0.01))
            alpha = np.clip(alpha, 0.0, self.C)
            current_lr = max(current_lr * 0.98, 1e-3)

        dual = alpha * y_binary
        dual[np.abs(dual) < 1e-10] = 0.0

        return dual, bias

    def fit(self, X, y):
        """
        Train the SVM on the given data.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,) with class labels
        """
        self._validate_kernel_params()
        X, y = self._validate_X_y(X, y)

        self._fit_imputer(X)
        X = self._impute_missing(X)

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values after imputation")

        self.gamma_ = self._resolve_gamma(X.shape[1])
        self._X_train = X
        self.support_vectors_ = X

        self.classes_ = np.unique(y)
        self._single_class = None

        if self.classes_.size == 1:
            self._single_class = self.classes_[0]
            self.dual_coef_ = np.zeros((1, X.shape[0]), dtype=float)
            self.bias_ = np.zeros(1, dtype=float)
            return self

        K = self._kernel_matrix(X, X)
        rng = np.random.default_rng(self.random_state)

        n_classes = self.classes_.size
        self.dual_coef_ = np.zeros((n_classes, X.shape[0]), dtype=float)
        self.bias_ = np.zeros(n_classes, dtype=float)

        for class_index, class_label in enumerate(self.classes_):
            y_binary = np.where(y == class_label, 1.0, -1.0)
            dual, bias = self._fit_binary_classifier(K, y_binary, rng)
            self.dual_coef_[class_index] = dual
            self.bias_[class_index] = bias

        return self

    def decision_function(self, X):
        """
        Compute the decision function values.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of decision values
        """
        X = self._validate_X(X)

        if self._X_train is None:
            raise ValueError("Model must be fitted before prediction")
        if X.shape[1] != self._X_train.shape[1]:
            raise ValueError("X must have the same number of features as during fit")

        X = self._impute_missing(X)

        if self._single_class is not None:
            return np.zeros((X.shape[0], 1), dtype=float)

        K = self._kernel_matrix(X, self._X_train)
        return K @ self.dual_coef_.T + self.bias_

    def predict(self, X):
        """
        Predict class labels for the given input.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with predicted class labels
        """
        X = self._validate_X(X)

        if self._X_train is None:
            raise ValueError("Model must be fitted before prediction")
        if X.shape[1] != self._X_train.shape[1]:
            raise ValueError("X must have the same number of features as during fit")

        if self._single_class is not None:
            return np.full(X.shape[0], self._single_class)

        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]


class SVM(SVC):
    """Alias for SVC."""


class SVR(_KernelMixin):
    """
    Support Vector Regressor with kernel support.
    Uses a simple epsilon-insensitive online update rule.
    """

    def __init__(
        self,
        kernel="rbf",
        C=1.0,
        epsilon=0.1,
        degree=3,
        gamma=None,
        learning_rate=0.05,
        n_iterations=300,
        random_state=None
    ):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.degree = degree
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state

        self.dual_coef_ = None
        self.bias_ = None
        self.support_vectors_ = None

        self._X_train = None
        self._feature_means = None

    def _validate_params(self):
        """Validate SVR-specific parameters."""
        self._validate_kernel_params()

        if not isinstance(
            self.epsilon, (int, float, np.integer, np.floating)
        ):
            raise ValueError("epsilon must be numeric")
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")

    def fit(self, X, y):
        """
        Train the SVR on the given data.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,) with continuous target values
        """
        self._validate_params()
        X, y = self._validate_X_y(X, y)
        y = np.asarray(y, dtype=float)

        self._fit_imputer(X)
        X = self._impute_missing(X)

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values after imputation")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite values")

        self.gamma_ = self._resolve_gamma(X.shape[1])
        self._X_train = X
        self.support_vectors_ = X

        n_samples = X.shape[0]
        K = self._kernel_matrix(X, X)

        self.dual_coef_ = np.zeros(n_samples, dtype=float)
        self.bias_ = float(np.median(y))

        rng = np.random.default_rng(self.random_state)
        current_lr = self.learning_rate

        for _ in range(self.n_iterations):
            for idx in rng.permutation(n_samples):
                prediction = np.dot(K[idx], self.dual_coef_) + self.bias_
                error = y[idx] - prediction

                if error > self.epsilon:
                    update = current_lr * min(self.C, error - self.epsilon + 1.0)
                    self.dual_coef_[idx] = np.clip(
                        self.dual_coef_[idx] + update,
                        -self.C,
                        self.C
                    )
                    self.bias_ += current_lr
                elif error < -self.epsilon:
                    update = current_lr * min(self.C, -error - self.epsilon + 1.0)
                    self.dual_coef_[idx] = np.clip(
                        self.dual_coef_[idx] - update,
                        -self.C,
                        self.C
                    )
                    self.bias_ -= current_lr

            self.dual_coef_ *= (1.0 - min(current_lr * 0.01, 0.01))
            current_lr = max(current_lr * 0.98, 1e-3)

        return self

    def predict(self, X):
        """
        Predict target values for the given input.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,)
        """
        X = self._validate_X(X)

        if self._X_train is None:
            raise ValueError("Model must be fitted before prediction")
        if X.shape[1] != self._X_train.shape[1]:
            raise ValueError("X must have the same number of features as during fit")

        X = self._impute_missing(X)
        K = self._kernel_matrix(X, self._X_train)
        return K @ self.dual_coef_ + self.bias_