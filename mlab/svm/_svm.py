# GenAI usage note:
# I used GenAI for general guidance and explanations.
# I reviewed, edited, and tested this implementation myself.

import numpy as np


class _KernelMixin:
    """Shared validation, imputation, and kernel methods."""

    def _validate_kernel_params(self):
        if self.kernel not in ("linear", "rbf", "poly"):
            raise ValueError("kernel must be 'linear', 'rbf', or 'poly'")

        if not isinstance(self.C, (int, float, np.integer, np.floating)):
            raise ValueError("C must be numeric")
        if self.C <= 0:
            raise ValueError("C must be positive")

        if not isinstance(self.degree, (int, np.integer)) or self.degree < 1:
            raise ValueError("degree must be an integer of at least 1")

        if self.gamma is not None:
            if not isinstance(self.gamma, (int, float, np.integer, np.floating)):
                raise ValueError("gamma must be numeric or None")
            if self.gamma <= 0:
                raise ValueError("gamma must be positive")

        if not isinstance(
            self.learning_rate,
            (int, float, np.integer, np.floating),
        ):
            raise ValueError("learning_rate must be numeric")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if not isinstance(self.n_iterations, (int, np.integer)):
            raise ValueError("n_iterations must be an integer")
        if self.n_iterations <= 0:
            raise ValueError("n_iterations must be positive")

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
        y = np.asarray(y)

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        elif y.ndim != 1:
            raise ValueError("y must be a 1D array or a column vector")

        if len(y) == 0:
            raise ValueError("y must not be empty")
        if len(y) != X.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        return X, y

    def _fit_imputer(self, X):
        # An all-missing feature is replaced with zero.
        valid_counts = np.sum(~np.isnan(X), axis=0)
        feature_sums = np.nansum(X, axis=0)

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

    def _resolve_gamma(self, n_features):
        if self.gamma is None:
            return 1.0 / n_features
        return float(self.gamma)

    def _kernel_matrix(self, X, Z):
        if self.kernel == "linear":
            return X @ Z.T

        if self.kernel == "poly":
            return (X @ Z.T + 1.0) ** self.degree

        differences = X[:, None, :] - Z[None, :, :]
        squared_distances = np.sum(differences ** 2, axis=2)
        return np.exp(-self.gamma_ * squared_distances)


class SVC(_KernelMixin):
    """Kernel SVM classifier using one-vs-rest training."""

    def __init__(
        self,
        kernel="rbf",
        C=1.0,
        degree=3,
        gamma=None,
        learning_rate=0.1,
        n_iterations=200,
        random_state=None,
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

    @staticmethod
    def _class_weights(y_binary):
        n_samples = len(y_binary)
        positive_count = np.sum(y_binary == 1.0)
        negative_count = np.sum(y_binary == -1.0)

        positive_weight = n_samples / (2 * positive_count)
        negative_weight = n_samples / (2 * negative_count)

        return np.where(y_binary == 1.0, positive_weight, negative_weight)

    def _fit_binary_classifier(self, kernel_matrix, y_binary, rng):
        n_samples = len(y_binary)
        alpha = np.zeros(n_samples)
        bias = 0.0
        sample_weights = self._class_weights(y_binary)
        learning_rate = self.learning_rate

        for _ in range(self.n_iterations):
            for index in rng.permutation(n_samples):
                score = np.dot(alpha * y_binary, kernel_matrix[:, index]) + bias
                margin = y_binary[index] * score

                if margin < 1.0:
                    update = learning_rate * sample_weights[index]
                    alpha[index] = min(self.C, alpha[index] + update)
                    bias += update * y_binary[index]

            # A small decay prevents coefficients from growing indefinitely.
            alpha *= 1.0 - min(learning_rate * 0.01, 0.01)
            alpha = np.clip(alpha, 0.0, self.C)
            learning_rate = max(learning_rate * 0.98, 1e-3)

        dual_coefficients = alpha * y_binary
        dual_coefficients[np.abs(dual_coefficients) < 1e-10] = 0.0
        return dual_coefficients, bias

    def fit(self, X, y):
        self._validate_kernel_params()
        X, y = self._validate_training_data(X, y)

        self._fit_imputer(X)
        X = self._impute_missing(X)

        self.gamma_ = self._resolve_gamma(X.shape[1])
        self._X_train = X
        self.classes_ = np.unique(y)
        self._single_class = None

        if len(self.classes_) == 1:
            self._single_class = self.classes_[0]
            self.dual_coef_ = np.zeros((1, len(X)))
            self.bias_ = np.zeros(1)
            self.support_vectors_ = X.copy()
            return self

        kernel_matrix = self._kernel_matrix(X, X)
        rng = np.random.default_rng(self.random_state)

        self.dual_coef_ = np.zeros((len(self.classes_), len(X)))
        self.bias_ = np.zeros(len(self.classes_))

        for class_index, class_label in enumerate(self.classes_):
            y_binary = np.where(y == class_label, 1.0, -1.0)
            dual, bias = self._fit_binary_classifier(kernel_matrix, y_binary, rng)

            self.dual_coef_[class_index] = dual
            self.bias_[class_index] = bias

        # Keep rows that affect at least one one-vs-rest classifier.
        support_mask = np.any(np.abs(self.dual_coef_) > 1e-10, axis=0)
        self.support_vectors_ = X[support_mask]

        return self

    def decision_function(self, X):
        if self._X_train is None:
            raise ValueError("Model must be fitted before prediction")

        X = self._validate_X(X)

        if X.shape[1] != self._X_train.shape[1]:
            raise ValueError("X must have the same number of features as during fit")

        X = self._impute_missing(X)

        if self._single_class is not None:
            return np.zeros((len(X), 1))

        kernel_matrix = self._kernel_matrix(X, self._X_train)
        return kernel_matrix @ self.dual_coef_.T + self.bias_

    def predict(self, X):
        scores = self.decision_function(X)

        if self._single_class is not None:
            return np.full(len(scores), self._single_class)

        return self.classes_[np.argmax(scores, axis=1)]


class SVM(SVC):
    """Alias for SVC."""


class SVR(_KernelMixin):
    """Kernel support vector regressor with epsilon-insensitive updates."""

    def __init__(
        self,
        kernel="rbf",
        C=1.0,
        epsilon=0.1,
        degree=3,
        gamma=None,
        learning_rate=0.05,
        n_iterations=300,
        random_state=None,
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
        self._validate_kernel_params()

        if not isinstance(self.epsilon, (int, float, np.integer, np.floating)):
            raise ValueError("epsilon must be numeric")
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")

    def fit(self, X, y):
        self._validate_params()
        X, y = self._validate_training_data(X, y)
        y = np.asarray(y, dtype=float)

        if not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite values")

        self._fit_imputer(X)
        X = self._impute_missing(X)

        self.gamma_ = self._resolve_gamma(X.shape[1])
        self._X_train = X

        n_samples = len(X)
        kernel_matrix = self._kernel_matrix(X, X)
        self.dual_coef_ = np.zeros(n_samples)
        self.bias_ = float(np.median(y))

        rng = np.random.default_rng(self.random_state)
        learning_rate = self.learning_rate

        for _ in range(self.n_iterations):
            for index in rng.permutation(n_samples):
                prediction = kernel_matrix[index] @ self.dual_coef_ + self.bias_
                error = y[index] - prediction

                if error > self.epsilon:
                    update = learning_rate * min(self.C, error - self.epsilon + 1.0)
                    self.dual_coef_[index] = np.clip(
                        self.dual_coef_[index] + update,
                        -self.C,
                        self.C,
                    )
                    self.bias_ += learning_rate

                elif error < -self.epsilon:
                    update = learning_rate * min(self.C, -error - self.epsilon + 1.0)
                    self.dual_coef_[index] = np.clip(
                        self.dual_coef_[index] - update,
                        -self.C,
                        self.C,
                    )
                    self.bias_ -= learning_rate

            self.dual_coef_ *= 1.0 - min(learning_rate * 0.01, 0.01)
            learning_rate = max(learning_rate * 0.98, 1e-3)

        support_mask = np.abs(self.dual_coef_) > 1e-10
        self.support_vectors_ = X[support_mask]

        return self

    def predict(self, X):
        if self._X_train is None:
            raise ValueError("Model must be fitted before prediction")

        X = self._validate_X(X)

        if X.shape[1] != self._X_train.shape[1]:
            raise ValueError("X must have the same number of features as during fit")

        X = self._impute_missing(X)
        kernel_matrix = self._kernel_matrix(X, self._X_train)

        return kernel_matrix @ self.dual_coef_ + self.bias_