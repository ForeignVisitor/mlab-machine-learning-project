import numpy as np


def _validate_training_data(X, y, non_negative=False):
    """Check supervised-learning data before fitting."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

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

    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")

    if non_negative and np.any(X < 0):
        raise ValueError("X must contain only non-negative values")

    return X, y


def _validate_prediction_data(X, n_features, non_negative=False):
    """Check new samples before prediction."""
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    if X.shape[0] == 0:
        raise ValueError("X must not be empty")

    if X.shape[1] != n_features:
        raise ValueError("X has a different number of features than the training data")

    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")

    if non_negative and np.any(X < 0):
        raise ValueError("X must contain only non-negative values")

    return X


def _probabilities_from_log_scores(log_scores):
    """Convert log scores into probabilities without numerical overflow."""
    shifted_scores = log_scores - log_scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted_scores)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


class GaussianNaiveBayes:
    """Naive Bayes classifier for continuous features."""

    def __init__(self, epsilon=1e-9):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.epsilon = epsilon
        self.classes_ = None
        self.priors_ = None
        self.mean_ = None
        self.variance_ = None

    def fit(self, X, y):
        X, y = _validate_training_data(X, y)
        n_samples, n_features = X.shape

        self.classes_, class_counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)

        self.priors_ = class_counts / n_samples
        self.mean_ = np.zeros((n_classes, n_features))
        self.variance_ = np.zeros((n_classes, n_features))

        for index, class_label in enumerate(self.classes_):
            X_class = X[y == class_label]
            self.mean_[index] = X_class.mean(axis=0)

            # A small value prevents division by zero for constant features.
            self.variance_[index] = X_class.var(axis=0) + self.epsilon

        return self

    def _joint_log_likelihood(self, X):
        if self.classes_ is None:
            raise ValueError("Model must be fitted before prediction")

        X = _validate_prediction_data(X, self.mean_.shape[1])
        log_scores = np.zeros((X.shape[0], len(self.classes_)))

        for index in range(len(self.classes_)):
            mean = self.mean_[index]
            variance = self.variance_[index]

            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * variance) + (X - mean) ** 2 / variance,
                axis=1,
            )
            log_scores[:, index] = np.log(self.priors_[index]) + log_likelihood

        return log_scores

    def predict(self, X):
        log_scores = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(log_scores, axis=1)]

    def predict_proba(self, X):
        return _probabilities_from_log_scores(self._joint_log_likelihood(X))


class MultinomialNaiveBayes:
    """Naive Bayes classifier for non-negative count features."""

    def __init__(self, alpha=1.0):
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y):
        X, y = _validate_training_data(X, y, non_negative=True)
        n_samples, n_features = X.shape

        self.classes_, class_counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)

        self.class_log_prior_ = np.log(class_counts / n_samples)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for index, class_label in enumerate(self.classes_):
            feature_counts = X[y == class_label].sum(axis=0)
            smoothed_counts = feature_counts + self.alpha
            smoothed_total = smoothed_counts.sum()

            self.feature_log_prob_[index] = np.log(smoothed_counts / smoothed_total)

        return self

    def _joint_log_likelihood(self, X):
        if self.classes_ is None:
            raise ValueError("Model must be fitted before prediction")

        X = _validate_prediction_data(
            X,
            self.feature_log_prob_.shape[1],
            non_negative=True,
        )
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict(self, X):
        log_scores = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(log_scores, axis=1)]

    def predict_proba(self, X):
        return _probabilities_from_log_scores(self._joint_log_likelihood(X))