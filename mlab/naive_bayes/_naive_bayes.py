import numpy as np

# GenAI usage note:
# I used GenAI for general guidance and explanations.
# I reviewed, edited, and tested this implementation myself.


class GaussianNaiveBayes:
    """
    Naive Bayes classifier for continuous features using Gaussian distributions.
    """

    def __init__(self):
        self.classes_ = None
        self.priors_ = None
        self.mean_ = None
        self.variance_ = None
        self._epsilon = 1e-9

    def fit(self, X, y):
        """
        Train the model by computing class priors, means, and variances.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Validate input data before fitting.
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            y = y.ravel()
        if X.size == 0 or y.size == 0:
            raise ValueError("X and y must not be empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")

        n_samples, n_features = X.shape
        self.classes_, class_counts = np.unique(y, return_counts=True)

        n_classes = len(self.classes_)
        self.priors_ = class_counts / n_samples
        self.mean_ = np.zeros((n_classes, n_features), dtype=float)
        self.variance_ = np.zeros((n_classes, n_features), dtype=float)

        # Compute class-specific mean and variance for each feature.
        for idx, class_label in enumerate(self.classes_):
            X_class = X[y == class_label]
            self.mean_[idx] = np.mean(X_class, axis=0)
            self.variance_[idx] = np.var(X_class, axis=0) + self._epsilon

        return self

    def _joint_log_likelihood(self, X):
        """Compute log posterior scores up to an additive constant."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.size == 0:
            raise ValueError("X must not be empty")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        if self.classes_ is None:
            raise ValueError("Model must be fitted before prediction")
        if X.shape[1] != self.mean_.shape[1]:
            raise ValueError("X must have the same number of features as the fitted data")

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_likelihood = np.zeros((n_samples, n_classes), dtype=float)

        for idx in range(n_classes):
            mean = self.mean_[idx]
            variance = self.variance_[idx]
            log_prior = np.log(self.priors_[idx])

            log_prob = -0.5 * np.sum(
                np.log(2.0 * np.pi * variance) + ((X - mean) ** 2) / variance,
                axis=1
            )
            log_likelihood[:, idx] = log_prior + log_prob

        return log_likelihood

    def predict(self, X):
        """
        Predict class labels for the given input.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,)
        """
        log_likelihood = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(log_likelihood, axis=1)]

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples, n_classes)
        """
        log_likelihood = self._joint_log_likelihood(X)
        max_log = np.max(log_likelihood, axis=1, keepdims=True)
        stabilized = np.exp(log_likelihood - max_log)
        probabilities = stabilized / np.sum(stabilized, axis=1, keepdims=True)
        return probabilities


class MultinomialNaiveBayes:
    """
    Naive Bayes classifier for discrete or count-based features.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y):
        """
        Train the model by computing class priors and feature likelihoods.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Validate input data before fitting.
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            y = y.ravel()
        if X.size == 0 or y.size == 0:
            raise ValueError("X and y must not be empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        if np.any(X < 0):
            raise ValueError("X must contain only non-negative values for MultinomialNaiveBayes")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")

        n_samples, n_features = X.shape
        self.classes_, class_counts = np.unique(y, return_counts=True)

        n_classes = len(self.classes_)
        self.class_log_prior_ = np.log(class_counts / n_samples)
        self.feature_log_prob_ = np.zeros((n_classes, n_features), dtype=float)

        # Compute smoothed feature probabilities for each class.
        for idx, class_label in enumerate(self.classes_):
            X_class = X[y == class_label]
            feature_count = np.sum(X_class, axis=0)
            total_count = np.sum(feature_count)

            smoothed_count = feature_count + self.alpha
            smoothed_total = total_count + self.alpha * n_features

            self.feature_log_prob_[idx] = np.log(smoothed_count) - np.log(smoothed_total)

        return self

    def _joint_log_likelihood(self, X):
        """Compute log posterior scores for count-based features."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.size == 0:
            raise ValueError("X must not be empty")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        if np.any(X < 0):
            raise ValueError("X must contain only non-negative values for MultinomialNaiveBayes")
        if self.classes_ is None:
            raise ValueError("Model must be fitted before prediction")
        if X.shape[1] != self.feature_log_prob_.shape[1]:
            raise ValueError("X must have the same number of features as the fitted data")

        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict(self, X):
        """
        Predict class labels for the given input.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,)
        """
        log_likelihood = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(log_likelihood, axis=1)]

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples, n_classes)
        """
        log_likelihood = self._joint_log_likelihood(X)
        max_log = np.max(log_likelihood, axis=1, keepdims=True)
        stabilized = np.exp(log_likelihood - max_log)
        probabilities = stabilized / np.sum(stabilized, axis=1, keepdims=True)
        return probabilities