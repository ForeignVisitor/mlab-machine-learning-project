import numpy as np


class _TreeNode:
    """A node in the decision tree."""

    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.left is None and self.right is None


class DecisionTree:
    """Classification decision tree using Gini impurity."""

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
        max_features=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.max_features = max_features

        self.feature_importances_ = None
        self.max_depth_ = None
        self._tree = None
        self._rng = None
        self._n_features_in = None
        self._feature_means = None
        self._importance_sums = None

    def _validate_params(self):
        if not isinstance(self.max_depth, (int, np.integer)) or self.max_depth <= 0:
            raise ValueError("max_depth must be a positive integer")

        if (
            not isinstance(self.min_samples_split, (int, np.integer))
            or self.min_samples_split < 2
        ):
            raise ValueError("min_samples_split must be an integer of at least 2")

        if (
            not isinstance(self.min_samples_leaf, (int, np.integer))
            or self.min_samples_leaf < 1
        ):
            raise ValueError("min_samples_leaf must be a positive integer")

        if self.random_state is not None and not isinstance(
            self.random_state,
            (int, np.integer),
        ):
            raise ValueError("random_state must be an integer or None")

        if self.max_features is None:
            return

        if self.max_features == "sqrt":
            return

        if (
            not isinstance(self.max_features, (int, np.integer))
            or self.max_features <= 0
        ):
            raise ValueError("max_features must be a positive integer, 'sqrt', or None")

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
        if X.shape[0] != len(y):
            raise ValueError("X and y must have the same number of samples")

        return X, y

    def _fit_imputer(self, X):
        # Columns containing only NaN values use zero as their replacement.
        self._feature_means = np.nanmean(X, axis=0)
        self._feature_means = np.nan_to_num(self._feature_means, nan=0.0)

    def _impute_missing(self, X):
        X = X.copy()
        missing_rows, missing_columns = np.where(np.isnan(X))
        X[missing_rows, missing_columns] = self._feature_means[missing_columns]
        return X

    @staticmethod
    def _gini(y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)

    @staticmethod
    def _majority_class(y):
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

    def _resolve_max_features(self, n_features):
        if self.max_features is None:
            return n_features
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        return min(n_features, self.max_features)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        parent_impurity = self._gini(y)

        if parent_impurity == 0.0:
            return None, None, 0.0

        n_candidates = self._resolve_max_features(n_features)
        if n_candidates == n_features:
            feature_indices = np.arange(n_features)
        else:
            feature_indices = self._rng.choice(
                n_features,
                size=n_candidates,
                replace=False,
            )

        best_feature = None
        best_threshold = None
        best_gain = 0.0

        for feature_index in feature_indices:
            values = np.unique(X[:, feature_index])

            if len(values) < 2:
                continue

            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                n_left = left_mask.sum()
                n_right = right_mask.sum()

                if (
                    n_left < self.min_samples_leaf
                    or n_right < self.min_samples_leaf
                ):
                    continue

                impurity_after_split = (
                    n_left / n_samples * self._gini(y[left_mask])
                    + n_right / n_samples * self._gini(y[right_mask])
                )
                gain = parent_impurity - impurity_after_split

                if gain > best_gain:
                    best_feature = feature_index
                    best_threshold = threshold
                    best_gain = gain

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth, total_samples):
        node = _TreeNode(self._majority_class(y))

        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(y) < 2 * self.min_samples_leaf
            or len(np.unique(y)) == 1
        ):
            return node

        feature_index, threshold, gain = self._best_split(X, y)

        if feature_index is None:
            return node

        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        node.feature_index = feature_index
        node.threshold = threshold

        # Importance is the weighted impurity reduction from this split.
        self._importance_sums[feature_index] += gain * len(y) / total_samples

        node.left = self._build_tree(
            X[left_mask],
            y[left_mask],
            depth + 1,
            total_samples,
        )
        node.right = self._build_tree(
            X[right_mask],
            y[right_mask],
            depth + 1,
            total_samples,
        )

        return node

    @staticmethod
    def _predict_one(sample, node):
        while not node.is_leaf():
            if sample[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.predicted_class

    @staticmethod
    def _compute_depth(node):
        if node.is_leaf():
            return 0

        return 1 + max(
            DecisionTree._compute_depth(node.left),
            DecisionTree._compute_depth(node.right),
        )

    def fit(self, X, y):
        self._validate_params()
        X, y = self._validate_training_data(X, y)

        self._fit_imputer(X)
        X = self._impute_missing(X)

        self._rng = np.random.default_rng(self.random_state)
        self._n_features_in = X.shape[1]
        self._importance_sums = np.zeros(self._n_features_in)

        self._tree = self._build_tree(X, y, depth=0, total_samples=len(X))
        self.max_depth_ = self._compute_depth(self._tree)

        total_importance = self._importance_sums.sum()
        if total_importance > 0:
            self.feature_importances_ = self._importance_sums / total_importance
        else:
            self.feature_importances_ = np.zeros(self._n_features_in)

        return self

    def predict(self, X):
        if self._tree is None:
            raise ValueError("Model must be fitted before prediction")

        X = self._validate_X(X)

        if X.shape[1] != self._n_features_in:
            raise ValueError("X must have the same number of features as during fit")

        X = self._impute_missing(X)
        return np.asarray([self._predict_one(sample, self._tree) for sample in X])

    def get_depth(self):
        if self._tree is None:
            raise ValueError("Model must be fitted before querying depth")

        return self.max_depth_