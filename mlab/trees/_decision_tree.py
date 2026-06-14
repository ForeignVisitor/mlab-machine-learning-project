import numpy as np

# GenAI usage note:
# I used GenAI for general guidance and explanations.
# I reviewed, edited, and tested this implementation myself.


class _TreeNode:
    """Single node used internally by the decision tree."""

    def __init__(self, predicted_class, depth):
        self.predicted_class = predicted_class
        self.depth = depth
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None

    def is_leaf(self):
        """Return True if this node is a leaf."""
        return self.left is None and self.right is None


class DecisionTree:
    """
    Classification decision tree built from scratch using Gini impurity.
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=2,
        depth=None,
        random_state=None,
        max_features=None
    ):
        if depth is not None:
            max_depth = depth

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.max_features = max_features

        self.feature_importances_ = None
        self.max_depth_ = None
        self.tree_depth = None
        self.depth = None

        self._tree = None
        self._rng = None
        self._n_features_in = None
        self._importance_sums = None
        self._feature_means = None

    def _validate_params(self):
        """Validate constructor parameters."""
        if not isinstance(self.max_depth, (int, np.integer)):
            raise ValueError("max_depth must be an integer")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")

        if not isinstance(self.min_samples_split, (int, np.integer)):
            raise ValueError("min_samples_split must be an integer")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")

        if not isinstance(self.min_samples_leaf, (int, np.integer)):
            raise ValueError("min_samples_leaf must be an integer")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")

        if self.random_state is not None and not isinstance(
            self.random_state, (int, np.integer)
        ):
            raise ValueError("random_state must be an integer or None")

        if self.max_features is not None:
            if isinstance(self.max_features, str):
                if self.max_features != "sqrt":
                    raise ValueError("max_features string must be 'sqrt'")
            elif not isinstance(self.max_features, (int, np.integer)):
                raise ValueError("max_features must be an integer, 'sqrt', or None")
            elif self.max_features <= 0:
                raise ValueError("max_features must be positive")

    def _validate_data(self, X, y=None):
        """Validate input arrays."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.size == 0:
            raise ValueError("X must not be empty")

        if y is None:
            return X

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

    def _gini(self, y):
        """Compute Gini impurity for a label vector."""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return 1.0 - np.sum(probabilities ** 2)

    def _majority_class(self, y):
        """Return the most frequent class label."""
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _resolve_max_features(self, n_features):
        """Resolve how many features to consider at each split."""
        if self.max_features is None:
            return n_features
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        return min(n_features, int(self.max_features))

    def _best_split(self, X, y):
        """Find the best feature and threshold for a split."""
        n_samples, n_features = X.shape
        parent_impurity = self._gini(y)

        if parent_impurity == 0.0:
            return None, None, 0.0

        n_candidate_features = self._resolve_max_features(n_features)

        if n_candidate_features < n_features:
            feature_indices = self._rng.choice(
                n_features,
                size=n_candidate_features,
                replace=False
            )
        else:
            feature_indices = np.arange(n_features)

        best_feature = None
        best_threshold = None
        best_gain = 0.0

        for feature_index in feature_indices:
            feature_values = X[:, feature_index]
            unique_values = np.unique(feature_values)

            if unique_values.size <= 1:
                continue

            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                left_count = np.sum(left_mask)
                right_count = np.sum(right_mask)

                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                left_weight = y_left.size / n_samples
                right_weight = y_right.size / n_samples

                child_impurity = (
                    left_weight * self._gini(y_left) +
                    right_weight * self._gini(y_right)
                )
                gain = parent_impurity - child_impurity

                if gain > best_gain:
                    best_feature = feature_index
                    best_threshold = threshold
                    best_gain = gain

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth, total_samples):
        """Recursively build the tree."""
        predicted_class = self._majority_class(y)
        node = _TreeNode(predicted_class=predicted_class, depth=depth)

        if depth >= self.max_depth:
            return node
        if y.size < self.min_samples_split:
            return node
        if y.size < 2 * self.min_samples_leaf:
            return node
        if np.unique(y).size == 1:
            return node

        feature_index, threshold, gain = self._best_split(X, y)

        if feature_index is None or gain <= 0.0:
            return node

        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        node.feature_index = feature_index
        node.threshold = threshold

        self._importance_sums[feature_index] += gain * (y.size / total_samples)

        node.left = self._build_tree(
            X[left_mask],
            y[left_mask],
            depth + 1,
            total_samples
        )
        node.right = self._build_tree(
            X[right_mask],
            y[right_mask],
            depth + 1,
            total_samples
        )

        return node

    def _predict_one(self, x, node):
        """Predict the class for one sample by traversing the tree."""
        current = node

        while not current.is_leaf():
            if x[current.feature_index] <= current.threshold:
                current = current.left
            else:
                current = current.right

        return current.predicted_class

    def _compute_depth(self, node):
        """Compute tree depth from the root node."""
        if node is None or node.is_leaf():
            return 0
        return 1 + max(
            self._compute_depth(node.left),
            self._compute_depth(node.right)
        )

    def fit(self, X, y):
        """
        Build the decision tree from training data.
        """
        self._validate_params()
        X, y = self._validate_data(X, y)

        self._fit_imputer(X)
        X = self._impute_missing(X)

        self._rng = np.random.default_rng(self.random_state)
        self._n_features_in = X.shape[1]
        self._importance_sums = np.zeros(self._n_features_in, dtype=float)

        self._tree = self._build_tree(X, y, depth=0, total_samples=X.shape[0])

        total_importance = np.sum(self._importance_sums)
        if total_importance > 0:
            self.feature_importances_ = self._importance_sums / total_importance
        else:
            self.feature_importances_ = np.zeros(self._n_features_in, dtype=float)

        self.max_depth_ = self._compute_depth(self._tree)
        self.tree_depth = self.max_depth_
        self.depth = self.max_depth_

        return self

    def predict(self, X):
        """
        Predict class labels for the given input.
        """
        X = self._validate_data(X)

        if self._tree is None:
            raise ValueError("Model must be fitted before prediction")
        if X.shape[1] != self._n_features_in:
            raise ValueError("X must have the same number of features as during fit")

        X = self._impute_missing(X)

        predictions = [self._predict_one(sample, self._tree) for sample in X]
        return np.asarray(predictions)

    def get_depth(self):
        """Return the learned tree depth."""
        if self._tree is None:
            raise ValueError("Model must be fitted before querying depth")
        return self.max_depth_