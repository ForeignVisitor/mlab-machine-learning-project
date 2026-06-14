import numpy as np

from mlab.tree_ensemble._decision_tree import DecisionTree

# GenAI usage note:
# I used GenAI for general guidance and explanations.
# I reviewed, edited, and tested this implementation myself.


class RandomForest:
    """
    Random forest classifier using DecisionTree as the base estimator.
    """

    def __init__(
        self,
        n_estimators=20,
        max_depth=5,
        depth=None,
        n_trees=None,
        num_trees=None,
        min_samples_split=2,
        min_samples_leaf=2,
        random_state=None
    ):
        if depth is not None:
            max_depth = depth
        if n_trees is not None:
            n_estimators = n_trees
        if num_trees is not None:
            n_estimators = num_trees

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.trees_ = []
        self.feature_importances_ = None
        self.oob_score_ = None

        self._classes = None
        self._n_features_in = None

    def _validate_params(self):
        """Validate constructor parameters."""
        if not isinstance(self.n_estimators, (int, np.integer)):
            raise ValueError("n_estimators must be an integer")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")

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

    def _balanced_bootstrap_indices(self, y, rng):
        """Draw a more balanced bootstrap sample across classes."""
        classes, counts = np.unique(y, return_counts=True)
        max_count = np.max(counts)

        sampled_indices = []
        for class_label in classes:
            class_indices = np.where(y == class_label)[0]
            class_sample = rng.choice(class_indices, size=max_count, replace=True)
            sampled_indices.append(class_sample)

        sampled_indices = np.concatenate(sampled_indices)
        rng.shuffle(sampled_indices)
        return sampled_indices

    def fit(self, X, y):
        """
        Build the random forest from training data using bootstrap sampling.
        """
        self._validate_params()
        X, y = self._validate_data(X, y)

        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        self._classes = np.unique(y)
        self._n_features_in = n_features
        self.trees_ = []

        class_to_index = {label: idx for idx, label in enumerate(self._classes)}
        oob_votes = np.zeros((n_samples, len(self._classes)), dtype=int)
        oob_counts = np.zeros(n_samples, dtype=int)

        for _ in range(self.n_estimators):
            bootstrap_indices = self._balanced_bootstrap_indices(y, rng)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            in_bag = np.zeros(n_samples, dtype=bool)
            in_bag[np.unique(bootstrap_indices)] = True
            oob_indices = np.where(~in_bag)[0]

            tree_seed = int(rng.integers(0, 1_000_000_000))
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=tree_seed,
                max_features="sqrt"
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees_.append(tree)

            if oob_indices.size > 0:
                oob_predictions = tree.predict(X[oob_indices])
                for sample_index, predicted_label in zip(oob_indices, oob_predictions):
                    oob_votes[sample_index, class_to_index[predicted_label]] += 1
                    oob_counts[sample_index] += 1

        tree_importances = np.array(
            [tree.feature_importances_ for tree in self.trees_],
            dtype=float
        )
        self.feature_importances_ = np.mean(tree_importances, axis=0)

        total_importance = np.sum(self.feature_importances_)
        if total_importance > 0:
            self.feature_importances_ = self.feature_importances_ / total_importance

        valid_oob = oob_counts > 0
        if np.any(valid_oob):
            oob_pred_indices = np.argmax(oob_votes[valid_oob], axis=1)
            oob_predictions = self._classes[oob_pred_indices]
            self.oob_score_ = float(np.mean(oob_predictions == y[valid_oob]))
        else:
            self.oob_score_ = None

        return self

    def predict(self, X):
        """
        Predict class labels using majority voting across all trees.
        """
        X = self._validate_data(X)

        if not self.trees_:
            raise ValueError("Model must be fitted before prediction")
        if X.shape[1] != self._n_features_in:
            raise ValueError("X must have the same number of features as during fit")

        all_predictions = np.array([tree.predict(X) for tree in self.trees_])

        majority_predictions = []
        for sample_predictions in all_predictions.T:
            values, counts = np.unique(sample_predictions, return_counts=True)
            majority_predictions.append(values[np.argmax(counts)])

        return np.asarray(majority_predictions)