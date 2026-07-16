# GenAI usage note:
# I used GenAI for general guidance and explanations.
# I reviewed, edited, and tested this implementation myself.

import numpy as np

from mlab.trees._decision_tree import DecisionTree


class RandomForest:
    """Random forest classifier built from decision trees."""

    def __init__(
        self,
        n_estimators=20,
        max_depth=5,
        depth=None,
        n_trees=None,
        num_trees=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
    ):
        # These aliases keep the class compatible with the assignment API.
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
        if (
            not isinstance(self.n_estimators, (int, np.integer))
            or self.n_estimators <= 0
        ):
            raise ValueError("n_estimators must be a positive integer")

        if (
            not isinstance(self.max_depth, (int, np.integer))
            or self.max_depth <= 0
        ):
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

    @staticmethod
    def _balanced_bootstrap_indices(y, rng):
        """Sample every class up to the size of the largest class."""
        classes, counts = np.unique(y, return_counts=True)
        largest_class_size = counts.max()

        samples = [
            rng.choice(
                np.flatnonzero(y == class_label),
                size=largest_class_size,
                replace=True,
            )
            for class_label in classes
        ]

        bootstrap_indices = np.concatenate(samples)
        rng.shuffle(bootstrap_indices)
        return bootstrap_indices

    def fit(self, X, y):
        self._validate_params()
        X, y = self._validate_training_data(X, y)

        rng = np.random.default_rng(self.random_state)
        n_samples, self._n_features_in = X.shape
        self._classes = np.unique(y)
        self.trees_ = []

        class_to_index = {
            class_label: index
            for index, class_label in enumerate(self._classes)
        }
        oob_votes = np.zeros((n_samples, len(self._classes)), dtype=int)
        oob_counts = np.zeros(n_samples, dtype=int)

        for _ in range(self.n_estimators):
            bootstrap_indices = self._balanced_bootstrap_indices(y, rng)
            in_bag = np.zeros(n_samples, dtype=bool)
            in_bag[np.unique(bootstrap_indices)] = True
            oob_indices = np.flatnonzero(~in_bag)

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features="sqrt",
                random_state=int(rng.integers(0, 1_000_000_000)),
            ).fit(X[bootstrap_indices], y[bootstrap_indices])

            self.trees_.append(tree)

            # A small bootstrap sample can occasionally contain every original row.
            if len(oob_indices) > 0:
                oob_predictions = tree.predict(X[oob_indices])

                for sample_index, predicted_label in zip(oob_indices, oob_predictions):
                    oob_votes[sample_index, class_to_index[predicted_label]] += 1
                    oob_counts[sample_index] += 1

        self._set_feature_importances()
        self._set_oob_score(y, oob_votes, oob_counts)
        return self

    def _set_feature_importances(self):
        importances = np.mean(
            [tree.feature_importances_ for tree in self.trees_],
            axis=0,
        )
        total_importance = importances.sum()

        if total_importance > 0:
            importances = importances / total_importance

        self.feature_importances_ = importances

    def _set_oob_score(self, y, oob_votes, oob_counts):
        valid_samples = oob_counts > 0

        if not np.any(valid_samples):
            self.oob_score_ = None
            return

        predicted_indices = np.argmax(oob_votes[valid_samples], axis=1)
        predictions = self._classes[predicted_indices]
        self.oob_score_ = float(np.mean(predictions == y[valid_samples]))

    def predict(self, X):
        if not self.trees_:
            raise ValueError("Model must be fitted before prediction")

        X = self._validate_X(X)

        if X.shape[1] != self._n_features_in:
            raise ValueError("X must have the same number of features as during fit")

        all_predictions = np.asarray([tree.predict(X) for tree in self.trees_])
        predictions = []

        for sample_predictions in all_predictions.T:
            labels, counts = np.unique(sample_predictions, return_counts=True)
            predictions.append(labels[np.argmax(counts)])

        return np.asarray(predictions)