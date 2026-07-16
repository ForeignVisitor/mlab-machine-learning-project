# GenAI usage note:
# I used GenAI for general guidance and explanations.
# I reviewed, edited, and tested this implementation myself.

import numpy as np


class KMeans:
    """K-Means clustering with random centroid initialization."""

    def __init__(
        self,
        n_clusters=3,
        max_iter=300,
        tol=1e-4,
        n_init=10,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _validate_params(self):
        if not isinstance(self.n_clusters, (int, np.integer)) or self.n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")

        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")

        if not isinstance(self.n_init, (int, np.integer)) or self.n_init <= 0:
            raise ValueError("n_init must be a positive integer")

        if not isinstance(self.tol, (int, float, np.integer, np.floating)):
            raise ValueError("tol must be a number")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")

        if self.random_state is not None and not isinstance(
            self.random_state, (int, np.integer)
        ):
            raise ValueError("random_state must be an integer or None")

    def _validate_X(self, X, fitting=False):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] == 0:
            raise ValueError("X must not be empty")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")

        if fitting and X.shape[0] < self.n_clusters:
            raise ValueError("n_clusters cannot be greater than number of samples")

        return X

    @staticmethod
    def _squared_distances(X, centers):
        """Return squared distance from every point to every center."""
        return np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)

    def _assign_labels(self, X, centers):
        return np.argmin(self._squared_distances(X, centers), axis=1)

    @staticmethod
    def _inertia(X, centers, labels):
        return float(np.sum((X - centers[labels]) ** 2))

    def _initialize_centroids(self, X, rng):
        indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        return X[indices].copy()

    def _run_single_kmeans(self, X, centers, rng):
        for _ in range(self.max_iter):
            labels = self._assign_labels(X, centers)
            new_centers = centers.copy()

            for cluster_index in range(self.n_clusters):
                cluster_points = X[labels == cluster_index]

                if len(cluster_points) > 0:
                    new_centers[cluster_index] = cluster_points.mean(axis=0)
                else:
                    # Empty clusters are reset to an existing data point.
                    new_centers[cluster_index] = X[rng.integers(X.shape[0])]

            shift = np.max(np.linalg.norm(new_centers - centers, axis=1))
            centers = new_centers

            if shift <= self.tol:
                break

        labels = self._assign_labels(X, centers)
        return centers, labels, self._inertia(X, centers, labels)

    def fit(self, X):
        self._validate_params()
        X = self._validate_X(X, fitting=True)
        rng = np.random.default_rng(self.random_state)

        best_centers = None
        best_labels = None
        best_inertia = np.inf

        # Different starting points reduce the chance of a poor local solution.
        for _ in range(self.n_init):
            centers = self._initialize_centroids(X, rng)
            centers, labels, inertia = self._run_single_kmeans(X, centers, rng)

            if inertia < best_inertia:
                best_centers = centers.copy()
                best_labels = labels.copy()
                best_inertia = inertia

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = float(best_inertia)
        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before prediction")

        X = self._validate_X(X)

        if X.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError("X must have the same number of features as the fitted data")

        return self._assign_labels(X, self.cluster_centers_)


class KMeansPlusPlus(KMeans):
    """K-Means clustering with K-Means++ centroid initialization."""

    def _initialize_centroids(self, X, rng):
        n_samples = X.shape[0]
        selected_indices = [rng.integers(n_samples)]

        for _ in range(1, self.n_clusters):
            centers = X[selected_indices]
            distances = self._squared_distances(X, centers)
            closest_distance_sq = distances.min(axis=1)
            total_distance = closest_distance_sq.sum()

            if total_distance == 0:
                available_indices = np.setdiff1d(
                    np.arange(n_samples),
                    selected_indices,
                )
                next_index = rng.choice(available_indices)
            else:
                probabilities = closest_distance_sq / total_distance
                next_index = rng.choice(n_samples, p=probabilities)

            selected_indices.append(next_index)

        return X[selected_indices].copy()