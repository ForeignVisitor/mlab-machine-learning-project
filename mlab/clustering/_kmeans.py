import numpy as np

# GenAI usage note:
# I used GenAI for general guidance and explanations.
# I reviewed, edited, and tested this implementation myself.


class KMeans:
    """
    Standard K-Means clustering algorithm.
    """

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self._n_init = 10

    def _validate_params(self):
        """Validate constructor parameters before fitting."""
        if not isinstance(self.n_clusters, (int, np.integer)):
            raise ValueError("n_clusters must be an integer")
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")

        if not isinstance(self.max_iter, (int, np.integer)):
            raise ValueError("max_iter must be an integer")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")

        if not isinstance(self.tol, (int, float, np.integer, np.floating)):
            raise ValueError("tol must be a number")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")

        if self.random_state is not None and not isinstance(
            self.random_state, (int, np.integer)
        ):
            raise ValueError("random_state must be an integer or None")

    def _validate_X(self, X):
        """Validate input data before fitting or prediction."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.size == 0:
            raise ValueError("X must not be empty")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_clusters cannot be greater than number of samples")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")

        return X

    def _compute_distances(self, X, centers):
        """Compute squared Euclidean distances from each point to each center."""
        return np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)

    def _assign_labels(self, X, centers):
        """Assign each sample to the nearest cluster center."""
        distances = self._compute_distances(X, centers)
        return np.argmin(distances, axis=1)

    def _compute_inertia(self, X, centers, labels):
        """Compute inertia as the sum of squared distances to assigned centers."""
        return float(np.sum((X - centers[labels]) ** 2))

    def _initialize_centroids(self, X, rng):
        """Randomly choose initial centroids from the dataset."""
        indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        return X[indices].copy()

    def _run_single_kmeans(self, X, initial_centers, rng):
        """Run one K-Means optimization from given initial centers."""
        centers = initial_centers.copy()

        for _ in range(self.max_iter):
            labels = self._assign_labels(X, centers)
            new_centers = centers.copy()

            # Update every centroid using the mean of its assigned points.
            for cluster_idx in range(self.n_clusters):
                cluster_points = X[labels == cluster_idx]

                if len(cluster_points) > 0:
                    new_centers[cluster_idx] = np.mean(cluster_points, axis=0)
                else:
                    # Reinitialize empty clusters with a random data point.
                    random_index = rng.integers(0, X.shape[0])
                    new_centers[cluster_idx] = X[random_index]

            # Stop when centroid movement is below the tolerance.
            center_shift = np.max(np.linalg.norm(new_centers - centers, axis=1))
            centers = new_centers

            if center_shift <= self.tol:
                break

        labels = self._assign_labels(X, centers)
        inertia = self._compute_inertia(X, centers, labels)

        return centers, labels, inertia

    def fit(self, X):
        """
        Fit the K-Means model to the data.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            self
        """
        self._validate_params()
        X = self._validate_X(X)
        rng = np.random.default_rng(self.random_state)

        best_centers = None
        best_labels = None
        best_inertia = np.inf

        # Run multiple initializations and keep the best solution.
        for _ in range(self._n_init):
            initial_centers = self._initialize_centroids(X, rng)
            centers, labels, inertia = self._run_single_kmeans(X, initial_centers, rng)

            if inertia < best_inertia:
                best_centers = centers.copy()
                best_labels = labels.copy()
                best_inertia = inertia

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = float(best_inertia)

        return self

    def predict(self, X):
        """
        Predict cluster labels for new data points.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with cluster labels
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.size == 0:
            raise ValueError("X must not be empty")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before prediction")
        if X.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError("X must have the same number of features as the fitted data")

        return self._assign_labels(X, self.cluster_centers_)


class KMeansPlusPlus(KMeans):
    """
    K-Means clustering with K-Means++ initialization.
    """

    def _initialize_centroids(self, X, rng):
        """Initialize centroids using the K-Means++ strategy."""
        n_samples = X.shape[0]
        centers = []

        # Choose the first center uniformly at random.
        first_idx = rng.integers(0, n_samples)
        centers.append(X[first_idx].copy())

        # Choose later centers with probability proportional to D(x)^2.
        for _ in range(1, self.n_clusters):
            current_centers = np.array(centers, dtype=float)
            distances = self._compute_distances(X, current_centers)
            closest_dist_sq = np.min(distances, axis=1)

            total_distance = np.sum(closest_dist_sq)

            if total_distance == 0:
                next_idx = rng.integers(0, n_samples)
            else:
                probabilities = closest_dist_sq / total_distance
                next_idx = rng.choice(n_samples, p=probabilities)

            centers.append(X[next_idx].copy())

        return np.array(centers, dtype=float)