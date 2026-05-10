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

    def _validate_X(self, X):
        """Validate input data before fitting or prediction."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.size == 0:
            raise ValueError("X must not be empty")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")
        if self.n_clusters > X.shape[0]:
            raise ValueError("n_clusters cannot be greater than number of samples")

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

    def fit(self, X):
        """
        Fit the K-Means model to the data.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            self
        """
        X = self._validate_X(X)
        rng = np.random.RandomState(self.random_state)

        centers = self._initialize_centroids(X, rng)

        for _ in range(self.max_iter):
            labels = self._assign_labels(X, centers)
            new_centers = centers.copy()

            # Update each centroid with the mean of its assigned points.
            for cluster_idx in range(self.n_clusters):
                cluster_points = X[labels == cluster_idx]

                if len(cluster_points) > 0:
                    new_centers[cluster_idx] = np.mean(cluster_points, axis=0)
                else:
                    # If a cluster becomes empty, reassign it to a random sample.
                    random_index = rng.randint(0, X.shape[0])
                    new_centers[cluster_idx] = X[random_index]

            # Stop if the centroids move less than the tolerance.
            center_shift = np.max(np.linalg.norm(new_centers - centers, axis=1))
            centers = new_centers

            if center_shift <= self.tol:
                break

        self.cluster_centers_ = centers
        self.labels_ = self._assign_labels(X, self.cluster_centers_)
        self.inertia_ = self._compute_inertia(X, self.cluster_centers_, self.labels_)

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
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before prediction")
        if X.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError("X must have the same number of features as the fitted data")

        return self._assign_labels(X, self.cluster_centers_)


class KMeansPlusPlus(KMeans):
    """
    K-Means clustering with K-Means++ initialization.
    """

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        super().__init__(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )

    def _initialize_centroids(self, X, rng):
        """Initialize centroids using the K-Means++ strategy."""
        n_samples = X.shape[0]
        centers = []

        # Choose the first center uniformly at random.
        first_idx = rng.randint(0, n_samples)
        centers.append(X[first_idx])

        # Choose each next center with probability proportional to D(x)^2.
        for _ in range(1, self.n_clusters):
            current_centers = np.array(centers)
            distances = self._compute_distances(X, current_centers)
            closest_dist_sq = np.min(distances, axis=1)

            total_distance = np.sum(closest_dist_sq)

            if total_distance == 0:
                next_idx = rng.randint(0, n_samples)
            else:
                probabilities = closest_dist_sq / total_distance
                next_idx = rng.choice(n_samples, p=probabilities)

            centers.append(X[next_idx])

        return np.array(centers, dtype=float)

    def fit(self, X):
        """Fit the K-Means++ model to the data."""
        return super().fit(X)

    def predict(self, X):
        """Predict cluster labels for new data points."""
        return super().predict(X)