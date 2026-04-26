import numpy as np


class LinearRegressor:
    def __init__(self):
        self.weights_ = None
        self.bias_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            y = y.ravel()

        X_aug = np.c_[np.ones(X.shape[0]), X]
        theta = np.linalg.pinv(X_aug) @ y

        self.bias_ = theta[0]
        self.weights_ = theta[1:]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        return X @ self.weights_ + self.bias_


class SGDRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.weights_ = None
        self.bias_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            y = y.ravel()

        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features, dtype=float)
        self.bias_ = 0.0

        for _ in range(self.n_iterations):
            order = np.random.permutation(n_samples)
            X_epoch = X[order]
            y_epoch = y[order]

            for start in range(0, n_samples, self.batch_size):
                stop = start + self.batch_size
                X_batch = X_epoch[start:stop]
                y_batch = y_epoch[start:stop]

                predictions = X_batch @ self.weights_ + self.bias_
                errors = predictions - y_batch

                grad_w = (2.0 / len(X_batch)) * (X_batch.T @ errors)
                grad_b = (2.0 / len(X_batch)) * np.sum(errors)

                self.weights_ -= self.learning_rate * grad_w
                self.bias_ -= self.learning_rate * grad_b

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        return X @ self.weights_ + self.bias_