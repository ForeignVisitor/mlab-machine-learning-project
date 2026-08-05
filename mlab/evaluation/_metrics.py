import numpy as np


def _validate_targets(y_true, y_pred):
    """Check that true and predicted targets have the same length."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1:
        y_true = y_true.ravel()
    if y_pred.ndim != 1:
        y_pred = y_pred.ravel()

    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    return y_true, y_pred


def accuracy_score(y_true, y_pred):
    """Return the proportion of correctly predicted labels."""
    y_true, y_pred = _validate_targets(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, labels=None):
    """Return a confusion matrix with rows=true labels and columns=predictions."""
    y_true, y_pred = _validate_targets(y_true, y_pred)

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    else:
        labels = np.asarray(labels)

    if len(labels) == 0:
        raise ValueError("labels must not be empty")

    label_to_index = {
        label: index
        for index, label in enumerate(labels)
    }
    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for true_label, predicted_label in zip(y_true, y_pred):
        if true_label not in label_to_index:
            raise ValueError("y_true contains a label not included in labels")
        if predicted_label not in label_to_index:
            raise ValueError("y_pred contains a label not included in labels")

        matrix[
            label_to_index[true_label],
            label_to_index[predicted_label],
        ] += 1

    return matrix


def precision_recall_f1(y_true, y_pred, average="macro"):
    """
    Return precision, recall, and F1 score.

    average can be "macro", "weighted", or None.
    """
    y_true, y_pred = _validate_targets(y_true, y_pred)

    if average not in ("macro", "weighted", None):
        raise ValueError("average must be 'macro', 'weighted', or None")

    labels = np.unique(np.concatenate((y_true, y_pred)))
    matrix = confusion_matrix(y_true, y_pred, labels)

    true_counts = matrix.sum(axis=1)
    predicted_counts = matrix.sum(axis=0)
    true_positives = np.diag(matrix)

    precision = np.divide(
        true_positives,
        predicted_counts,
        out=np.zeros(len(labels), dtype=float),
        where=predicted_counts > 0,
    )
    recall = np.divide(
        true_positives,
        true_counts,
        out=np.zeros(len(labels), dtype=float),
        where=true_counts > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(labels), dtype=float),
        where=(precision + recall) > 0,
    )

    if average is None:
        return precision, recall, f1

    if average == "macro":
        return float(precision.mean()), float(recall.mean()), float(f1.mean())

    weights = true_counts / true_counts.sum()
    return (
        float(np.sum(weights * precision)),
        float(np.sum(weights * recall)),
        float(np.sum(weights * f1)),
    )


def mean_absolute_error(y_true, y_pred):
    """Return mean absolute error."""
    y_true, y_pred = _validate_targets(y_true, y_pred)
    return float(np.mean(np.abs(y_true.astype(float) - y_pred.astype(float))))


def mean_squared_error(y_true, y_pred):
    """Return mean squared error."""
    y_true, y_pred = _validate_targets(y_true, y_pred)
    return float(np.mean((y_true.astype(float) - y_pred.astype(float)) ** 2))


def root_mean_squared_error(y_true, y_pred):
    """Return root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true, y_pred):
    """Return the coefficient of determination."""
    y_true, y_pred = _validate_targets(y_true, y_pred)
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)

    total_sum_squares = np.sum((y_true - y_true.mean()) ** 2)
    residual_sum_squares = np.sum((y_true - y_pred) ** 2)

    if total_sum_squares == 0:
        return 1.0 if residual_sum_squares == 0 else 0.0

    return float(1.0 - residual_sum_squares / total_sum_squares)


def clustering_inertia(X, centers, labels):
    """Return the sum of squared distances to assigned cluster centers."""
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)
    labels = np.asarray(labels)

    if X.ndim != 2 or centers.ndim != 2:
        raise ValueError("X and centers must be 2D arrays")
    if len(X) == 0:
        raise ValueError("X must not be empty")
    if len(labels) != len(X):
        raise ValueError("labels must have the same length as X")
    if X.shape[1] != centers.shape[1]:
        raise ValueError("X and centers must have the same number of features")
    if np.any(labels < 0) or np.any(labels >= len(centers)):
        raise ValueError("labels must refer to valid center indices")

    return float(np.sum((X - centers[labels]) ** 2))


def silhouette_score(X, labels):
    """
    Return the mean silhouette coefficient using Euclidean distance.

    This implementation is simple and suitable for small coursework datasets.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if len(X) == 0:
        raise ValueError("X must not be empty")
    if labels.ndim != 1:
        labels = labels.ravel()
    if len(X) != len(labels):
        raise ValueError("X and labels must have the same number of samples")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")

    unique_labels = np.unique(labels)

    if len(unique_labels) < 2 or len(unique_labels) >= len(X):
        raise ValueError("silhouette_score requires between 2 and n_samples - 1 clusters")

    distances = np.sqrt(
        np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    )
    scores = np.zeros(len(X))

    for index, label in enumerate(labels):
        same_cluster = labels == label
        same_cluster[index] = False

        if np.any(same_cluster):
            intra_cluster_distance = distances[index, same_cluster].mean()
        else:
            intra_cluster_distance = 0.0

        other_cluster_distances = []

        for other_label in unique_labels:
            if other_label != label:
                other_cluster_distances.append(
                    distances[index, labels == other_label].mean()
                )

        nearest_cluster_distance = min(other_cluster_distances)
        denominator = max(intra_cluster_distance, nearest_cluster_distance)

        if denominator > 0:
            scores[index] = (
                nearest_cluster_distance - intra_cluster_distance
            ) / denominator

    return float(scores.mean())