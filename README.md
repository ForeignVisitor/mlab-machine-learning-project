# Machine Learning Lab Library

Souhail Karam — from-scratch implementation of core ML algorithms in NumPy, benchmarked against baselines on real datasets.

A small machine-learning library implemented with NumPy for the Machine Learning Lab course.

The core algorithms and evaluation methods are implemented manually. Third-party libraries may be used in the final project only for supporting tasks such as data loading, preprocessing, scaling, and visualisation.

## GenAI Usage

GenAI was used for general guidance, debugging support, and explanations during development.

All implementations were reviewed, edited, tested, and understood by the author. The algorithm logic, experiments, evaluation, and final project work are the author's responsibility.

## Requirements

- Python 3.10 or newer
- NumPy

Install NumPy if necessary:

```cmd
pip install numpy
```

Optional libraries for the final notebook:

```cmd
pip install pandas matplotlib scikit-learn
```

`scikit-learn` may be used for supporting tasks such as preprocessing, feature scaling, encoding, or train/test splitting. It must not replace the implemented machine-learning algorithms or evaluation methods.

## Project Structure

```text
mlab/
├── regression/
│   ├── _linear.py
│   └── _logistic.py
├── clustering/
│   └── _kmeans.py
├── naive_bayes/
│   └── _naive_bayes.py
├── trees/
│   └── _decision_tree.py
├── ensemble/
│   └── _random_forest.py
├── svm/
│   └── _svm.py
├── neural_networks/
│   ├── _mlp.py
│   └── _cnn.py
└── evaluation/
    └── _metrics.py
```

Local test files are used during development and are not part of the final library submission.

## Implemented Algorithms

| Module | Implementations |
|---|---|
| `regression/_linear.py` | `LinearRegressor`, `SGDRegression` |
| `regression/_logistic.py` | `LogisticRegression`, `SGDClassifier` |
| `clustering/_kmeans.py` | `KMeans`, `KMeansPlusPlus` |
| `naive_bayes/_naive_bayes.py` | `GaussianNaiveBayes`, `MultinomialNaiveBayes` |
| `trees/_decision_tree.py` | `DecisionTree` |
| `ensemble/_random_forest.py` | `RandomForest` |
| `svm/_svm.py` | `SVC`, `SVM`, `SVR` |
| `neural_networks/_mlp.py` | `MLPRegressor` and reusable neural-network layers |
| `neural_networks/_cnn.py` | `CNNClassifier` and reusable CNN layers |
| `evaluation/_metrics.py` | Classification, regression, and clustering evaluation metrics |

## Evaluation Metrics

| Problem type | Metrics |
|---|---|
| Classification | Accuracy, confusion matrix, precision, recall, F1-score |
| Regression | MAE, MSE, RMSE, \(R^2\) |
| Clustering | Inertia, silhouette score |

The evaluation metrics are implemented in:

```text
mlab/evaluation/_metrics.py
```

## Usage Examples

### Linear Regression

```python
import numpy as np

from mlab.regression._linear import LinearRegressor
from mlab.evaluation._metrics import (
    mean_squared_error,
    r2_score,
)

X_train = np.array([
    [1.0],
    [2.0],
    [3.0],
])
y_train = np.array([3.0, 5.0, 7.0])

X_test = np.array([
    [4.0],
    [5.0],
])
y_test = np.array([9.0, 11.0])

model = LinearRegressor()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(predictions)
print(mean_squared_error(y_test, predictions))
print(r2_score(y_test, predictions))
```

### Logistic Regression

```python
import numpy as np

from mlab.regression._logistic import LogisticRegression
from mlab.evaluation._metrics import accuracy_score, precision_recall_f1

X_train = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [4.0, 4.0],
    [5.0, 4.0],
])
y_train = np.array()

model = LogisticRegression(
    learning_rate=0.1,
    n_iterations=1000,
)

model.fit(X_train, y_train)

predictions = model.predict(X_train)
accuracy = accuracy_score(y_train, predictions)
precision, recall, f1 = precision_recall_f1(
    y_train,
    predictions,
    average="macro",
)

print(accuracy)
print(precision, recall, f1)
```

### K-Means Clustering

```python
import numpy as np

from mlab.clustering._kmeans import KMeansPlusPlus
from mlab.evaluation._metrics import silhouette_score

X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [8.0, 8.0],
    [8.0, 9.0],
])

model = KMeansPlusPlus(
    n_clusters=2,
    random_state=42,
)

model.fit(X)

print(model.labels_)
print(model.cluster_centers_)
print(model.inertia_)
print(silhouette_score(X, model.labels_))
```

### Decision Tree

```python
import numpy as np

from mlab.trees._decision_tree import DecisionTree
from mlab.evaluation._metrics import accuracy_score

X_train = np.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [8.0, 9.0],
    [9.0, 8.0],
])
y_train = np.array()

model = DecisionTree(
    max_depth=3,
    random_state=42,
)

model.fit(X_train, y_train)

predictions = model.predict(X_train)

print(predictions)
print(accuracy_score(y_train, predictions))
```

### Random Forest

```python
import numpy as np

from mlab.ensemble._random_forest import RandomForest
from mlab.evaluation._metrics import accuracy_score

X_train = np.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [8.0, 9.0],
    [9.0, 8.0],
])
y_train = np.array()

model = RandomForest(
    n_estimators=20,
    max_depth=4,
    random_state=42,
)

model.fit(X_train, y_train)

predictions = model.predict(X_train)

print(predictions)
print(accuracy_score(y_train, predictions))
print(model.oob_score_)
```

## Testing

Each module was tested locally using Python `unittest`.

Run an individual test file:

```cmd
python -m unittest test_linear.py -v
```

Examples:

```cmd
python -m unittest test_logistic.py -v
python -m unittest test_kmeans.py -v
python -m unittest test_naive_bayes.py -v
python -m unittest test_decision_tree.py -v
python -m unittest test_random_forest.py -v
python -m unittest test_svm.py -v
python -m unittest test_mlp.py -v
python -m unittest test_cnn.py -v
python -m unittest test_metrics.py -v
```

A successful test run ends with:

```text
OK
```

## Final Project Rules

For the final Jupyter notebook:

- Use this library's implementations for the main machine-learning models.
- Use the evaluation functions in `mlab/evaluation/_metrics.py` for the main evaluation.
- NumPy is used for numerical computation.
- Matplotlib may be used for plots and visualisation.
- Pandas may be used for loading and preparing data.
- Scikit-learn may be used only for supporting work such as train/test splitting, preprocessing, encoding, or feature scaling when allowed.
- Do not replace the implemented algorithms or evaluation functions with scikit-learn estimators or metrics.

## Author

Souhail Karam  
M.Sc. Artificial Intelligence, University of Passau
