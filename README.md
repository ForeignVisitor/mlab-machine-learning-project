# ML Lab Library

This repository contains my implementations for the weekly machine learning lab tasks.

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITLAB_REPO_URL>
cd <YOUR_REPO_NAME>
```

### 2. Create and activate a virtual environment

On Windows CMD:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install numpy
```

## Repository structure

```text
mlab/
├── clustering/
│   └── _kmeans.py
└── regression/
    ├── _linear.py
    └── _logistic.py
```

## Implemented modules

### Linear regression

File: `mlab/regression/_linear.py`

Contains:
- `LinearRegressor`
- `SGDRegression`

### Logistic regression

File: `mlab/regression/_logistic.py`

Contains:
- `LogisticRegression`
- `SGDClassifier`

### K-Means clustering

File: `mlab/clustering/_kmeans.py`

Contains:
- `KMeans`
- `KMeansPlusPlus`

## Example usage

### K-Means

```python
import numpy as np
from mlab.clustering._kmeans import KMeans, KMeansPlusPlus

X = np.array([
    [1.0, 1.0],
    [1.5, 2.0],
    [3.0, 4.0],
    [5.0, 7.0],
    [3.5, 5.0],
    [4.5, 5.0],
    [3.5, 4.5]
], dtype=float)

model = KMeans(n_clusters=2, random_state=42)
model.fit(X)
print(model.cluster_centers_)
print(model.labels_)
print(model.inertia_)

model_pp = KMeansPlusPlus(n_clusters=2, random_state=42)
model_pp.fit(X)
print(model_pp.cluster_centers_)
print(model_pp.labels_)
print(model_pp.inertia_)
```


### Naive Bayes

File: `mlab/naive_bayes/_naive_bayes.py`

Contains:
- `GaussianNaiveBayes`
- `MultinomialNaiveBayes`

### Trees & Ensembles

Files:
- `mlab/tree_ensemble/_decision_tree.py`
- `mlab/tree_ensemble/_random_forest.py`

Contains:
- `DecisionTree`
- `RandomForest`

## Notes

- All implementations use NumPy arrays as input.
- The source files include basic validation, comments, and a GenAI usage note.
- K-Means inertia is stored after fitting and can be used to compare clusterings.

