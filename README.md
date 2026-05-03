# ML Lab Library

This repository contains my implementations for the weekly machine learning library tasks.

## Setup

### 1. Clone the repository

```bash
git clone https://git.fim.uni-passau.de/padas/26ss-mllab/p31/regression
cd regression/mlab
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

## Example usage

### Linear regression

```python
import numpy as np
from mlab.regression._linear import LinearRegressor, SGDRegression

X = np.array([,,, ], dtype=float)[3][4][5][6]
y = np.array(, dtype=float)[5][7][8][9]

model = LinearRegressor()
model.fit(X, y)
print(model.predict(X))

sgd_model = SGDRegression(learning_rate=0.01, n_iterations=1000, batch_size=2)
sgd_model.fit(X, y)
print(sgd_model.predict(X))
```

### Logistic regression

```python
import numpy as np
from mlab.regression._logistic import LogisticRegression, SGDClassifier

X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], dtype=float)

y = np.array(, dtype=int)[3]

model = LogisticRegression(learning_rate=0.1, n_iterations=2000)
model.fit(X, y)
print(model.predict(X))

sgd_model = SGDClassifier(learning_rate=0.1, n_iterations=2000, batch_size=2)
sgd_model.fit(X, y)
print(sgd_model.predict(X))
```

## Notes

- All implementations use NumPy arrays as input.
- The regression files include basic input validation.
- The source files contain comments and a GenAI usage note as required by the course.