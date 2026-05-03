# Linear Regression Implementation

This file contains my implementation for the linear regression task.

The code is written in `mlab/regression/_linear.py` and includes two classes:

- `LinearRegressor`
- `SGDRegression`

## What is implemented

### 1. LinearRegressor

This class fits a linear regression model using the closed-form solution.

During training, I first add a column of ones to the input data so the model can learn the bias term.  
Then I compute the parameter vector with the pseudo-inverse.

After fitting:
- `weights_` stores the learned coefficients
- `bias_` stores the intercept

### 2. SGDRegression

This class fits a linear regression model using stochastic gradient descent.

The constructor allows changing:
- `learning_rate`
- `n_iterations`
- `batch_size`

During training, the data is shuffled at each iteration and processed in mini-batches.  
For every batch, the gradients for the weights and bias are computed and then updated.

After fitting:
- `weights_` stores the learned coefficients
- `bias_` stores the intercept

## Input checks

I added a few validation checks before training and prediction:

- `X` must be a 2D array
- `y` is converted to a 1D array if needed
- empty input is rejected
- `X` and `y` must have the same number of samples
- prediction is not allowed before fitting

These checks help avoid common edge-case errors.