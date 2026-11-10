import numpy as np


def mse(y, y_pred):
    """Mean squared error."""
    err = y - y_pred
    loss = 0.5 * np.mean(err**2)
    return loss


def linear_regression_mse_gradient(y, tx, w):
    """Linear regression gradient at w."""
    N = tx.shape[0]
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / N
    return grad


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using Gradient Descent (GD)."""
    w = initial_w
    for _ in range(max_iters):
        grad = linear_regression_mse_gradient(y, tx, w)
        w -= gamma * grad
    loss = mse(y, tx.dot(w))
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using Stochastic Gradient Descent (SGD)."""
    w = initial_w
    for _ in range(max_iters):
        for y_item, tx_item in zip(y, tx):
            err = y_item - tx_item.dot(w)
            grad = -tx_item.T * err
            w -= gamma * grad
    loss = mse(y, tx.dot(w))
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    N, D = tx.shape
    regularization = 2 * N * lambda_ * np.eye(D)
    A = tx.T.dot(tx) + regularization
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)  # solves Aw = b
    loss = mse(y, tx.dot(w))
    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations."""
    return ridge_regression(y, tx, lambda_=0)


def sigmoid(x):
    """Sigmoid function."""
    return 1.0 / (1 + np.exp(-x))


def nll(y, y_pred):
    """Negative log likelihood."""
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


def logistic_regression_nll_gradient(y, tx, w):
    """Logistic regression gradient at w."""
    N = tx.shape[0]
    err = y - sigmoid(tx.dot(w))
    grad = -tx.T.dot(err) / N
    return grad


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent."""
    w = initial_w
    for _ in range(max_iters):
        grad = logistic_regression_nll_gradient(y, tx, w)
        w -= gamma * grad
    loss = nll(y, sigmoid(tx.dot(w)))
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent."""
    w = initial_w
    for _ in range(max_iters):
        grad = logistic_regression_nll_gradient(y, tx, w) + 2 * lambda_ * w
        w -= gamma * grad
    loss = nll(y, sigmoid(tx.dot(w)))
    return w, loss