#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
All Function Implementation
"""

from proj1_helpers import compute_gradient, batch_iter


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        grad, loss = compute_gradient(y, tx, w)
        # update w by gradient
        w = w - gamma * grad
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
            bi=n_iter, ti=max_iters - 1, l=loss))

    return (w, loss)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent."""
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        # get a random minibatch of data
        for minibatch_y, minibatch_x in batch_iter(y, tx, 1):
            grad, loss = compute_gradient(minibatch_y, minibatch_x, w)
            w = w - gamma * grad
        print("Stochastic Gradient Descent({bi}/{ti}): loss={l}".format(
            bi=n_iter, ti=max_iters - 1, l=loss))

    return (w, loss)


def least_squares(y, tx):
    """Least squares regression using normal eqations."""
    raise NotImplementedError


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    raise NotImplementedError


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD."""
    raise NotImplementedError


def reg_logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD."""
    raise NotImplementedError


def main():
    """Main function."""
    return


if __name__ == '__main__':

    main()
