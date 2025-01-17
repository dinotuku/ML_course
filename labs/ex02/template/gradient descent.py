# -*- coding: utf-8 -*-
"""Gradient Descent"""

import numpy as np

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - np.dot(tx, w)
    loss = 1 / 2 * np.mean(e**2)
    grad = -1  / len(e) * np.dot(tx.T, e)
    return grad, loss


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        # update w by gradient
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws