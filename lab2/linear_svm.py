from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    dW = np.zeros(W.shape)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin

                # TODO: compute gradient update for incorrect class j
                dW[:, j] += X[i]

                # TODO: compute gradient update for correct class y[i]
                dW[:, y[i]] -= X[i]

    # TODO: average loss
    loss /= num_train

    # TODO: average gradient
    dW /= num_train

    # TODO: regularize loss
    loss += reg * np.sum(W * W)

    # TODO: regularize gradient
    dW += 2 * reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros(W.shape)

    # TODO: compute scores
    scores = X.dot(W)

    # TODO: select correct class scores
    correct_class_scores = scores[np.arange(X.shape[0]), y][:, np.newaxis]

    # TODO: compute margins
    margins = np.maximum(0, scores - correct_class_scores + 1)
    margins[np.arange(X.shape[0]), y] = 0

    # TODO: compute loss
    loss = np.sum(margins) / X.shape[0]

    # TODO: regularize loss
    loss += reg * np.sum(W * W)

    # TODO: compute gradient mask
    mask = np.zeros_like(margins)
    mask[margins > 0] = 1
    mask[np.arange(X.shape[0]), y] = -np.sum(mask, axis=1)

    # TODO: compute gradient
    dW = X.T.dot(mask) / X.shape[0]

    # TODO: regularize gradient
    dW += 2 * reg * W

    return loss, dW
