from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
from linear_svm import *
from softmax import *
from past.builtins import xrange


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        verbose=False,
    ):
        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            # TODO: sample X_batch and y_batch
            inds = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[inds]
            y_batch = y[inds]

            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # TODO: update weights
            self.W -= learning_rate * grad

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])

        # TODO: implement predict
        y_pred = np.argmax(X @ self.W, axis=1)

        return y_pred

    def loss(self, X_batch, y_batch, reg):
        # TODO: implement loss function
        pass


class LinearSVM(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
