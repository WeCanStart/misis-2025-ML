from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    results = X @ W

    for i in range(results.shape[0]):
        loss += -results[i][y[i]] + np.log(np.sum(np.exp(results[i])))

    # left part is -(B * X).T, B is corrects like (1 0 0)
    #                                             (0 1 1)
    # plus 
    loss /= results.shape[0]
    loss += reg * np.sum(W * W)

    B = np.zeros((results.shape[0], results.shape[1]))
    for i in range(results.shape[0]):
        B[i][y[i]] = 1
    dW = X.T @ (np.exp(results) / np.exp(results).sum(axis=1, keepdims=True) - B)
    dW /= results.shape[0]
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    results = X @ W
    results -= np.max(results, axis=1, keepdims=True)

    exp_results = np.exp(results)
    softmax_probs = exp_results / np.sum(exp_results, axis=1, keepdims=True)

    correct_class_probs = softmax_probs[np.arange(X.shape[0]), y]
    loss = -np.mean(np.log(correct_class_probs)) + reg * np.sum(W * W)

    B = np.zeros_like(softmax_probs)  
    B[np.arange(X.shape[0]), y] = 1

    dW = X.T @ (softmax_probs - B) / X.shape[0]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
