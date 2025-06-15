"""This file defines layer types that are commonly used for recurrent neural networks.
"""

import numpy as np
import torch

def affine_forward(x, w, b):
    out = x.reshape(x.shape[0], -1) @ w + b
    cache = (x, w, b)
    return out, cache



def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer."""
    x, w, b = cache
    N = x.shape[0]

    x_reshaped = x.reshape(N, -1)

    dx = dout @ w.T
    dx = dx.reshape(x.shape)

    dw = x_reshaped.T @ dout
    db = dout.sum(dim=0)

    return dx, dw, db



def rnn_step_forward(x, prev_h, Wx, Wh, b):
    next_h = torch.tanh(x @ Wx + prev_h @ Wh + b)
    cache = (next_h, x, prev_h, Wx, Wh)
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    next_h, x, prev_h, Wx, Wh = cache
    dz = dnext_h * (1 - next_h**2)

    dx = dz @ Wx.T
    dprev_h = dz @ Wh.T
    dWx = x.T @ dz
    dWh = prev_h.T @ dz
    db = dz.sum(dim=0)
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    N, T, D = x.shape
    H = h0.shape[1]
    h = torch.zeros((N, T, H), dtype=x.dtype, device=x.device)
    cache = []
    prev_h = h0
    for t in range(T):
        next_h, cache_t = rnn_step_forward(x[:, t], prev_h, Wx, Wh, b)
        h[:, t] = next_h
        prev_h = next_h
        cache.append(cache_t)
    return h, cache



def rnn_backward(dh, cache):
    N, T, H = dh.shape
    x_sample = cache[0][1]
    D = x_sample.shape[1]

    dx = torch.zeros((N, T, D), dtype=x_sample.dtype, device=x_sample.device)
    dWx = torch.zeros((D, H), dtype=x_sample.dtype, device=x_sample.device)
    dWh = torch.zeros((H, H), dtype=x_sample.dtype, device=x_sample.device)
    db = torch.zeros((H,), dtype=x_sample.dtype, device=x_sample.device)
    dprev_h = torch.zeros((N, H), dtype=x_sample.dtype, device=x_sample.device)

    for t in reversed(range(T)):
        dcurrent_h = dh[:, t] + dprev_h
        dx_t, dprev_h, dWx_t, dWh_t, db_t = rnn_step_backward(dcurrent_h, cache[t])
        dx[:, t] = dx_t
        dWx += dWx_t
        dWh += dWh_t
        db += db_t

    dh0 = dprev_h
    return dx, dh0, dWx, dWh, db





def word_embedding_forward(x, W):
    """Forward pass for word embeddings.
    
    We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out, cache = W[x], (x, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """Backward pass for word embeddings.
    
    We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D)
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, W = cache
    dW = torch.zeros_like(W)
    x_flat = x.reshape(-1)
    dout_flat = dout.reshape(-1, W.shape[1])
    dW.index_add_(0, x_flat, dout_flat)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """A numerically stable version of the logistic sigmoid function."""
    pos_mask = x >= 0
    neg_mask = x < 0
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Compute the activation vector and split to 4
    a = np.hsplit(x @ Wx + prev_h @ Wh + b, 4)

    # Compute gate vals
    i = sigmoid(a[0])
    f = sigmoid(a[1])
    o = sigmoid(a[2])
    g = np.tanh(a[3])

    # Next cell and hidden states
    next_c = f * prev_c + i * g
    next_h = o * np.tanh(next_c)

    cache = (prev_h, prev_c, next_c, i, f, o, g, x, Wx, Wh)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Retrieve elements from cache
    prev_h, prev_c, next_c, i, f, o, g, x, Wx, Wh = cache

    # Compute full partial derivative of dnext_c and dprev_c
    dnext_c += dnext_h * o * (1 - np.square(np.tanh(next_c)))
    dprev_c = dnext_c * f

    # Partial derivatives w.r.t. a
    da0 = dnext_c * g * i * (1 - i)
    da1 = dnext_c * prev_c * f * (1 - f)
    da2 = dnext_h * np.tanh(next_c) * o * (1 - o)
    da3 = dnext_c * i * (1 - np.square(g))
    da = np.hstack((da0, da1, da2, da3))

    # Derivatives w.r.t. primary values
    dx = da @ Wx.T
    dprev_h = da @ Wh.T
    dWx = x.T @ da
    dWh = prev_h.T @ da
    db = da.sum(axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """Forward pass for an LSTM over an entire sequence of data.
    
    We assume an input sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running the LSTM forward,
    we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell state is set to zero.
    Also note that the cell state is not returned; it is an internal variable to the LSTM and is not
    accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Init cell, hidden states and cache list
    c, hs, cache = np.zeros_like(h0), [h0], []

    for t in range(x.shape[1]):
        # Compute hidden + cell state at timestep t, append cache_t to list
        h, c, cache_t = lstm_step_forward(x[:, t], hs[-1], c, Wx, Wh, b)
        hs.append(h)
        cache.append(cache_t)
    
    # Stack along T, excluding h0
    h = np.stack(hs[1:], axis=1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """Backward pass for an LSTM over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Get the shape values and initialize gradients
    (N, T, H), (D, H4) = dh.shape, cache[0][8].shape
    dx = np.empty((N, T, D))
    dh0 = np.zeros((N, H))
    dc0 = np.zeros((N, H))
    dWx = np.zeros((D, H4))
    dWh = np.zeros((H, H4))
    db = np.zeros(H4)
    
    for t in range(T-1, -1, -1):
        # Run backward pass for t^th timestep and update the gradient matrices
        dx_t, dh0, dc0, dWx_t, dWh_t, db_t = lstm_step_backward(dh0 + dh[:, t], dc0, cache[t])
        dx[:, t] = dx_t
        dWx += dWx_t
        dWh += dWh_t
        db += db_t

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """Forward pass for a temporal affine layer.
    
    The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D) @ w
    out = out.reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Inputs:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Tuple of:
        - x: Input data of shape (N, T, D)
        - w: Weights of shape (D, M)
        - b: Biases of shape (M,)
        - out: Output from forward pass (not needed here)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, T, D)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b, _ = cache
    N, T, D = x.shape
    M = b.shape[0]

    dout_flat = dout.reshape(N * T, M)
    x_flat = x.reshape(N * T, D)

    dx = dout_flat @ w.T
    dx = dx.reshape(N, T, D)

    dw = x_flat.T @ dout_flat
    db = dout_flat.sum(dim=0)

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    Temporal softmax loss using PyTorch.

    Inputs:
    - x: Tensor of shape (N, T, V)
    - y: Tensor of shape (N, T) containing ground-truth indices
    - mask: Tensor of shape (N, T) containing 1 where loss should be applied

    Returns:
    - loss: scalar tensor
    - dx: gradient of loss with respect to x, same shape as x
    """
    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    # Compute probabilities
    probs = torch.softmax(x_flat, dim=1)

    # Compute log probabilities of correct class
    log_probs = torch.log(probs[torch.arange(N * T), y_flat] + 1e-12)
    
    # Apply mask and average loss
    loss = -torch.sum(mask_flat * log_probs) / N

    # Gradient
    dx_flat = probs.clone()
    dx_flat[torch.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]  # apply mask

    dx = dx_flat.reshape(N, T, V)

    if verbose:
        print("loss: ", loss.item())
        print("dx: ", dx.shape)

    return loss, dx


