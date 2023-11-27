from builtins import range
import numpy as np
from random import shuffle
#from past.builtins import xrange

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
    n_type = W.shape[1]
    n_train = X.shape[0]

    f = np.dot(X, W)
    f_exp = np.exp(f)

    for i in range(n_train):
      loss = loss - np.log(f_exp[i][y[i]]/np.sum(f_exp[i]) * 1.0)
    loss = loss / n_train
    loss = loss + reg * np.sum(W ** 2)

    for i in range(n_train):
      for j in range(n_type):
        dW[:, j] = dW[:, j] + (f_exp[i][j] / np.sum(f_exp[i]) * 1.0) * X[i]
        if j == y[i]:
          dW[:, j] = dW[:, j]- X[i]
    dW = dW / n_train
    dW = dW + 2 * reg * W
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
    n_train = X.shape[0]

    f = np.dot(X, W)
    f_exp = np.exp(f)
    
    loss_i = - np.log(f_exp[range(n_train), y] / np.sum(f_exp, axis=1))
    loss = np.sum(loss_i)
    loss = loss / n_train
    loss = loss + reg * np.sum(W ** 2)

    temp = (f_exp.T / np.sum(f_exp, axis=1)).T
    temp[range(n_train), y] -= 1.0
    dW = np.dot(X.T, temp)
    dW = dW / n_train
    dW = dW + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
