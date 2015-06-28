# sigmoid
from theano.tensor.nnet import sigmoid

# softmax (row-wise)
from theano.tensor.nnet import softmax

# tanh
from theano.tensor import tanh


# rectify
def rectify(x):
    return (x + abs(x)) / 2.0


# identity
def linear(x):
    return x

identity = linear