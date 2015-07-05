import sys
sys.path.append("../")
sys.path.append("../neuralmind")

import gzip
import cPickle
import numpy as np

import theano
import theano.tensor as T

from neuralmind import NeuralNetwork
from layers import HiddenLayer
import activations

import datasets

# Load MNIST
datasets = datasets.load_mnist("mnist.pkl.gz")

model = NeuralNetwork(
	n_inputs=28*28,
	layers = [
		(HiddenLayer,
		{
			'n_units': 10, 
			'non_linearity': activations.softmax
		})
	]
)

model.train(datasets[0], datasets[1])