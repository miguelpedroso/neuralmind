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
from layers import DropoutLayer
import activations

from trainers import SGDTrainer
from trainers import ExponentialDecay

import datasets

# Load MNIST
datasets = datasets.load_mnist("mnist.pkl.gz")

model = NeuralNetwork(
	n_inputs=28*28,
	layers = [
		(DropoutLayer, {'probability': 0.2}),
		(HiddenLayer,
		{
			'n_units': 800, 
			'non_linearity': activations.rectify
		}),
		(DropoutLayer, {'probability': 0.5}),
		(HiddenLayer,
		{
			'n_units': 800, 
			'non_linearity': activations.rectify
		}),
		(DropoutLayer, {'probability': 0.5}),
		(HiddenLayer,
		{
			'n_units': 10, 
			'non_linearity': activations.softmax
		})
	],
	trainer=(SGDTrainer,
		{
			'batch_size': 100,
			'learning_rate': 0.1,
			'n_epochs': 400,
			#'global_L2_regularization': 0.0001,
			'dynamic_learning_rate': (ExponentialDecay, {'decay': 0.99}),
		}
	)
)

model.train(datasets[0], datasets[1])