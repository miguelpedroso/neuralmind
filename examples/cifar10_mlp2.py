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
#datasets = datasets.load_cifar10("/home/miguel/deeplearning/datasets")
datasets = datasets.load_cifar10("/home/ubuntu/deeplearning/datasets")

"""
model = NeuralNetwork(
	n_inputs=32*32*3,
	layers = [
		(HiddenLayer,
		{
			'n_units': 512, 
			'non_linearity': activations.rectify
		}),
		(HiddenLayer,
		{
			'n_units': 512, 
			'non_linearity': activations.rectify
		}),
		(HiddenLayer,
		{
			'n_units': 10, 
			'non_linearity': activations.softmax
		})
	],
	trainer=(SGDTrainer,
		{
			'batch_size': 20,
			'learning_rate': 0.1,
			'n_epochs': 400,
			'global_L2_regularization': 0.0001,
			'dynamic_learning_rate': (ExponentialDecay, {'decay': 0.99}),
		}
	)
)
"""

model = NeuralNetwork(
	n_inputs=32*32*3,
	layers = [
		(HiddenLayer,
		{
			'n_units': 1024, 
			'non_linearity': activations.rectify
		}),
		(DropoutLayer, {'probability': 0.5}),
		(HiddenLayer,
		{
			'n_units': 1024, 
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
			'batch_size': 512,
			'learning_rate': 0.1,
			'n_epochs': 400,
			#'global_L2_regularization': 0.0001,
			'dynamic_learning_rate': (ExponentialDecay, {'decay': 0.99}),
		}
	)
)

model.train(datasets[0], datasets[1])