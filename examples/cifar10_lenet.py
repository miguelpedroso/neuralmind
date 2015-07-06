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
from layers import ConvolutionLayer
from layers import FlattenLayer
from layers import DropoutLayer
import activations

from trainers import SGDTrainer
from trainers import ExponentialDecay

import datasets

# Load MNIST
#datasets = datasets.load_cifar10("/home/miguel/deeplearning/datasets")
datasets = datasets.load_cifar10("/home/ubuntu/deeplearning/datasets")

model = NeuralNetwork(
	n_inputs=32*32*3,
	input_shape=(64, 3, 32, 32),
	layers = [
		(ConvolutionLayer,
		{
			'image_shape': (3, 32, 32),
			'filter_shape': (3, 5, 5),
			'n_kernels': 20,
			'non_linearity': activations.rectify
		}),
		(ConvolutionLayer,
		{
			'filter_shape': (20, 3, 3), # Change this to only (3, 3), the first arg is the number of kernels in the previous layer
			'n_kernels': 40,
			'non_linearity': activations.rectify
		}),
		(FlattenLayer, {}),
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
			'batch_size': 64,
			'learning_rate': 0.1,
			'n_epochs': 400,
			#'global_L2_regularization': 0.0001,
			'dynamic_learning_rate': (ExponentialDecay, {'decay': 0.99}),
		}
	)
)

model.train(datasets[0], datasets[1])