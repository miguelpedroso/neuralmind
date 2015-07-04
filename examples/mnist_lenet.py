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
import activations

from trainers import SGDTrainer
from trainers import ExponentialDecay

def load_data(dataset):

	print '... loading data'

	# Load the dataset
	f = gzip.open(dataset, 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()

	def shared_dataset(data_xy, borrow=True):
		data_x, data_y = data_xy
		shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
		shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)

		return shared_x, T.cast(shared_y, 'int32')

	test_set_x, test_set_y = shared_dataset(test_set)
	valid_set_x, valid_set_y = shared_dataset(valid_set)
	train_set_x, train_set_y = shared_dataset(train_set)

	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]

	return rval

# Load MNIST
datasets = load_data("mnist.pkl.gz")

model = NeuralNetwork(
	n_inputs=28*28,
	batch_size=20,
	input_shape=(20, 1, 28, 28),
	layers=[
		(ConvolutionLayer,
		{
			'image_shape': (1, 28, 28),
			'filter_shape': (1, 5, 5),
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
			'n_units': 80,
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
			'n_epochs': 100,
			'global_L2_regularization': 0.0001,
			'dynamic_learning_rate': (ExponentialDecay, {'decay': 0.99}),
		}
	)
		
)

model.train(datasets[0], datasets[1])