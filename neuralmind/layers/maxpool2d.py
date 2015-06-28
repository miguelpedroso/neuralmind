import numpy as np

import theano
import theano.tensor as T

import activations

class MaxPool(object):
	def __init__(self, rng, input, n_in, n_units, non_linearity = activations.softmax, W = None, b = None):
		self.input = input
		self.n_out = n_out = n_units
		
		
		pooled_out = downsample.max_pool_2d(
			input=input,
			ds=poolsize,
			ignore_border=True
		)
		

		self.y_pred = T.argmax(self.output, axis = 1)

		self.params = [self.W, self.b]

	def classification_errors(self, y):
		if y.ndim != self.y_pred.ndim:
			raise TypeError(
				'y should have the same shape as self.output',
				('y', y.type, 'y_pred', self.y_pred.type)
			)

		if y.dtype.startswith('int'):
			return T.mean(T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()