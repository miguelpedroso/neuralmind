import numpy as np

import theano
import theano.tensor as T

import activations

class HiddenLayer(object):
	def __init__(self, rng, input, n_in, n_units, non_linearity = activations.softmax, W = None, b = None, model=None):
		self.input = input
		self.n_out = n_out = n_units
		self.output_params = {}

		if W is None:
			W_values = np.asarray(
				rng.uniform(
					low=-np.sqrt(6. / (n_in + n_out)),
					high=np.sqrt(6. / (n_in + n_out)),
					size=(n_in, n_out)
				),
				dtype=theano.config.floatX
			)
			if non_linearity == theano.tensor.nnet.sigmoid:
				W_values *= 4

			W = theano.shared(value=W_values, name='W', borrow=True)

		if b is None:
			b_values = np.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)


		self.W = W
		self.b = b

		#self.W = theano.shared(np.zeros((n_in, n_out), dtype=theano.config.floatX), name = 'W', borrow = True)
		#self.b = theano.shared(np.zeros((n_out,), dtype=theano.config.floatX), name = 'b', borrow = True)	

		linear_output = T.dot(input, self.W) + self.b

		self.output = (
			lin_output if non_linearity is None
			else non_linearity(linear_output)
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