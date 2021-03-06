import activations
import costs

import numpy as np

import theano
import theano.tensor as T

class ExponentialDecay(object):
	def __init__(self, trainer, learning_rate, decay=0.99):
		self.trainer = trainer
		self.update_function = theano.function(inputs=[], outputs=learning_rate, updates={learning_rate: learning_rate * decay})

	def get_update_inputs(self):
		return []