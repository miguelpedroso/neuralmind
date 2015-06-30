import numpy as np

import theano
import theano.tensor as T

import activations

class DropoutLayer(object):
	def __init__(self, input, n_in, probability=0.5, outdim=2, model=None, rng=1):
		self.input = input
		self.output = model.theano_rng.binomial(size=input.size, n=1, p=1-probability, dtype=theano.config.floatX) * input
		
		self.n_out = n_in
		self.output_params = {}
		self.params = []