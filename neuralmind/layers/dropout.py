import numpy as np

import theano
import theano.tensor as T

import activations

class DropoutLayer(object):
	def __init__(self, input, n_in, probability=0.5, outdim=2, model=None, rng=1):
		self.input = input
		self.probability = probability

		if model.theano_rng is None:
			model.theano_rng = T.shared_randomstreams.RandomStreams(rng.randint(2 ** 30))

		#self.output = model.theano_rng.binomial(size=input.size, n=1, p=1-probability, dtype=theano.config.floatX) * input
		#print input.shape

		binomial_mask = model.theano_rng.binomial(n=1, p=1.-probability, size=input.shape, dtype=theano.config.floatX)
		self.output = binomial_mask * input
		self.output_pred = input
		#self.output = model.theano_rng.binomial(size=input.size, n=1, p=1-probability, dtype=theano.config.floatX) * input
		
		self.n_out = n_in
		self.output_params = {}
		self.params = []
		self.reg_terms = []