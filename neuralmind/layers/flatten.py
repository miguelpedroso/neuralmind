import numpy as np

import theano
import theano.tensor as T

import activations

class FlattenLayer(object):
	def __init__(self, input, n_in, outdim=2, model=None, rng=1):
		self.input = input
		self.output = input.flatten(outdim)
		
		self.n_out = n_in
		self.output_params = {}
		self.params = []