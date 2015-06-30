import activations
import costs

import numpy as np

import theano
import theano.tensor as T

import time

from trainers import SGDTrainer

class NeuralNetwork(object):
	def __init__(self, n_inputs, layers, batch_size=256, n_epochs=100, input_shape=None, cost=None, random_seed=23455, trainer=None):

		self.layer1_input = self.input = x = T.matrix('x')

		if input_shape != None:
			self.layer1_input = x.reshape(input_shape)

		self.cost = cost
		self.batch_size = batch_size
		self.trainer = trainer

		rng = np.random.RandomState(random_seed)

		self.layers = []

		#if not self.theano_rng:
		#self.theano_rng = T.shared_randomstreams.RandomStreams(rng.randint(2 ** 30)) # Put seed here

		for layer in layers:
			print layer[1]

			if not self.layers:
				l_params = {
					'input': self.layer1_input, 
					'n_in': n_inputs,
					'rng': rng,
					'model': self
				}
				l_params.update(layer[1])
				b = layer[0](**l_params) # Cascade inputs!
			else:
				self.prev_layer = prev_layer = self.layers[-1]
				l_params = {
					'input': prev_layer.output, 
					'n_in': prev_layer.n_out,
					'rng': rng,
					'model': self,
				}
				l_params.update(layer[1])
				b = layer[0](**l_params)

			self.layers.append(b)

		self.n_epochs = n_epochs
		
		self.predict_model = theano.function(
			inputs = [x],
			outputs = self.layers[-1].output,
		)

	def predict_output(self, x):
		return self.predict_model(x)

	def train(self, train_set, validation_set):
		if self.trainer is None:
			trainer = SGDTrainer(self)
		else:
			t_params = {
				'model': self
			}
			t_params.update(self.trainer[1])
			trainer = self.trainer[0](**t_params)

		trainer.train(train_set, validation_set)