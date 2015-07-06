import activations
import costs

import numpy as np

import theano
import theano.tensor as T

import time

from layers import DropoutLayer

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
		self.layers_pred = []

		self.theano_rng = None

		prev_layer_class = None
		for layer in layers:
			print layer

			if not self.layers:
				l_params = {
					'input': self.layer1_input, 
					'n_in': n_inputs,
					'rng': rng,
					'model': self
				}
				l_params.update(layer[1])
				b = layer[0](**l_params) # Cascade inputs!
				self.layers_pred.append(b)
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

				if prev_layer_class == DropoutLayer:
					l_params = {
						'input': self.layers[-1].output_pred, 
						'n_in': prev_layer.n_out,
						'rng': rng,
						'model': self,
						'W': b.W,  # TODO: RESCALE THESE WAITS
						'b': b.b,  # Put this more dynamic!
						'W_rescale': 1. - self.layers[-1].probability#0.5  #Fix this
					}
					l_params.update(layer[1])
					c = layer[0](**l_params)
					self.layers_pred.append(c)
				else:
					self.layers_pred.append(b) #Except dropout

			self.layers.append(b)

			prev_layer_class = layer[0]


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