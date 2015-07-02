import activations
import costs

import numpy as np

import theano
import theano.tensor as T

import time

class SGDTrainer(object):
	def __init__(self, model, batch_size=20, n_epochs=100, learning_rate=0.1, dynamic_learning_rate=None, cost=None, global_L1_regularization=None, global_L2_regularization=None,  random_seed=23455):

		self.model = model
		self.input = self.model.input
		self.layers = self.model.layers
		self.layers_pred =  self.model.layers_pred
		self.batch_size = batch_size
		self.initial_learning_rate = learning_rate
		self.n_epochs = n_epochs
		self.global_L1_regularization = global_L1_regularization
		self.global_L2_regularization = global_L2_regularization
		self.dynamic_learning_rate = dynamic_learning_rate
	

	def train(self, train_set, validation_set):	

		batch_size = self.batch_size
		print("building the model")
		this_validation_loss = 0

		# Segment the dataset into a training set and validation set
		train_set_x, train_set_y = train_set
		valid_set_x, valid_set_y = validation_set

		n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
		n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size

		index = T.lscalar()  # index to a [mini]batch
		x = self.input  # data, presented as rasterized images
		y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

		learning_rate = theano.shared(np.asarray(self.initial_learning_rate, dtype=theano.config.floatX))  # Dynamic learning rate

		# Create list of parameters
		self.params = []
		for layer in self.layers:
			self.params = self.params + layer.params

		cost = costs.negative_log_likelihood(self.layers[-1].output, y)

		#Ln regularization
		for layer in self.layers:
			for term in layer.reg_terms:
				if self.global_L1_regularization != None:
					cost = cost + (abs(term)).sum() * self.global_L1_regularization  # Add L1 term
				if self.global_L2_regularization != None:
					cost = cost + (term ** 2).sum() * self.global_L2_regularization  # Add L2 term

		# Gradients and upgrades
		grads = T.grad(cost, self.params)

		updates = [
			(param_i, param_i - learning_rate * grad_i)
			for param_i, grad_i in zip(self.params, grads)
		]

		# Create the theano functions to train the model 
		self.train_model = theano.function(
			inputs = [index],
			outputs = cost,
			updates = updates,
			givens = {
				x: train_set_x[index * batch_size: (index + 1) * batch_size],
				y: train_set_y[index * batch_size: (index + 1) * batch_size]
			}
		)

		self.validate_model = theano.function(
			inputs = [index],
			outputs = [self.layers_pred[-1].classification_errors(y), cost],
			givens = {
				x: valid_set_x[index * batch_size: (index + 1) * batch_size],
				y: valid_set_y[index * batch_size: (index + 1) * batch_size]
			}
		)

		if self.dynamic_learning_rate is not None:
			decay_learning_rate = theano.function(inputs=[], outputs=learning_rate, updates={learning_rate: learning_rate * 0.99})
		else:
			decay_learning_rate = None

		###############
		# TRAIN MODEL #
		###############

		print '... training the model'
		# early-stopping parameters
		patience = 5000  # look as this many examples regardless
		patience_increase = 2  # wait this much longer when a new best is
		# found
		improvement_threshold = 0.995  # a relative improvement of this much is
		# considered significant
		validation_frequency = min(n_train_batches, patience / 2)
		# go through this many
		# minibatche before checking the network
		# on the validation set; in this case we
		# check every epoch

		best_validation_loss = np.inf
		test_score = 0.
		start_time = time.clock()

		done_looping = False
		epoch = 0
		while (epoch < self.n_epochs) and (not done_looping):
			epoch = epoch + 1
			for minibatch_index in xrange(n_train_batches):

				minibatch_avg_cost = self.train_model(minibatch_index)
				# iteration number
				iter = (epoch - 1) * n_train_batches + minibatch_index

				if (iter + 1) % validation_frequency == 0:
					# compute zero-one loss on validation set
					#validation_losses, v_costs = [validate_model(i) for i in xrange(n_valid_batches)]
					#this_validation_loss = np.mean(validation_losses)
					#this_validation_cost = np.mean(v_costs)
					validation_losses = []
					v_costs = []
					for i in xrange(n_valid_batches):
						a, b = self.validate_model(i)
						validation_losses.append(a)
						v_costs.append(b)

					this_validation_loss = np.mean(validation_losses)
					this_validation_cost = np.mean(v_costs)

					if decay_learning_rate is None:
						new_lr = self.initial_learning_rate
					else:
						new_lr = decay_learning_rate()

					print('epoch %i, minibatch %i/%i, lr = %f, nll = %f, val loss %f, validation error %f %%' % (epoch, minibatch_index + 1, n_train_batches, new_lr, minibatch_avg_cost, this_validation_cost, this_validation_loss * 100.))
				
				# if we got the best validation score until now
				if this_validation_loss < best_validation_loss:
					#improve patience if loss improvement is good enough
					if this_validation_loss < best_validation_loss * improvement_threshold:
						patience = max(patience, iter * patience_increase)

					best_validation_loss = this_validation_loss
				
				#if patience <= iter:
				#	done_looping = True
				#	break

		end_time = time.clock()