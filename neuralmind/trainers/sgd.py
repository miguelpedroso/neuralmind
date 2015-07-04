import sys
sys.path.append("../")

import activations
import costs

import numpy as np

import theano
import theano.tensor as T

import time

from trainer import TrainerBase

import utils

class SGDTrainer(TrainerBase):
	def __init__(self, model, batch_size=20, n_epochs=100, learning_rate=0.1, dynamic_learning_rate=None, cost=None, global_L1_regularization=None, global_L2_regularization=None, early_stopping=False, patience=10000, improvement_threshold=0.995, random_seed=23455):

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

		self.early_stopping = early_stopping
		self.patience = patience
		self.improvement_threshold = improvement_threshold
	

	def train(self, train_set, validation_set):

		batch_size = self.batch_size
		print("... building the model")
		this_validation_loss = 0.

		# Segment the dataset into a training set and validation set
		train_set_x, train_set_y = train_set
		valid_set_x, valid_set_y = validation_set

		n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
		n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size

		index = T.lscalar()  # Index to a [mini]batch
		x = self.input
		y = T.ivector('y')

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
			params = {
				'trainer': self, 
				'learning_rate': learning_rate  # theano tensor object
			}
			params.update(self.dynamic_learning_rate[1])
			learning_rate_updater = self.dynamic_learning_rate[0](**params)
			decay_learning_rate = learning_rate_updater.update_function
		else:
			decay_learning_rate = None

		###############
		# TRAIN MODEL #
		###############

		print '... training the model'
		# early-stopping parameters
		patience = self.patience  # Look as this many examples regardless
		patience_increase = 2  # Wait this much longer when a new best is found
		#improvement_threshold = self.improvement_threshold  # a relative improvement of this much is considered significant
		validation_frequency = min(n_train_batches, patience / 2) # Go through this many minibatches before checking the network on the validation set; in this case we check every epoch

		best_validation_loss = np.inf
		test_score = 0.
		start_time = time.clock()

		done_looping = False
		epoch = 0

		start_epoch_time = None

		print('\t\t Epoch \t|  Train loss  |  Valid loss  |  Valid Acc  |  Learn. rate  |  Duration')

		while (epoch < self.n_epochs) and (not done_looping):


			start_epoch_time = time.time()


			epoch = epoch + 1

			for minibatch_index in xrange(n_train_batches):

				minibatch_avg_cost = self.train_model(minibatch_index)
				
				iter = (epoch - 1) * n_train_batches + minibatch_index  # Iteration number

				if (iter + 1) % validation_frequency == 0:
					# compute zero-one loss on validation set
					validation_losses = []
					v_costs = []
					for i in xrange(n_valid_batches):
						a, b = self.validate_model(i)
						validation_losses.append(a)
						v_costs.append(b)

					this_validation_loss = np.mean(validation_losses)
					this_validation_cost = np.mean(v_costs)

					# Add stats
					self.train_loss_history.append(minibatch_avg_cost)
					self.validation_loss_history.append(this_validation_cost)
					self.validation_accuracy_history.append(this_validation_loss)

					# Decay learning rate if needed
					if decay_learning_rate is None:
						new_lr = self.initial_learning_rate
					else:
						new_lr = decay_learning_rate()

					if start_epoch_time is not None:
						end_epoch_time = time.time()
						elapsed_time = end_epoch_time - start_epoch_time

					#print('\t\t %i \t|  %.8f  |  %.8f  |   %.4f%%  |  %f' % (epoch, minibatch_avg_cost, this_validation_cost, (1. - this_validation_loss) * 100., new_lr))
					str_line = '\t\t %i \t|  ' % epoch
					if self.train_loss_history.index(min(self.train_loss_history)) == epoch - 1:
						str_line += utils.bcolors.OKBLUE + ('%.8f' % minibatch_avg_cost) + utils.bcolors.ENDC
					else:
						str_line += '%.8f' % minibatch_avg_cost

					str_line += '  |  '

					if self.validation_loss_history.index(min(self.validation_loss_history)) == epoch - 1:
						str_line += utils.bcolors.OKGREEN + ('%.8f' % this_validation_cost) + utils.bcolors.ENDC
					else:
						str_line += '%.8f' % this_validation_cost

					str_line += '  |  '

					if self.validation_accuracy_history.index(min(self.validation_accuracy_history)) == epoch - 1:
						str_line += utils.bcolors.WARNING + ('%.5f%%' % ((1. - this_validation_loss) * 100.)) + utils.bcolors.ENDC
					else:
						str_line += '%.5f%%' % ((1. - this_validation_loss) * 100.)
					
					
					str_line += '  |  %.9f  |  %.3fs ' % (new_lr, elapsed_time)

					print(str_line)

				if self.early_stopping:
					# If we got the best validation score until now
					if this_validation_loss < best_validation_loss:
						# Improve patience if loss improvement is good enough
						if this_validation_loss < best_validation_loss * self.improvement_threshold:
							patience = max(patience, iter * patience_increase)

						best_validation_loss = this_validation_loss
				
					if patience <= iter:
						done_looping = True
						break

		end_time = time.clock()