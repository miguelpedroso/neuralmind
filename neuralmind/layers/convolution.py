import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

import activations

class ConvolutionLayer(object):
	def __init__(self, rng, input, filter_shape, n_kernels, image_shape=None, poolsize=(2, 2), non_linearity=activations.sigmoid, W=None, b=None, n_in=None, model=None):
		self.input = input

		if image_shape == None:
			image_shape = model.prev_layer.output_image_shape

		self.n_out = ((image_shape[1] - filter_shape[1] + 1) / poolsize[0]) * ((image_shape[2] - filter_shape[2] + 1) / poolsize[1]) * n_kernels  # Number of outputs

		self.output_image_shape = (n_kernels, (image_shape[1] - filter_shape[1] + 1) / poolsize[0], (image_shape[2] - filter_shape[2] + 1) / poolsize[1])  # Output image size
		self.output_params = {'image_shape': self.output_image_shape}


		# Set up the parameters
		image_shape = (model.batch_size, image_shape[0], image_shape[1], image_shape[2])
		filter_shape = (n_kernels, filter_shape[0], filter_shape[1], filter_shape[2])
		

		print image_shape
		print filter_shape

		# there are "num input feature maps * filter height * filter width"
		# inputs to each hidden unit
		fan_in = np.prod(filter_shape[1:])
		# each unit in the lower layer receives a gradient from:
		# "num output feature maps * filter height * filter width" /
		#   pooling size
		fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
		# initialize weights with random weights
		if W is None:
			W_bound = np.sqrt(6. / (fan_in + fan_out))
			self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX), borrow=True)

		if b is None:
			# the bias is a 1D tensor -- one bias per output feature map
			b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
			self.b = theano.shared(value=b_values, borrow=True)

		# Convolve input feature maps with filters
		conv_out = conv.conv2d(
			input=input,
			filters=self.W,
			filter_shape=filter_shape,
			image_shape=image_shape
		)

		# downsample each feature map individually, using maxpooling
		pooled_out = downsample.max_pool_2d(
			input=conv_out,
			ds=poolsize,
			ignore_border=True
		)

		# add the bias term. Since the bias is a vector (1D array), we first
		# reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
		# thus be broadcasted across mini-batches and feature map
		# width & height
		self.output = non_linearity(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

		self.params = [self.W, self.b]