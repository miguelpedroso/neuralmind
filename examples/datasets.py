import os
import gzip
import cPickle
import numpy as np

import theano
import theano.tensor as T

def load_mnist(dataset):

	print '... loading data'

	# Load the dataset
	f = gzip.open(dataset, 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()

	def shared_dataset(data_xy, borrow=True):
		data_x, data_y = data_xy
		shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
		shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)

		return shared_x, T.cast(shared_y, 'int32')

	test_set_x, test_set_y = shared_dataset(test_set)
	valid_set_x, valid_set_y = shared_dataset(valid_set)
	train_set_x, train_set_y = shared_dataset(train_set)

	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]

	return rval


"""
def load_cifar10(dataset_dir):

	print '... loading data'

	# Load the train set
	f = open(os.path.join(dataset_dir, 'data_batch_1'), 'rb')
	cifar = cPickle.load(f)
	f.close()

	data = cifar['data']
	fulldata = []

	for row in data:
		row = row / 256.0
		fulldata.append(row)

	train_set = (fulldata, cifar['labels'])


	# Load the test set
	f = open(os.path.join(dataset_dir, 'test_batch'), 'rb')
	cifar = cPickle.load(f)
	f.close()

	data = cifar['data']
	testdata = []

	for row in data:
		row = row / 256.0
		testdata.append(row)

	valid_set = (testdata, cifar['labels'])



	def shared_dataset(data_xy, borrow=True):
		data_x, data_y = data_xy
		shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
		shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)

		return shared_x, T.cast(shared_y, 'int32')

	train_set_x, train_set_y = shared_dataset(train_set)
	valid_set_x, valid_set_y = shared_dataset(valid_set)

	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]

	return rval
"""

def load_cifar10(dataset_dir):

	print '... loading data'

	# Load the train set
	full_data = []
	full_labels = []

	for i in xrange(1, 6):
		print '    loading part %i' % i
		f = open(os.path.join(dataset_dir, 'data_batch_' + str(i)), 'rb')
		cifar = cPickle.load(f)
		f.close()

		data = cifar['data']
		labels = cifar['labels']

		for row in data:
			row = row / 256.0
			full_data.append(row)

		for row in labels:
			full_labels.append(row)

	train_set = (full_data, full_labels)


	# Load the test set
	f = open(os.path.join(dataset_dir, 'test_batch'), 'rb')
	cifar = cPickle.load(f)
	f.close()

	data = cifar['data']
	testdata = []

	for row in data:
		row = row / 256.0
		testdata.append(row)

	valid_set = (testdata, cifar['labels'])



	def shared_dataset(data_xy, borrow=True):
		data_x, data_y = data_xy
		shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
		shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)

		return shared_x, T.cast(shared_y, 'int32')

	train_set_x, train_set_y = shared_dataset(train_set)
	valid_set_x, valid_set_y = shared_dataset(valid_set)

	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]

	return rval