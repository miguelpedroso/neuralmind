import theano
import theano.tensor as T

def negative_log_likelihood(predicted_y, y):
	#return -T.mean(T.log(self.output)[T.arange(y.shape[0]), y])
	return -T.mean(T.log(predicted_y)[T.arange(y.shape[0]), y])