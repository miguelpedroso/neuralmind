import abc

class TrainerBase(object):
	__metaclass__ = abc.ABCMeta

	train_loss_history = []
	validation_loss_history = []
	validation_accuracy_history = []

	def __init__(self):
		return

	@abc.abstractmethod
	def train(self, train_set, validation_set):
		return