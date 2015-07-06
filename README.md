# neuralmind
A python based Deep Learning framework using Theano

Types of feedforward layers:
* Hidden layer (fully-connected)
* Convolution layer (2d with max-pooling)
* Flatten layer
* Dropout layer

An example, training a feedforward neural network on the MNIST dataset.

```python
# Load MNIST
datasets = datasets.load_mnist("mnist.pkl.gz")

model = NeuralNetwork(
	n_inputs=28*28,
	layers = [
		(HiddenLayer,
		{
			'n_units': 150, 
			'non_linearity': activations.rectify
		}),
		(HiddenLayer,
		{
			'n_units': 80, 
			'non_linearity': activations.rectify
		}),
		(HiddenLayer,
		{
			'n_units': 10, 
			'non_linearity': activations.softmax
		})
	],
	trainer=(SGDTrainer,
		{
			'batch_size': 20,
			'learning_rate': 0.1,
			'n_epochs': 400,
			'global_L2_regularization': 0.0001,
			'dynamic_learning_rate': (ExponentialDecay, {'decay': 0.99}),
		}
	)
)

model.train(datasets[0], datasets[1])
```
