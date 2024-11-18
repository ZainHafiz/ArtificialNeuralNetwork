import numpy as np
import nnfs

# Initialise a random seed, and make sure the dot product takes the same data type
nnfs.init()


class Layer:
    def __init__(self, n_inputs, n_neurons, weight_regulariser_l2=0, bias_regulariser_l2=0):
        # initialise random weights between -1 and 1.
        self.bias_regulariser_l2 = bias_regulariser_l2
        self.weight_regulariser_l2 = weight_regulariser_l2
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # The forward function calculates the output

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    # differentiating
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dweights += 2 * self.weights * self.weight_regulariser_l2
        self.dbiases += 2 * self.biases * self.bias_regulariser_l2
        self.dinputs = np.dot(dvalues, self.weights.T)
class ReLU:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        self.inputs = inputs

    def backward(self, dvalues):

        # Since we need to modify the original variable,
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


# Activation function, applied to output layer to return probabilities of each output
class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        bounded_outputs = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_values = np.exp(bounded_outputs)
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    # dvalues is the CE loss gradeint w.r.t. the inputs
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
                single_output = single_output.reshape(- 1, 1)
                jacobianMatrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
                # sample-wise gradient dotted with the loss function
                self.dinputs[index] = np.dot(jacobianMatrix, single_dvalues)
# Generic loss function
class Loss:
    # y is target values
    # output is predicted values from network
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        batch_loss = np.mean(sample_losses)
        return batch_loss

    def regularization_loss(self, layer):
        regularisation_loss = 0
        regularisation_loss += layer.weight_regulariser_l2 * np.sum(np.square(layer.weights))
        regularisation_loss += layer.bias_regulariser_l2 * np.sum(np.square(layer.biases))
        return regularisation_loss
# This type of Loss function takes the negative log of the predicted probability
class LossCategoricalCrossEntropy(Loss):
    def forward(self, yPred, yTrue):
        samples = len(yPred)
        # We clip to avoid taking the log of zero
        yPredClipped = np.clip(yPred, 1e-7, 1 - 1e-7)

        # Takes the probability predicted for the correct class for each batch item
        # accounts for whether yTrue is given in scalar or one-hot encoding form
        if len(yTrue.shape) == 1:
            correct_confidences = yPredClipped[range(samples), yTrue]
        elif len(yTrue.shape) == 2:
            correct_confidences = np.sum(yPredClipped * yTrue, axis=1)

        negativeloglikelihoods = -np.log(correct_confidences)
        return negativeloglikelihoods

    #dvalues is the softmax outputs
    def backward(self, dvalues, yTrue):
        samples = len(dvalues)
        labels = len(dvalues[0]) # number of predicted values
        if len(yTrue.shape) == 1:
            yTrue = np.eye(labels)[yTrue] # onehot encoding
        self.dinputs = -yTrue/dvalues # calc change in loss w.r.t inputs
        self.dinputs = self.dinputs / samples # this means when summing the gradients, the mean is calculated


# When backpropagating, either use the backward methods of the Softmax and Loss classes
# or used the combined class below, which is faster
class Softmax_LossCategoricalCrossEntropy(Loss):
    def __init__(self):
        self.softmax = Softmax()
        self.loss = LossCategoricalCrossEntropy()

    def forward(self, inputs, yTrue):
        self.softmax.forward(inputs)
        self.output = self.softmax.output
        batch_loss = self.loss.calculate(self.output, yTrue)
        return batch_loss

    # the change in the loss function with respect to z is
    # the predicted out of the softmax function minus the real output
    # dvalues is the softmax output
    def backward(self, dvalues, yTrue):
        samples = len(dvalues)
        if len(yTrue.shape) == 2:
            yTrue = np.argmax(yTrue, axis=1) # scaler encoding
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), yTrue] -= 1 # minus 1 from
        self.dinputs = self.dinputs/samples # mean loss


# Stochastic Gradient descent, with 1/t decaying learning rate
# This uses adaptive momentum which include RMS propagation
class OptimiserAdam:
    def __init__(self, learning_rate=0.001, decay=0.0,
                 beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        # initial learning rate
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate/(1+self.decay*self.iterations)

    def update_params(self, layer):
        # If layer does not contain momentum/cache arrays, create them
        # filled with zeros
        if not hasattr(layer, 'bias_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # update momentum with gradient
        # using RMS propagation equation
        layer.weight_momentums = layer.weight_momentums*self.beta_1 \
                                + (1-self.beta_1)*layer.dweights

        layer.bias_momentums = layer.bias_momentums*self.beta_1 \
                                + (1-self.beta_1)*layer.dbiases

        layer.weight_cache = layer.weight_cache * self.beta_2 \
                             + (1-self.beta_2) * layer.dweights**2
        layer.bias_cache = layer.bias_cache * self.beta_2 \
                           + (1 - self.beta_2) * layer.dbiases**2
        # get the corrected momentums/caches,
        # with the 1/(1-beta^steps) correction
        # this give a large starting momentum and cache
        # but then approaches 1
        # self.iterations is 0 at first pass
        weight_momentums_corrected = layer.weight_momentums / \
                                     (1 - (self.beta_1 ** (self.iterations+1)))
        bias_momentums_corrected = layer.bias_momentums / \
                                   (1 - (self.beta_1 ** (self.iterations+1)))
        weight_cache_corrected = layer.weight_cache / \
                                 (1 - (self.beta_2 ** (self.iterations+1)))
        bias_cache_corrected = layer.bias_cache / \
                                 (1 - (self.beta_2 ** (self.iterations+1)))

        weights_update = -self.current_learning_rate * weight_momentums_corrected/(np.sqrt(weight_cache_corrected) + self.epsilon)
        biases_update = -self.current_learning_rate * bias_momentums_corrected/(np.sqrt(bias_cache_corrected) + self.epsilon)

        layer.weights += weights_update
        layer.biases += biases_update

    def post_update_params(self):
        self.iterations += 1


# n_inputs is the number of inputs from the input layer (number of features).
# n_neurons can be whatever you want
layer1 = Layer(2, 64, 5e-4, 5e-4)
activation1 = ReLU()

layer2 = Layer(64, 3)
activation2 = Softmax()

lossFunction = LossCategoricalCrossEntropy()
lossSoftmax = Softmax_LossCategoricalCrossEntropy()

optimiser = OptimiserAdam(learning_rate=0.05, decay=1e-7)
