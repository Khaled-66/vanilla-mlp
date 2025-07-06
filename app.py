import numpy as np  

class MLP(object):
    def __init__(self, num_inputs=3, hidden_layers=[4,3,6,3], num_outputs=4):
        self.num_inputs=num_inputs
        self.hidden_layers=hidden_layers
        self.num_outputs=num_outputs

        layers=[num_inputs] + hidden_layers + [num_outputs]
        
        # create random connection weights for the layers
        weights = []
        for i in range(len(layers)-1):
            w = np.random.randn(layers[i], layers[i+1]) * 0.01
            weights.append(w)
        self.weights=weights

        # save derivatives per layer
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        # save activations per layer
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

    def forward_propagation(self, inputs):
        activations = inputs
           # save the activations for backpropogation
        self.activations[0] = activations
        for i, w in enumerate(self.weights):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)

            # save the activations for backpropogation
            self.activations[i + 1] = activations
        return activations
    
    
    def backward_propagation(self,error):
        for i in reversed(range(len(self.derivatives))):
            # get activation for previous layer
            activations = self.activations[i+1]

            # apply sigmoid derivative function
            delta = error * self._sigmoid_derivative(activations)

            # reshape delta as to have it as a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T

            # get activations for current layer
            current_activations = self.activations[i]

            # reshape activations as to have them as a 2d column matrix
            current_activations = current_activations.reshape(current_activations.shape[0],-1)

            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(current_activations, delta_re)

            # backpropogate the next error
            error = np.dot(delta, self.weights[i].T)


    def gradient_descent(self, learning_rate=1):
         for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def _mse(self, target, output):
        return np.average((target - output) ** 2)

if __name__ == "__main__":

    # create a Multilayer Perceptron
    mlp = MLP()

    # set random values for network's input
    inputs = np.random.rand(mlp.num_inputs)

    # perform forward propagation
    output = mlp.forward_propagation(inputs)

    print("Network activation: {}".format(output))