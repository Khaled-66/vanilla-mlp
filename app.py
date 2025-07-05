import numpy as np  

class MLP(object):
    def __init__(self, num_inputs=3, hidden_layers=[4,3,6,3], num_outputs=4):
        self.num_inputs=num_inputs
        self.hidden_layers=hidden_layers
        self.num_outputs=num_outputs

        layers=[num_inputs] + hidden_layers + [num_outputs]
        weights = []
        for i in range(len(layers)-1):
            w = np.random.randn(layers[i], layers[i+1]) * 0.01
            weights.append(w)
        self.weights=weights

    def forward_propagation(self, inputs):
        activations = inputs
        for w in self.weights:
            z = np.dot(activations, w)
            activations = self.sigmoid(z)
        return activations
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))



if __name__ == "__main__":

    # create a Multilayer Perceptron
    mlp = MLP()

    # set random values for network's input
    inputs = np.random.rand(mlp.num_inputs)

    # perform forward propagation
    output = mlp.forward_propagation(inputs)

    print("Network activation: {}".format(output))