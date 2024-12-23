import numpy as np

class MLP:
    
    parameters = {} #Définition d'un attribut de classe pour garder
    cost = []

    def __init__(self, number_of_neurons_in_each_layer: list):
        self.number_of_neurons_in_each_layer = number_of_neurons_in_each_layer

        

    def initialize(self):
        """
        Initialize the parameters of the model.
        Args:
            n0 (int): Number of input features.
            n1 (int): Number of neurons in the first layer.
        Returns:
            
        """
        for c in range(1, len(self.number_of_neurons_in_each_layer)):
            MLP.parameters[f"W{c}"] = np.random.randn(self.number_of_neurons_in_each_layer[c -1], self.number_of_neurons_in_each_layer[c])
            MLP.parameters[f"b{c}"] = np.random.randn(1, self.number_of_neurons_in_each_layer[c]) 

        return MLP.parameters
    
    def predictions(self, X, parameters):

        activations = {"A0": X}
        for c in range(1, len(self.number_of_neurons_in_each_layer)):
            z = np.dot(activations[f"A{c - 1}"], parameters[f"W{c}"]) + parameters[f"b{c}"]
            activations[f"A{c}"] = 1 / (1 + np.exp(-z))

        return activations
    
    def compute_cost(self, predictions, y):
        """
        Compute the cost of the model.
        Args:
            predictions (array): predicted values.
            y (ndarray): Shape(m,) target values.
        
        Returns:
            cost (scalar): The cost of the model.
        """
        m = len(y)
        c = (1/m) * np.sum(-y * np.log(predictions) - (1 - y) * np.log(1 - predictions))
        return c
    
    def compute_gradient(self,y, activations, parameters):
        """
        Compute the gradient of the cost function with respect to the parameters.
        Args:
            predictions (array): predicted values.
            X (ndarray): Shape(m,) Input to the model.
            y (ndarray): Shape(m,) target values.
        
        Returns:
            dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
            dj_db (scalar): The gradient of the cost w.r.t. the parameter b 
        """
        dZ =  activations[f"A{len(self.number_of_neurons_in_each_layer) - 1}"] - y
        gradients = {}

        for c in range (1, len(self.number_of_neurons_in_each_layer))[::-1]:
            gradients[f"dW{c}"] = 1/len(y) * np.dot(activations[f"A{c-1}"].T, dZ)
            gradients[f"db{c}"] = 1/len(y) * np.sum(dZ, axis = 0, keepdims = True)
            dZ = np.dot(dZ, parameters[f"W{c}"].T) * (activations[f"A{c - 1}"] * (1 - activations[f"A{c - 1}"]))
        return gradients
    

    #Fonction de mise à jour des paramètres
    def update_parameters(self, parameters, gradients, learning_rate):
        for c in range(1, len(self.number_of_neurons_in_each_layer)):
            parameters[f"W{c}"] -= learning_rate * gradients[f"dW{c}"]
            parameters[f"b{c}"] -= learning_rate * gradients[f"db{c}"]
        return parameters
    
    def train(self, X_train, y_train, num_iterations, learning_rate):
        parameters = self.initialize()
        for iteration in num_iterations:
            activations = self.predictions(X_train, parameters)
            cost = self.compute_cost(activations[f"A{len(self.number_of_neurons_in_each_layer) - 1}"], y_train)
            MLP.cost.append(cost)
            gradients = self.compute_gradient(y_train, activations, parameters)
            parameters = self.update_parameters(parameters, gradients, learning_rate)
            print(f"**Cost **: {cost}")
        MLP.parameters = parameters



mlp = MLP([2, 32, 32, 1])
parameters = mlp.initialize()
activations = (mlp.predictions(np.random.randn(100, 2)))
grad = mlp.compute_gradient(np.random.randn(100,1), activations, parameters)
for k, v in parameters.items():
    print(k, v.shape)
for k, v in activations.items():
    print(k, v.shape)

for k, v in grad.items():
    print(k, v.shape)
