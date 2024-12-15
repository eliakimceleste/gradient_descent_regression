import numpy as np

class LogisticRegression:

    def __init__(self):
        pass

    def initialize_parameters(n:int):
        """
        Initialize the parameters of the model.
        Args:
            n (int): Number of features.
        Returns:
            W (array): Initialized weights.
            b (scalar): Initialized bias.
        """
        W = np.random.randn(n) # Initialize W
        W= W.reshape(-1, 1) # Reshape W to (n, 1)
        b = 0
        return W, b

    def predict(self, X, W, b):
        """
        Compute the predictions of the model.
        Args:
            X (ndarray): Shape(m,) Input to the model.
            W (scalar): Weight of the model.
            b (scalar): Bias of the model.
        
        Returns:
            predictions (ndarray): Predicted values.
        """
        z = np.dot(X, W) + b
        prediction = 1 / (1 + np.exp(-z))
        return prediction

    def cost(self, predictions, y):
        """
        Compute the cost of the model.
        Args:
            predictions (array): predicted values.
            y (ndarray): Shape(m,) target values.
        
        Returns:
            cost (scalar): The cost of the model.
        """
        m = len(y)
        cost = (1/m) * np.sum(-y * np.log(predictions) - (1 - y) * np.log(1 - predictions))
        return cost
    
    def compute_gradient(self, predictions, X, y):

        """
        Compute the gradient of the cost function with respect to the parameters.
        Args:
            predictions (array): predicted values.
            X (ndarray): Shape(m,) Input to the model.
            y (ndarray): Shape(m,) target values.
        
        Returns:
            dj_dw (array): The gradient of the cost w.r.t. the parameters w
            dj_db (scalar): The gradient of the cost w.r.t. the parameter b 
        """
        
        dj_dw = 0
        dj_db = 0
        m = len(y)
        
        # dj_dw = np.mean(np.dot(X.T, (predictions - y)))
        dj_dw = (1/m) * (np.dot(X.T, (predictions - y)))
        dj_db = np.mean(predictions - y)
        return dj_dw, dj_db

    def update_parameters(self, W, b, dj_dw, dj_db, learning_rate):
        """
        Update the parameters using the gradients.
        Args:
            dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
            dj_db (scalar): The gradient of the cost w.r.t. the parameter b 
        
        Returns:

        """
        W = W - learning_rate * dj_dw
        b = b - learning_rate * dj_db
        return W, b

    
    def train(self, X, y, learning_rate, num_iterations):
        """
        Perform gradient descent to learn the parameters W and b.
        Args:
            X (ndarray): Shape(m,) Input to the model.
            y (ndarray): Shape(m,) target values.
            learning_rate (scalar): Learning rate.
            num_iterations (int): Number of iterations.
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            cost = []
        n = X.shape[1]
        W, b = self.initialize_parameters(n) # Initialize parameters
        for iteration in num_iterations:
            y_predict = self.predict(X, W, b) #Make predictions
            cost.append(self.cost(y_predict, y)) # Compute prediction's cost
            dj_dw, dj_db = self.compute_gradient(y_predict, X, y) # Compute gradient
            W, b = self.update_parameters(W, b, dj_dw, dj_db, learning_rate) # Update parameters
            return W, b, cost

a = np.random.randn(10)
a = a.reshape(-1, 1)
print(a, "\n", a.shape)