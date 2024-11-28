import numpy as np

class LinearRegression:

    def __init__(self, ):
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
        W = np.random.randn(size= n)
        b = 0
        return W, b

    def predictions(self, X, W, b):
        """
        Compute the predictions of the model.
        Args:
            X (ndarray): Shape(m,) Input to the model.
            W (scalar): Weight of the model.
            b (scalar): Bias of the model.
        
        Returns:
            predictions (ndarray): Predicted values.
        """
        prediction = np.dot(X, W) + b
        return prediction

    def compute_gradient(self, predictions, X, y):

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

    
    def gradient_descent(self, X, y, learning_rate, num_iterations):
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
        n = X.shape[1]
        W, b = self.initialize_parameters(n) # Initialize parameters
        for iteration in num_iterations:
            y_predict = self.predictions(X, W, b) #Make predictions
            cost = (1/(2*len(y))) * np.sum((y_predict - y)**2) # Compute prediction's loss
            dj_dw, dj_db = self.compute_gradient(y_predict, X, y) # Compute gradient
            W, b = self.update_parameters(W, b, dj_dw, dj_db, learning_rate) # Update parameters
            return W, b
