import numpy as np

class LinearLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.biases = None

    def _softmax(self, z):
        exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 20))
        self.biases = np.zeros(20)
        y_encoded = np.eye(20)[y]

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.biases
            probabilities = self._softmax(linear_model)
            
            dw = (1 / n_samples) * np.dot(X.T, (probabilities - y_encoded))
            db = (1 / n_samples) * np.sum(probabilities - y_encoded, axis=0)
            self.weights -= self.learning_rate * dw
            self.biases -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.biases
        probabilities = self._softmax(linear_model)
        return np.argmax(probabilities, axis=1)
    
    def compute_nll(self, X, y):
        linear_model = np.dot(X, self.weights) + self.biases
        probabilities = self._softmax(linear_model)
        y_encoded = np.eye(probabilities.shape[1])[y]
        nll = -np.sum(y_encoded * np.log(probabilities))
        return nll
    
    def compute_aic(self, X, y):
        nll = self.compute_nll(X, y)
        k = np.prod(self.weights.shape) + len(self.biases)  # Total number of parameters
        aic = 2 * k + 2 * nll
        return aic
    
    def compute_bic(self, X, y):
        n_samples = X.shape[0]
        nll = self.compute_nll(X, y)
        k = np.prod(self.weights.shape) + len(self.biases)  # Total number of parameters
        bic = k * np.log(n_samples) + 2 * nll
        return bic

class RegularLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, reg_lambda=0.01):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.reg_lambda = reg_lambda
        self.weights = None
        self.biases = None

    def _softmax(self, z):
        exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 20))
        self.biases = np.zeros(20)
        y_encoded = np.eye(20)[y]

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.biases
            probabilities = self._softmax(linear_model)
            
            # Compute the gradient of the loss with regularization
            dw = (1 / n_samples) * np.dot(X.T, (probabilities - y_encoded)) - (self.reg_lambda * self.weights)
            db = (1 / n_samples) * np.sum(probabilities - y_encoded, axis=0)
            
            # Gradient descent parameter update
            self.weights -= self.learning_rate * dw
            self.biases -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.biases
        probabilities = self._softmax(linear_model)
        return np.argmax(probabilities, axis=1)
    
    def compute_nll(self, X, y):
        linear_model = np.dot(X, self.weights) + self.biases
        probabilities = self._softmax(linear_model)
        y_encoded = np.eye(probabilities.shape[1])[y]
        nll = -np.sum(y_encoded * np.log(probabilities))
        return nll
    
    def compute_aic(self, X, y):
        nll = self.compute_nll(X, y)
        k = np.prod(self.weights.shape) + len(self.biases)
        aic = 2 * k + 2 * nll
        return aic
    
    def compute_bic(self, X, y):
        n_samples = X.shape[0]
        nll = self.compute_nll(X, y)
        k = np.prod(self.weights.shape) + len(self.biases) 
        bic = k * np.log(n_samples) + 2 * nll
        return bic