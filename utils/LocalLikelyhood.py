import numpy as np
from scipy.stats import logistic

class LocallyWeightedLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, tau=1.0):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.tau = tau
        self.weights = None

    def _weighting_function(self, test_point, data_points):
        distances = np.linalg.norm(data_points - test_point, axis=1)
        return np.exp(- (distances ** 2) / (2 * self.tau ** 2))
    
    def _local_gradient_descent(self, X, y, weights):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        for _ in range(self.num_iterations):
            predictions = logistic.cdf(np.dot(X, self.weights))
            errors = y - predictions
            weighted_errors = weights * errors
            gradient = np.dot(X.T, weighted_errors) / n_samples
            self.weights += self.learning_rate * gradient

    def fit(self, X, y, x0):
        weights = self._weighting_function(x0, X)
        self._local_gradient_descent(X, y, weights)
    
    def predict_proba(self, X, x0):
        weights = self._weighting_function(x0, X)
        self._local_gradient_descent(X, x0, weights)
        return logistic.cdf(np.dot(X, self.weights))
    
    def predict(self, X, x0):
        probabilities = self.predict_proba(X, x0)
        return (probabilities >= 0.5).astype(int)
